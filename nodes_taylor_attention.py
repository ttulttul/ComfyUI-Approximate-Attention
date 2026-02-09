from __future__ import annotations

import logging
import math
import os
import random
from typing import Dict, Any, Optional

import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.patcher_extension as patcher_extension

import taylor_attention
import hybrid_attention
import flux2_ttr
import flux2_ttr_controller
import sweep_utils

try:
    import comfy.sample as comfy_sample
    import comfy.samplers as comfy_samplers
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_sample = None
    comfy_samplers = None
    comfy_model_management = None

logger = logging.getLogger(__name__)

_TRAINING_CONFIG_KEYS = (
    "loss_config",
    "optimizer_config",
    "schedule_config",
    "logging_config",
)

_TRAINING_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    "loss_config": {
        "rmse_weight": 1.0,
        "cosine_weight": 1.0,
        "lpips_weight": 0.0,
        "huber_beta": 0.05,
    },
    "optimizer_config": {
        "learning_rate": 1e-4,
        "grad_clip_norm": 1.0,
        "alpha_lr_multiplier": 5.0,
        "phi_lr_multiplier": 1.0,
    },
    "schedule_config": {
        "target_ttr_ratio": 0.7,
        "lambda_eff": 1.0,
        "lambda_entropy": 0.1,
        "gumbel_temperature_start": 1.0,
        "gumbel_temperature_end": 0.5,
        "warmup_steps": 0,
    },
    "logging_config": {
        "comet_enabled": False,
        "comet_api_key": "",
        "comet_project_name": "ttr-distillation",
        "comet_workspace": "",
        "comet_experiment": "",
        "log_every": 10,
    },
}

_LEGACY_TRAINER_DEFAULTS = {
    "learning_rate": 1e-4,
    "grad_clip_norm": 1.0,
    "alpha_lr_multiplier": 5.0,
    "phi_lr_multiplier": 1.0,
    "huber_beta": 0.05,
    "comet_enabled": True,
    "comet_api_key": "",
    "comet_project_name": "ttr-distillation",
    "comet_workspace": "comet-workspace",
    "comet_experiment": "",
    "log_every": 50,
}

_CONTROLLER_COMET_EXPERIMENTS: dict[str, Any] = {}
_CONTROLLER_COMET_LOGGED_PARAM_KEYS: set[str] = set()


@io.comfytype(io_type="TTR_TRAINING_CONFIG")
class TTRTrainingConfig(io.ComfyTypeIO):
    Type = dict


@io.comfytype(io_type="TTR_CONTROLLER")
class TTRControllerType(io.ComfyTypeIO):
    Type = flux2_ttr_controller.TTRController


def _normalize_training_config(training_config: Optional[dict]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {key: {} for key in _TRAINING_CONFIG_KEYS}
    if not isinstance(training_config, dict):
        return normalized
    for key in _TRAINING_CONFIG_KEYS:
        section = training_config.get(key)
        if isinstance(section, dict):
            normalized[key] = dict(section)
    return normalized


def _float_or(value: Any, default: float) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _bool_or(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _string_or(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    return str(value)


def _default_comet_experiment_name() -> str:
    return sweep_utils.normalize_comet_experiment_key(f"exp{random.getrandbits(64)}")


def _extract_layer_idx(layer_key: str) -> Optional[int]:
    if ":" not in layer_key:
        return None
    _, idx_text = layer_key.split(":", 1)
    try:
        return int(idx_text)
    except Exception:
        return None


def _vae_input(name: str, *, tooltip: Optional[str] = None):
    # Comfy API variants use different capitalization for this type.
    for attr_name in ("Vae", "VAE"):
        vae_type = getattr(io, attr_name, None)
        if vae_type is not None and hasattr(vae_type, "Input"):
            return vae_type.Input(name, tooltip=tooltip)
    logger.warning("Flux2TTR: io.VAE is unavailable in this ComfyUI API build; using AnyType input for '%s'.", name)
    return io.AnyType.Input(name, tooltip=tooltip)


def _resolve_ttr_runtime_from_model(model_ttr) -> tuple[flux2_ttr.Flux2TTRRuntime, dict]:
    transformer_options = model_ttr.model_options.setdefault("transformer_options", {})
    cfg = transformer_options.get("flux2_ttr")
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        raise ValueError(
            "Flux2TTRControllerTrainer requires a model with trained TTR modules. Run Flux2TTRTrainer first."
        )

    runtime_id = cfg.get("runtime_id")
    runtime = flux2_ttr.get_runtime(runtime_id) if isinstance(runtime_id, str) else None
    if runtime is None:
        runtime = flux2_ttr._recover_runtime_from_config(cfg)
        if runtime is not None:
            new_runtime_id = flux2_ttr.register_runtime(runtime)
            cfg["runtime_id"] = new_runtime_id
    if runtime is None:
        raise ValueError(
            "Flux2TTRControllerTrainer requires a model with trained TTR modules. Run Flux2TTRTrainer first."
        )

    ready_layers = [layer_key for layer_key, ready in runtime.layer_ready.items() if ready and layer_key.startswith("single:")]
    if not ready_layers:
        raise ValueError(
            "Flux2TTRControllerTrainer requires a model with trained TTR modules. Run Flux2TTRTrainer first."
        )
    return runtime, cfg


def _build_training_config_payload(
    *,
    rmse_weight: float,
    cosine_weight: float,
    lpips_weight: float,
    huber_beta: float,
    learning_rate: float,
    grad_clip_norm: float,
    alpha_lr_multiplier: float,
    phi_lr_multiplier: float,
    target_ttr_ratio: float,
    lambda_eff: float,
    lambda_entropy: float,
    gumbel_temperature_start: float,
    gumbel_temperature_end: float,
    warmup_steps: int,
    comet_enabled: bool,
    comet_api_key: str,
    comet_project_name: str,
    comet_workspace: str,
    comet_experiment: str,
    log_every: int,
) -> dict[str, dict[str, Any]]:
    return {
        "loss_config": {
            "rmse_weight": float(rmse_weight),
            "cosine_weight": float(cosine_weight),
            "lpips_weight": float(lpips_weight),
            "huber_beta": float(huber_beta),
        },
        "optimizer_config": {
            "learning_rate": float(learning_rate),
            "grad_clip_norm": float(grad_clip_norm),
            "alpha_lr_multiplier": float(alpha_lr_multiplier),
            "phi_lr_multiplier": float(phi_lr_multiplier),
        },
        "schedule_config": {
            "target_ttr_ratio": float(target_ttr_ratio),
            "lambda_eff": max(0.0, float(lambda_eff)),
            "lambda_entropy": max(0.0, float(lambda_entropy)),
            "gumbel_temperature_start": float(gumbel_temperature_start),
            "gumbel_temperature_end": float(gumbel_temperature_end),
            "warmup_steps": int(warmup_steps),
        },
        "logging_config": {
            "comet_enabled": bool(comet_enabled),
            "comet_api_key": str(comet_api_key or ""),
            "comet_project_name": str(comet_project_name or "ttr-distillation"),
            "comet_workspace": str(comet_workspace or ""),
            "comet_experiment": sweep_utils.normalize_comet_experiment_key(comet_experiment or ""),
            "log_every": max(1, int(log_every)),
        },
    }


def _build_config(
    backend: str,
    P: int,
    min_tokens: int,
    max_feature_dim_R: int,
    block_size_q: int,
    block_size_k: int,
    eps: float,
    fallback_on_negative: bool,
    allow_cross_attention: bool,
    max_head_dim: int,
    sub_head_blocks: int,
    qk_normalize: bool,
    qk_norm_clip: float,
    qk_norm_power: float,
    qk_norm_sigma_max: float,
    scale_mul: float,
    force_fp32: bool,
    memory_reserve: bool,
    memory_reserve_factor: float,
    memory_reserve_log: bool,
    early_probe: bool,
    probe_samples: int,
    denom_fp32: bool,
    denom_fallback_frac_limit: float,
    fused_kernel: bool,
    fused_full_kernel: bool,
    fused_feature_chunk_size: int,
    fused_value_chunk_size: int,
    s_store_bf16: bool,
    taylor_sigma_max: float,
    taylor_layer_start: int,
    taylor_layer_end: int,
    auto_tune: bool,
    auto_tune_steps: int,
    auto_tune_candidates: int,
    auto_tune_quality_samples: int,
    auto_tune_seed: int,
    auto_tune_qk_norm_power_min: float,
    auto_tune_qk_norm_power_max: float,
    auto_tune_qk_norm_clip_min: float,
    auto_tune_qk_norm_clip_max: float,
    auto_tune_scale_mul_min: float,
    auto_tune_scale_mul_max: float,
    auto_tune_max_tokens: int,
    quality_check: bool,
    quality_check_samples: int,
    quality_check_log_every: int,
    log_shapes: bool,
    log_fallbacks: bool,
) -> Dict[str, Any]:
    if backend != "taylor":
        return {"enabled": False}
    return {
        "enabled": True,
        "P": int(P),
        "min_tokens": int(min_tokens),
        "max_feature_dim_R": int(max_feature_dim_R),
        "block_size_q": int(block_size_q),
        "block_size_k": int(block_size_k),
        "eps": float(eps),
        "fallback_on_negative": bool(fallback_on_negative),
        "allow_cross_attention": bool(allow_cross_attention),
        "max_head_dim": int(max_head_dim),
        "sub_head_blocks": int(sub_head_blocks),
        "qk_normalize": bool(qk_normalize),
        "qk_norm_clip": float(qk_norm_clip),
        "qk_norm_power": float(qk_norm_power),
        "qk_norm_sigma_max": float(qk_norm_sigma_max),
        "scale_mul": float(scale_mul),
        "force_fp32": bool(force_fp32),
        "memory_reserve": bool(memory_reserve),
        "memory_reserve_factor": float(memory_reserve_factor),
        "memory_reserve_log": bool(memory_reserve_log),
        "early_probe": bool(early_probe),
        "probe_samples": int(probe_samples),
        "denom_fp32": bool(denom_fp32),
        "denom_fallback_frac_limit": float(denom_fallback_frac_limit),
        "fused_kernel": bool(fused_kernel),
        "fused_full_kernel": bool(fused_full_kernel),
        "fused_feature_chunk_size": int(fused_feature_chunk_size),
        "fused_value_chunk_size": int(fused_value_chunk_size),
        "s_store_bf16": bool(s_store_bf16),
        "taylor_sigma_max": float(taylor_sigma_max),
        "taylor_layer_start": int(taylor_layer_start),
        "taylor_layer_end": int(taylor_layer_end),
        "auto_tune": bool(auto_tune),
        "auto_tune_steps": int(auto_tune_steps),
        "auto_tune_candidates": int(auto_tune_candidates),
        "auto_tune_quality_samples": int(auto_tune_quality_samples),
        "auto_tune_seed": int(auto_tune_seed),
        "auto_tune_qk_norm_power_min": float(auto_tune_qk_norm_power_min),
        "auto_tune_qk_norm_power_max": float(auto_tune_qk_norm_power_max),
        "auto_tune_qk_norm_clip_min": float(auto_tune_qk_norm_clip_min),
        "auto_tune_qk_norm_clip_max": float(auto_tune_qk_norm_clip_max),
        "auto_tune_scale_mul_min": float(auto_tune_scale_mul_min),
        "auto_tune_scale_mul_max": float(auto_tune_scale_mul_max),
        "auto_tune_max_tokens": int(auto_tune_max_tokens),
        "quality_check": bool(quality_check),
        "quality_check_samples": int(quality_check_samples),
        "quality_check_log_every": int(quality_check_log_every),
        "log_shapes": bool(log_shapes),
        "log_fallbacks": bool(log_fallbacks),
    }


def _build_hybrid_config(
    enabled: bool,
    local_window: int,
    local_window_min: int,
    local_window_max: int,
    local_window_sigma_low: float,
    local_window_sigma_high: float,
    local_chunk: int,
    prefix_tokens: int,
    global_dim: int,
    global_P: int,
    global_weight: float,
    global_sigma_low: float,
    global_sigma_high: float,
    global_scale_mul: float,
    global_norm_power: float,
    global_norm_clip: float,
    use_pca: bool,
    pca_samples: int,
    allow_cross_attention: bool,
    layer_start: int,
    layer_end: int,
    eps: float,
    force_fp32: bool,
    log_steps: bool,
    log_quality_stats: bool,
) -> Dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "local_window": int(local_window),
        "local_window_min": int(local_window_min),
        "local_window_max": int(local_window_max),
        "local_window_sigma_low": float(local_window_sigma_low),
        "local_window_sigma_high": float(local_window_sigma_high),
        "local_chunk": int(local_chunk),
        "prefix_tokens": int(prefix_tokens),
        "global_dim": int(global_dim),
        "global_P": int(global_P),
        "global_weight": float(global_weight),
        "global_sigma_low": float(global_sigma_low),
        "global_sigma_high": float(global_sigma_high),
        "global_scale_mul": float(global_scale_mul),
        "global_norm_power": float(global_norm_power),
        "global_norm_clip": float(global_norm_clip),
        "use_pca": bool(use_pca),
        "pca_samples": int(pca_samples),
        "allow_cross_attention": bool(allow_cross_attention),
        "layer_start": int(layer_start),
        "layer_end": int(layer_end),
        "eps": float(eps),
        "force_fp32": bool(force_fp32),
        "log_steps": bool(log_steps),
        "log_quality_stats": bool(log_quality_stats),
    }


class TaylorAttentionBackend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TaylorAttentionBackend",
            display_name="Taylor Attention Backend",
            category="advanced/attention",
            description="Toggle Taylor-approximated attention for large-token diffusion transformers.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("backend", options=["standard", "taylor"], default="taylor"),
                io.Int.Input("P", default=4, min=1, max=8, step=1, tooltip="Number of Taylor terms."),
                io.Int.Input("min_tokens", default=100, min=0, max=200000, step=1, tooltip="Enable Taylor only when tokens >= min_tokens."),
                io.Int.Input("max_feature_dim_R", default=370000, min=1, max=1000000000, step=1, tooltip="Safety cap for feature dimension R."),
                io.Int.Input("block_size_q", default=32, min=1, max=8192, step=1),
                io.Int.Input("block_size_k", default=16, min=1, max=8192, step=1),
                io.Float.Input("eps", default=1e-6, min=1e-12, max=1e-2, step=1e-6),
                io.Boolean.Input("fallback_on_negative", default=True, tooltip="Fallback if denominators are too small."),
                io.Boolean.Input("allow_cross_attention", default=True),
                io.Int.Input("max_head_dim", default=128, min=1, max=512, step=1),
                io.Int.Input("sub_head_blocks", default=4, min=1, max=16, step=1, tooltip="Split each head into sub-blocks to reduce feature dimension."),
                io.Boolean.Input("qk_normalize", default=False, tooltip="L2-normalize queries/keys before Taylor features."),
                io.Float.Input("qk_norm_clip", default=0.0, min=0.0, max=100.0, step=0.5, tooltip="Clip L2 norm of queries/keys (0 disables)."),
                io.Float.Input("qk_norm_power", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Soften q/k magnitude by dividing by ||q||^power."),
                io.Float.Input("qk_norm_sigma_max", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Only apply q/k normalization when sigma <= this value (0 disables)."),
                io.Float.Input("scale_mul", default=1.0, min=0.0, max=4.0, step=0.05, tooltip="Additional scale multiplier for q·k before Taylor."),
                io.Boolean.Input("force_fp32", default=False, tooltip="Accumulate Taylor features in fp32 for stability."),
                io.Boolean.Input("memory_reserve", default=True, tooltip="Ask ComfyUI to free VRAM for Taylor attention."),
                io.Float.Input("memory_reserve_factor", default=1.1, min=1.0, max=4.0, step=0.05, tooltip="Safety multiplier for reserved VRAM estimate."),
                io.Boolean.Input("memory_reserve_log", default=True, tooltip="Log reserved VRAM estimates."),
                io.Boolean.Input("early_probe", default=True, tooltip="Probe denominators before full Taylor compute."),
                io.Int.Input("probe_samples", default=16, min=1, max=64, step=1, tooltip="Queries sampled for early probe."),
                io.Boolean.Input("denom_fp32", default=True, tooltip="Compute denominators in fp32 for stability."),
                io.Float.Input("denom_fallback_frac_limit", default=0.0, min=0.0, max=1.0, step=0.0001, tooltip="Fallback only if denom <= eps exceeds this fraction (0 = any)."),
                io.Boolean.Input("fused_kernel", default=False, tooltip="Use Triton fused kernel to stream features (CUDA only)."),
                io.Boolean.Input("fused_full_kernel", default=False, tooltip="Use fully fused Triton kernel without Python feature loops (CUDA only)."),
                io.Int.Input("fused_feature_chunk_size", default=8192, min=128, max=200000, step=128, tooltip="Feature chunk size for fused kernel streaming."),
                io.Int.Input("fused_value_chunk_size", default=0, min=0, max=4096, step=32, tooltip="Value dimension chunk size (0 = full)."),
                io.Boolean.Input("s_store_bf16", default=False, tooltip="Store S chunks in bf16 to reduce memory."),
                io.Float.Input("taylor_sigma_max", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Only run Taylor when sigma <= this value (0 disables)."),
                io.Int.Input("taylor_layer_start", default=-1, min=-1, max=512, step=1, tooltip="Only run Taylor on block_index >= this value (-1 disables)."),
                io.Int.Input("taylor_layer_end", default=-1, min=-1, max=512, step=1, tooltip="Only run Taylor on block_index <= this value (-1 disables)."),
                io.Boolean.Input("auto_tune", default=False, tooltip="Auto-tune q/k scaling during early steps."),
                io.Int.Input("auto_tune_steps", default=1, min=0, max=4, step=1, tooltip="Number of steps to search for a better config."),
                io.Int.Input("auto_tune_candidates", default=8, min=1, max=32, step=1, tooltip="Number of candidates per step."),
                io.Int.Input("auto_tune_quality_samples", default=4, min=1, max=32, step=1, tooltip="Samples used per candidate for quality scoring."),
                io.Int.Input("auto_tune_seed", default=0, min=0, max=100000, step=1),
                io.Float.Input("auto_tune_qk_norm_power_min", default=0.2, min=0.0, max=1.0, step=0.05),
                io.Float.Input("auto_tune_qk_norm_power_max", default=0.7, min=0.0, max=1.0, step=0.05),
                io.Float.Input("auto_tune_qk_norm_clip_min", default=8.0, min=0.0, max=100.0, step=0.5),
                io.Float.Input("auto_tune_qk_norm_clip_max", default=20.0, min=0.0, max=100.0, step=0.5),
                io.Float.Input("auto_tune_scale_mul_min", default=0.8, min=0.0, max=4.0, step=0.05),
                io.Float.Input("auto_tune_scale_mul_max", default=1.0, min=0.0, max=4.0, step=0.05),
                io.Int.Input("auto_tune_max_tokens", default=512, min=0, max=8192, step=1, tooltip="Max tokens used during auto-tune (0 = full)."),
                io.Boolean.Input("quality_check", default=True, tooltip="Log a sampled accuracy check vs softmax."),
                io.Int.Input("quality_check_samples", default=16, min=1, max=64, step=1, tooltip="Number of sampled queries per call."),
                io.Int.Input("quality_check_log_every", default=1, min=1, max=1000, step=1, tooltip="Log every N Taylor calls."),
                io.Boolean.Input("log_shapes", default=True),
                io.Boolean.Input("log_fallbacks", default=True),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        backend: str,
        P: int,
        min_tokens: int,
        max_feature_dim_R: int,
        block_size_q: int,
        block_size_k: int,
        eps: float,
        fallback_on_negative: bool,
        allow_cross_attention: bool,
        max_head_dim: int,
        sub_head_blocks: int,
        qk_normalize: bool,
        qk_norm_clip: float,
        qk_norm_power: float,
        qk_norm_sigma_max: float,
        scale_mul: float,
        force_fp32: bool,
        memory_reserve: bool,
        memory_reserve_factor: float,
        memory_reserve_log: bool,
        early_probe: bool,
        probe_samples: int,
        denom_fp32: bool,
        denom_fallback_frac_limit: float,
        fused_kernel: bool,
        fused_full_kernel: bool,
        fused_feature_chunk_size: int,
        fused_value_chunk_size: int,
        s_store_bf16: bool,
        taylor_sigma_max: float,
        taylor_layer_start: int,
        taylor_layer_end: int,
        auto_tune: bool,
        auto_tune_steps: int,
        auto_tune_candidates: int,
        auto_tune_quality_samples: int,
        auto_tune_seed: int,
        auto_tune_qk_norm_power_min: float,
        auto_tune_qk_norm_power_max: float,
        auto_tune_qk_norm_clip_min: float,
        auto_tune_qk_norm_clip_max: float,
        auto_tune_scale_mul_min: float,
        auto_tune_scale_mul_max: float,
        auto_tune_max_tokens: int,
        quality_check: bool,
        quality_check_samples: int,
        quality_check_log_every: int,
        log_shapes: bool,
        log_fallbacks: bool,
    ) -> io.NodeOutput:
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})

        if backend == "taylor":
            prev_override = transformer_options.get("optimized_attention_override")
            transformer_options["taylor_attention_prev_override"] = prev_override
            transformer_options["optimized_attention_override"] = taylor_attention.taylor_attention_override
            transformer_options["taylor_attention"] = _build_config(
                backend,
                P,
                min_tokens,
                max_feature_dim_R,
                block_size_q,
                block_size_k,
                eps,
                fallback_on_negative,
                allow_cross_attention,
                max_head_dim,
                sub_head_blocks,
                qk_normalize,
                qk_norm_clip,
                qk_norm_power,
                qk_norm_sigma_max,
                scale_mul,
                force_fp32,
                memory_reserve,
                memory_reserve_factor,
                memory_reserve_log,
                early_probe,
                probe_samples,
                denom_fp32,
                denom_fallback_frac_limit,
                fused_kernel,
                fused_full_kernel,
                fused_feature_chunk_size,
                fused_value_chunk_size,
                s_store_bf16,
                taylor_sigma_max,
                taylor_layer_start,
                taylor_layer_end,
                auto_tune,
                auto_tune_steps,
                auto_tune_candidates,
                auto_tune_quality_samples,
                auto_tune_seed,
                auto_tune_qk_norm_power_min,
                auto_tune_qk_norm_power_max,
                auto_tune_qk_norm_clip_min,
                auto_tune_qk_norm_clip_max,
                auto_tune_scale_mul_min,
                auto_tune_scale_mul_max,
                auto_tune_max_tokens,
                quality_check,
                quality_check_samples,
                quality_check_log_every,
                log_shapes,
                log_fallbacks,
            )
            logger.info("Enabled Taylor attention backend.")
        else:
            if transformer_options.get("optimized_attention_override") is taylor_attention.taylor_attention_override:
                transformer_options["optimized_attention_override"] = transformer_options.pop("taylor_attention_prev_override", None)
            transformer_options.pop("taylor_attention", None)
            logger.info("Disabled Taylor attention backend.")

        return io.NodeOutput(m)


class HybridTaylorAttentionBackend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HybridTaylorAttentionBackend",
            display_name="Hybrid Taylor Attention Backend",
            category="advanced/attention",
            description="Hybrid local exact attention with global low-dim Taylor approximation (Flux RoPE-compatible).",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=False, tooltip="Enable hybrid attention override for Flux."),
                io.Int.Input("local_window", default=512, min=0, max=8192, step=1, tooltip="Default local window radius for exact attention (0 = full)."),
                io.Int.Input("local_window_min", default=512, min=0, max=8192, step=1, tooltip="Minimum local window when scheduling by sigma."),
                io.Int.Input("local_window_max", default=512, min=0, max=8192, step=1, tooltip="Maximum local window when scheduling by sigma."),
                io.Float.Input("local_window_sigma_low", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma below which local window is min (0 disables schedule)."),
                io.Float.Input("local_window_sigma_high", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma above which local window is max (0 disables schedule)."),
                io.Int.Input("local_chunk", default=256, min=1, max=1000000000, step=1, tooltip="Query chunk size for local attention."),
                io.Int.Input("prefix_tokens", default=0, min=0, max=8192, step=1, tooltip="Always include the first N tokens (e.g., text tokens) in local attention."),
                io.Int.Input("global_dim", default=16, min=1, max=128, step=1, tooltip="Projection dimension for global approximation."),
                io.Int.Input("global_P", default=2, min=1, max=4, step=1, tooltip="Taylor order for global approximation."),
                io.Float.Input("global_weight", default=0.1, min=0.0, max=4.0, step=0.01, tooltip="Scale applied to global approximation."),
                io.Float.Input("global_sigma_low", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma below which global weight is full."),
                io.Float.Input("global_sigma_high", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma above which global weight is 0 (0 disables ramp)."),
                io.Float.Input("global_scale_mul", default=1.0, min=0.0, max=4.0, step=0.01, tooltip="Scale multiplier for global q·k before Taylor."),
                io.Float.Input("global_norm_power", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Global q/k norm power (0 disables)."),
                io.Float.Input("global_norm_clip", default=0.0, min=0.0, max=100.0, step=0.5, tooltip="Global q/k norm clip (0 disables)."),
                io.Boolean.Input("use_pca", default=True, tooltip="Compute PCA projection for global approximation."),
                io.Int.Input("pca_samples", default=2048, min=0, max=65536, step=256, tooltip="Samples used for PCA projection (0 = full)."),
                io.Boolean.Input("allow_cross_attention", default=True, tooltip="Allow hybrid attention on cross-attention."),
                io.Int.Input("layer_start", default=-1, min=-1, max=512, step=1, tooltip="Only run hybrid on block_index >= this value (-1 disables)."),
                io.Int.Input("layer_end", default=-1, min=-1, max=512, step=1, tooltip="Only run hybrid on block_index <= this value (-1 disables)."),
                io.Float.Input("eps", default=1e-6, min=1e-12, max=1e-2, step=1e-6),
                io.Boolean.Input("force_fp32", default=True, tooltip="Use fp32 for global approximation."),
                io.Boolean.Input("log_steps", default=True, tooltip="Log hybrid attention stats per step."),
                io.Boolean.Input("log_quality_stats", default=False, tooltip="Log hybrid-vs-exact quality stats after sampling completes."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled: bool,
        local_window: int,
        local_window_min: int,
        local_window_max: int,
        local_window_sigma_low: float,
        local_window_sigma_high: float,
        local_chunk: int,
        prefix_tokens: int,
        global_dim: int,
        global_P: int,
        global_weight: float,
        global_sigma_low: float,
        global_sigma_high: float,
        global_scale_mul: float,
        global_norm_power: float,
        global_norm_clip: float,
        use_pca: bool,
        pca_samples: int,
        allow_cross_attention: bool,
        layer_start: int,
        layer_end: int,
        eps: float,
        force_fp32: bool,
        log_steps: bool,
        log_quality_stats: bool,
    ) -> io.NodeOutput:
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})

        if enabled:
            transformer_options["hybrid_taylor_attention"] = _build_hybrid_config(
                enabled,
                local_window,
                local_window_min,
                local_window_max,
                local_window_sigma_low,
                local_window_sigma_high,
                local_chunk,
                prefix_tokens,
                global_dim,
                global_P,
                global_weight,
                global_sigma_low,
                global_sigma_high,
                global_scale_mul,
                global_norm_power,
                global_norm_clip,
                use_pca,
                pca_samples,
                allow_cross_attention,
                layer_start,
                layer_end,
                eps,
                force_fp32,
                log_steps,
                log_quality_stats,
            )
            callback_key = "hybrid_taylor_attention"
            m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
            m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)
            m.add_callback_with_key(
                patcher_extension.CallbacksMP.ON_PRE_RUN,
                callback_key,
                hybrid_attention.pre_run_callback,
            )
            m.add_callback_with_key(
                patcher_extension.CallbacksMP.ON_CLEANUP,
                callback_key,
                hybrid_attention.cleanup_callback,
            )
        else:
            transformer_options.pop("hybrid_taylor_attention", None)
            callback_key = "hybrid_taylor_attention"
            m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
            m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)

        return io.NodeOutput(m)


class Flux2TTRTrainingParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Flux2TTRTrainingParameters",
            display_name="Flux2TTRTrainingParameters",
            category="advanced/attention",
            description="Shared Flux2TTR training hyperparameters for Phase-1 and Phase-2 nodes.",
            inputs=[
                io.Float.Input("rmse_weight", default=1.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("cosine_weight", default=1.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("lpips_weight", default=0.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("huber_beta", default=0.05, min=1e-6, max=10.0, step=1e-4),
                io.Float.Input("learning_rate", default=1e-4, min=1e-7, max=1.0, step=1e-7),
                io.Float.Input("grad_clip_norm", default=1.0, min=0.0, max=100.0, step=1e-3),
                io.Float.Input("alpha_lr_multiplier", default=5.0, min=0.0, max=100.0, step=1e-3),
                io.Float.Input("phi_lr_multiplier", default=1.0, min=0.0, max=100.0, step=1e-3),
                io.Float.Input("target_ttr_ratio", default=0.7, min=0.0, max=1.0, step=1e-3),
                io.Float.Input("lambda_eff", default=1.0, min=0.0, max=100.0, step=1e-3),
                io.Float.Input("lambda_entropy", default=0.1, min=0.0, max=1.0, step=1e-3),
                io.Float.Input("gumbel_temperature_start", default=1.0, min=1e-4, max=10.0, step=1e-3),
                io.Float.Input("gumbel_temperature_end", default=0.5, min=1e-4, max=10.0, step=1e-3),
                io.Int.Input("warmup_steps", default=0, min=0, max=1000000, step=1),
                io.Boolean.Input("comet_enabled", default=False),
                io.String.Input("comet_api_key", default="", multiline=False),
                io.String.Input("comet_project_name", default="ttr-distillation", multiline=False),
                io.String.Input("comet_workspace", default="", multiline=False),
                io.String.Input(
                    "comet_experiment",
                    default=_default_comet_experiment_name(),
                    multiline=False,
                    tooltip="Comet experiment key. Use a stable value to keep logging to one persistent experiment.",
                ),
                io.Int.Input("log_every", default=10, min=1, max=1000000, step=1),
            ],
            outputs=[TTRTrainingConfig.Output("training_config")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        rmse_weight: float,
        cosine_weight: float,
        lpips_weight: float,
        huber_beta: float,
        learning_rate: float,
        grad_clip_norm: float,
        alpha_lr_multiplier: float,
        phi_lr_multiplier: float,
        target_ttr_ratio: float,
        lambda_eff: float,
        lambda_entropy: float,
        gumbel_temperature_start: float,
        gumbel_temperature_end: float,
        warmup_steps: int,
        comet_enabled: bool,
        comet_api_key: str,
        comet_project_name: str,
        comet_workspace: str,
        comet_experiment: str,
        log_every: int,
    ) -> io.NodeOutput:
        training_config = _build_training_config_payload(
            rmse_weight=rmse_weight,
            cosine_weight=cosine_weight,
            lpips_weight=lpips_weight,
            huber_beta=huber_beta,
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            alpha_lr_multiplier=alpha_lr_multiplier,
            phi_lr_multiplier=phi_lr_multiplier,
            target_ttr_ratio=target_ttr_ratio,
            lambda_eff=lambda_eff,
            lambda_entropy=lambda_entropy,
            gumbel_temperature_start=gumbel_temperature_start,
            gumbel_temperature_end=gumbel_temperature_end,
            warmup_steps=warmup_steps,
            comet_enabled=comet_enabled,
            comet_api_key=comet_api_key,
            comet_project_name=comet_project_name,
            comet_workspace=comet_workspace,
            comet_experiment=comet_experiment,
            log_every=log_every,
        )
        return io.NodeOutput(training_config)


class Flux2TTRTrainer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Flux2TTRTrainer",
            display_name="Flux2TTRTrainer",
            category="advanced/attention",
            description="Phase-1: distill Flux TTR linear attention modules from native attention.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("latents"),
                io.Conditioning.Input("conditioning"),
                TTRTrainingConfig.Input("training_config", optional=True),
                io.Int.Input("steps", default=512, min=0, max=200000, step=1),
                io.Boolean.Input("training", default=True, tooltip="Train TTR layers by distillation when enabled."),
                io.Boolean.Input(
                    "training_preview_ttr",
                    default=True,
                    tooltip="When training, output TTR student attention for visual preview instead of teacher passthrough.",
                ),
                io.String.Input(
                    "checkpoint_path",
                    default="",
                    multiline=False,
                    tooltip="Checkpoint file to load/save TTR layer weights.",
                ),
                io.Int.Input(
                    "feature_dim",
                    default=256,
                    min=128,
                    max=8192,
                    step=256,
                    tooltip="Kernel feature dimension (must be a multiple of 256).",
                ),
                io.Int.Input(
                    "query_chunk_size",
                    default=256,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Query chunk size for kernel attention evaluation.",
                ),
                io.Int.Input(
                    "key_chunk_size",
                    default=1024,
                    min=1,
                    max=8192,
                    step=1,
                    tooltip="Key chunk size for kernel KV/Ksum accumulation.",
                ),
                io.Float.Input(
                    "landmark_fraction",
                    default=0.08,
                    min=0.01,
                    max=0.5,
                    step=1e-3,
                    tooltip="Fraction of image tokens used as landmarks for exact softmax residual.",
                ),
                io.Int.Input(
                    "landmark_min",
                    default=64,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Minimum landmark count used at low resolution.",
                ),
                io.Int.Input(
                    "landmark_max",
                    default=512,
                    min=1,
                    max=2048,
                    step=1,
                    tooltip="Maximum landmark count used at high resolution.",
                ),
                io.Int.Input(
                    "text_tokens_guess",
                    default=77,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip="Assumed number of text tokens at the start of sequence for landmark selection.",
                ),
                io.Float.Input(
                    "alpha_init",
                    default=0.1,
                    min=0.0,
                    max=10.0,
                    step=1e-3,
                    tooltip="Initial residual gate for landmark softmax branch.",
                ),
                io.Int.Input(
                    "training_query_token_cap",
                    default=128,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Max number of query tokens per replay sample; keys/values always stay full length.",
                ),
                io.Int.Input(
                    "replay_buffer_size",
                    default=8,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Replay buffer capacity per layer for distillation samples.",
                ),
                io.Boolean.Input(
                    "replay_offload_cpu",
                    default=True,
                    tooltip="Store replay samples on CPU (reduced precision) to reduce VRAM pressure.",
                ),
                io.Int.Input(
                    "replay_max_mb",
                    default=768,
                    min=64,
                    max=65536,
                    step=1,
                    tooltip="Global replay memory budget in MB across all layers.",
                ),
                io.Int.Input(
                    "train_steps_per_call",
                    default=1,
                    min=1,
                    max=32,
                    step=1,
                    tooltip="Number of replay optimization steps run per attention call.",
                ),
                io.Float.Input(
                    "readiness_threshold",
                    default=0.12,
                    min=0.0,
                    max=10.0,
                    step=1e-4,
                    tooltip="Enable student inference for a layer only when EMA loss is below this threshold.",
                ),
                io.Int.Input(
                    "readiness_min_updates",
                    default=24,
                    min=0,
                    max=100000,
                    step=1,
                    tooltip="Minimum replay updates before a layer can be marked ready.",
                ),
                io.Boolean.Input(
                    "enable_memory_reserve",
                    default=False,
                    tooltip="Call ComfyUI free_memory before HKR attention allocations (can offload aggressively).",
                ),
                io.Int.Input(
                    "layer_start",
                    default=-1,
                    min=-1,
                    max=512,
                    step=1,
                    tooltip="Only apply TTR to single blocks with index >= layer_start (-1 disables).",
                ),
                io.Int.Input(
                    "layer_end",
                    default=-1,
                    min=-1,
                    max=512,
                    step=1,
                    tooltip="Only apply TTR to single blocks with index <= layer_end (-1 disables).",
                ),
                io.Float.Input(
                    "cfg_scale",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=1e-3,
                    tooltip="Default CFG scale used when transformer_options does not provide guidance scale.",
                ),
                io.Int.Input(
                    "min_swap_layers",
                    default=1,
                    min=0,
                    max=512,
                    step=1,
                    tooltip="Training mode: minimum number of eligible single layers to swap to TTR per diffusion step.",
                ),
                io.Int.Input(
                    "max_swap_layers",
                    default=-1,
                    min=-1,
                    max=512,
                    step=1,
                    tooltip="Training mode: maximum swapped layers per diffusion step (-1 means all eligible layers).",
                ),
                io.Boolean.Input(
                    "inference_mixed_precision",
                    default=True,
                    tooltip="Use input dtype (bf16/fp16) for TTR inference on CUDA for speed.",
                ),
                io.String.Input(
                    "controller_checkpoint_path",
                    default="",
                    multiline=False,
                    tooltip="Optional Phase-2 controller checkpoint used for inference-time layer routing.",
                ),
            ],
            outputs=[io.Model.Output(), io.Float.Output("loss_value")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        latents,
        conditioning,
        training_config: Optional[dict],
        steps: int,
        training: bool,
        training_preview_ttr: bool,
        checkpoint_path: str,
        feature_dim: int,
        query_chunk_size: int,
        key_chunk_size: int,
        landmark_fraction: float,
        landmark_min: int,
        landmark_max: int,
        text_tokens_guess: int,
        alpha_init: float,
        training_query_token_cap: int,
        replay_buffer_size: int,
        replay_offload_cpu: bool,
        replay_max_mb: int,
        train_steps_per_call: int,
        readiness_threshold: float,
        readiness_min_updates: int,
        enable_memory_reserve: bool,
        layer_start: int,
        layer_end: int,
        cfg_scale: float,
        min_swap_layers: int,
        max_swap_layers: int,
        inference_mixed_precision: bool,
        controller_checkpoint_path: str,
    ) -> io.NodeOutput:
        del latents, conditioning
        feature_dim = flux2_ttr.validate_feature_dim(feature_dim)
        checkpoint_path = (checkpoint_path or "").strip()
        controller_checkpoint_path = (controller_checkpoint_path or "").strip()
        train_steps = int(steps)

        normalized_cfg = _normalize_training_config(training_config)
        loss_cfg = normalized_cfg["loss_config"]
        opt_cfg = normalized_cfg["optimizer_config"]
        schedule_cfg = normalized_cfg["schedule_config"]
        logging_cfg = normalized_cfg["logging_config"]
        has_training_config = isinstance(training_config, dict)

        if has_training_config:
            learning_rate = _float_or(opt_cfg.get("learning_rate"), _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["learning_rate"])
            grad_clip_norm = _float_or(opt_cfg.get("grad_clip_norm"), _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["grad_clip_norm"])
            alpha_lr_multiplier = _float_or(opt_cfg.get("alpha_lr_multiplier"), _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["alpha_lr_multiplier"])
            phi_lr_multiplier = _float_or(opt_cfg.get("phi_lr_multiplier"), _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["phi_lr_multiplier"])
            huber_beta = _float_or(loss_cfg.get("huber_beta"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["huber_beta"])
            comet_enabled = _bool_or(logging_cfg.get("comet_enabled"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_enabled"])
            comet_api_key = _string_or(logging_cfg.get("comet_api_key"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_api_key"])
            comet_project_name = _string_or(logging_cfg.get("comet_project_name"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_project_name"])
            comet_workspace = _string_or(logging_cfg.get("comet_workspace"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_workspace"])
            comet_experiment = _string_or(logging_cfg.get("comet_experiment"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_experiment"])
            comet_log_every = max(1, _int_or(logging_cfg.get("log_every"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["log_every"]))
        else:
            learning_rate = float(_LEGACY_TRAINER_DEFAULTS["learning_rate"])
            grad_clip_norm = float(_LEGACY_TRAINER_DEFAULTS["grad_clip_norm"])
            alpha_lr_multiplier = float(_LEGACY_TRAINER_DEFAULTS["alpha_lr_multiplier"])
            phi_lr_multiplier = float(_LEGACY_TRAINER_DEFAULTS["phi_lr_multiplier"])
            huber_beta = float(_LEGACY_TRAINER_DEFAULTS["huber_beta"])
            comet_enabled = bool(_LEGACY_TRAINER_DEFAULTS["comet_enabled"])
            comet_api_key = str(_LEGACY_TRAINER_DEFAULTS["comet_api_key"])
            comet_project_name = str(_LEGACY_TRAINER_DEFAULTS["comet_project_name"])
            comet_workspace = str(_LEGACY_TRAINER_DEFAULTS["comet_workspace"])
            comet_experiment = str(_LEGACY_TRAINER_DEFAULTS["comet_experiment"])
            comet_log_every = int(_LEGACY_TRAINER_DEFAULTS["log_every"])

        target_ttr_ratio = _float_or(
            schedule_cfg.get("target_ttr_ratio"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["target_ttr_ratio"],
        )
        lambda_eff = _float_or(
            schedule_cfg.get("lambda_eff"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["lambda_eff"],
        )
        lambda_entropy = _float_or(
            schedule_cfg.get("lambda_entropy"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["lambda_entropy"],
        )
        gumbel_temperature_start = _float_or(
            schedule_cfg.get("gumbel_temperature_start"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["gumbel_temperature_start"],
        )
        gumbel_temperature_end = _float_or(
            schedule_cfg.get("gumbel_temperature_end"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["gumbel_temperature_end"],
        )
        warmup_steps = max(0, _int_or(schedule_cfg.get("warmup_steps"), _TRAINING_CONFIG_DEFAULTS["schedule_config"]["warmup_steps"]))
        rmse_weight = _float_or(loss_cfg.get("rmse_weight"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["rmse_weight"])
        cosine_weight = _float_or(loss_cfg.get("cosine_weight"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["cosine_weight"])
        lpips_weight = _float_or(loss_cfg.get("lpips_weight"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["lpips_weight"])

        resolved_training_config = _build_training_config_payload(
            rmse_weight=rmse_weight,
            cosine_weight=cosine_weight,
            lpips_weight=lpips_weight,
            huber_beta=huber_beta,
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            alpha_lr_multiplier=alpha_lr_multiplier,
            phi_lr_multiplier=phi_lr_multiplier,
            target_ttr_ratio=target_ttr_ratio,
            lambda_eff=lambda_eff,
            lambda_entropy=lambda_entropy,
            gumbel_temperature_start=gumbel_temperature_start,
            gumbel_temperature_end=gumbel_temperature_end,
            warmup_steps=warmup_steps,
            comet_enabled=comet_enabled,
            comet_api_key=comet_api_key,
            comet_project_name=comet_project_name,
            comet_workspace=comet_workspace,
            comet_experiment=comet_experiment,
            log_every=comet_log_every,
        )

        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})

        prev_cfg = transformer_options.get("flux2_ttr")
        if isinstance(prev_cfg, dict):
            prev_runtime = prev_cfg.get("runtime_id")
            if isinstance(prev_runtime, str):
                flux2_ttr.unregister_runtime(prev_runtime)

        runtime = flux2_ttr.Flux2TTRRuntime(
            feature_dim=feature_dim,
            learning_rate=float(learning_rate),
            training=bool(training),
            steps=train_steps,
            scan_chunk_size=int(query_chunk_size),
            key_chunk_size=int(key_chunk_size),
            landmark_fraction=float(landmark_fraction),
            landmark_min=int(landmark_min),
            landmark_max=int(landmark_max),
            text_tokens_guess=int(text_tokens_guess),
            alpha_init=float(alpha_init),
            alpha_lr_multiplier=float(alpha_lr_multiplier),
            phi_lr_multiplier=float(phi_lr_multiplier),
            training_query_token_cap=int(training_query_token_cap),
            replay_buffer_size=int(replay_buffer_size),
            replay_offload_cpu=bool(replay_offload_cpu),
            replay_max_bytes=int(replay_max_mb) * 1024 * 1024,
            train_steps_per_call=int(train_steps_per_call),
            huber_beta=float(huber_beta),
            grad_clip_norm=float(grad_clip_norm),
            readiness_threshold=float(readiness_threshold),
            readiness_min_updates=int(readiness_min_updates),
            enable_memory_reserve=bool(enable_memory_reserve),
            layer_start=int(layer_start),
            layer_end=int(layer_end),
            cfg_scale=float(cfg_scale),
            min_swap_layers=int(min_swap_layers),
            max_swap_layers=int(max_swap_layers),
            inference_mixed_precision=bool(inference_mixed_precision),
            training_preview_ttr=bool(training_preview_ttr),
            comet_enabled=bool(comet_enabled),
            comet_project_name=str(comet_project_name or "ttr-distillation"),
            comet_workspace=str(comet_workspace or "comet-workspace"),
            comet_api_key=str(comet_api_key or ""),
            comet_log_every=int(comet_log_every),
        )
        runtime.register_layer_specs(flux2_ttr.infer_flux_single_layer_specs(m))

        if training:
            if checkpoint_path and os.path.isfile(checkpoint_path):
                logger.info("Flux2TTRTrainer: loading existing checkpoint before online distillation: %s", checkpoint_path)
                runtime.load_checkpoint(checkpoint_path)
            runtime.training_mode = True
            runtime.training_enabled = train_steps > 0
            runtime.training_steps_total = max(0, train_steps)
            runtime.steps_remaining = max(0, train_steps)
            runtime.training_updates_done = 0
            loss_value = float(runtime.last_loss) if not math.isnan(runtime.last_loss) else 0.0
        else:
            if not checkpoint_path:
                raise ValueError("Flux2TTRTrainer: checkpoint_path is required when training is disabled.")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Flux2TTRTrainer: checkpoint not found: {checkpoint_path}")
            runtime.load_checkpoint(checkpoint_path)
            runtime.training_mode = False
            runtime.training_enabled = False
            runtime.training_steps_total = 0
            runtime.steps_remaining = 0
            runtime.training_updates_done = 0
            loss_value = float(runtime.last_loss) if not math.isnan(runtime.last_loss) else 0.0

        runtime.cfg_scale = float(cfg_scale)
        runtime.min_swap_layers = max(0, int(min_swap_layers))
        runtime.max_swap_layers = int(max_swap_layers)

        controller = None
        if controller_checkpoint_path:
            if not os.path.isfile(controller_checkpoint_path):
                raise FileNotFoundError(f"Flux2TTRTrainer: controller checkpoint not found: {controller_checkpoint_path}")
            controller = flux2_ttr_controller.load_controller_checkpoint(controller_checkpoint_path, map_location="cpu")

        runtime_id = flux2_ttr.register_runtime(runtime)
        transformer_options["flux2_ttr"] = {
            "enabled": True,
            "runtime_id": runtime_id,
            "training": runtime.training_enabled,
            "training_mode": runtime.training_mode,
            "training_preview_ttr": runtime.training_preview_ttr,
            "comet_enabled": runtime.comet_enabled,
            "comet_project_name": runtime.comet_project_name,
            "comet_workspace": runtime.comet_workspace,
            "comet_log_every": int(runtime.comet_log_every),
            "training_steps_total": int(runtime.training_steps_total),
            "training_steps_remaining": int(runtime.steps_remaining),
            "learning_rate": float(learning_rate),
            "feature_dim": feature_dim,
            "query_chunk_size": int(query_chunk_size),
            "scan_chunk_size": int(query_chunk_size),
            "key_chunk_size": int(key_chunk_size),
            "landmark_fraction": float(landmark_fraction),
            "landmark_min": int(landmark_min),
            "landmark_max": int(landmark_max),
            "text_tokens_guess": int(text_tokens_guess),
            "alpha_init": float(alpha_init),
            "alpha_lr_multiplier": float(alpha_lr_multiplier),
            "phi_lr_multiplier": float(phi_lr_multiplier),
            "training_query_token_cap": int(training_query_token_cap),
            "replay_buffer_size": int(replay_buffer_size),
            "replay_offload_cpu": bool(replay_offload_cpu),
            "replay_max_bytes": int(replay_max_mb) * 1024 * 1024,
            "train_steps_per_call": int(train_steps_per_call),
            "huber_beta": float(huber_beta),
            "grad_clip_norm": float(grad_clip_norm),
            "readiness_threshold": float(readiness_threshold),
            "readiness_min_updates": int(readiness_min_updates),
            "enable_memory_reserve": bool(enable_memory_reserve),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "cfg_scale": float(cfg_scale),
            "min_swap_layers": int(min_swap_layers),
            "max_swap_layers": int(max_swap_layers),
            "inference_mixed_precision": bool(inference_mixed_precision),
            "max_safe_inference_loss": float(runtime.max_safe_inference_loss),
            "checkpoint_path": checkpoint_path,
            "controller_checkpoint_path": controller_checkpoint_path,
            "training_config": resolved_training_config,
        }
        if controller is not None:
            transformer_options["flux2_ttr"]["controller"] = controller

        callback_key = "flux2_ttr"
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_PRE_RUN,
            callback_key,
            flux2_ttr.pre_run_callback,
        )
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_CLEANUP,
            callback_key,
            flux2_ttr.cleanup_callback,
        )

        logger.info(
            (
                "Flux2TTRTrainer configured: training_mode=%s training_preview_ttr=%s comet_enabled=%s "
                "training_steps=%d feature_dim=%d q_chunk=%d k_chunk=%d landmarks=(fraction=%.4g,min=%d,max=%d) "
                "lr=%.6g alpha_lr_mul=%.4g phi_lr_mul=%.4g huber_beta=%.6g grad_clip=%.4g "
                "replay=%d replay_offload_cpu=%s replay_max_mb=%d train_steps_per_call=%d readiness=(%.6g,%d) reserve=%s layer_range=[%d,%d] "
                "cfg_scale=%.4g swap_layers=[%d,%d] mixed_precision=%s checkpoint=%s controller=%s loss=%.6g"
            ),
            training,
            bool(training_preview_ttr),
            bool(comet_enabled),
            train_steps,
            feature_dim,
            int(query_chunk_size),
            int(key_chunk_size),
            float(landmark_fraction),
            int(landmark_min),
            int(landmark_max),
            float(learning_rate),
            float(alpha_lr_multiplier),
            float(phi_lr_multiplier),
            float(huber_beta),
            float(grad_clip_norm),
            int(replay_buffer_size),
            bool(replay_offload_cpu),
            int(replay_max_mb),
            int(train_steps_per_call),
            float(readiness_threshold),
            int(readiness_min_updates),
            bool(enable_memory_reserve),
            int(layer_start),
            int(layer_end),
            float(cfg_scale),
            int(min_swap_layers),
            int(max_swap_layers),
            bool(inference_mixed_precision),
            checkpoint_path if checkpoint_path else "<none>",
            controller_checkpoint_path if controller_checkpoint_path else "<none>",
            float(loss_value),
        )
        return io.NodeOutput(m, float(loss_value))


class Flux2TTRControllerTrainer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        sampler_options = list(comfy_samplers.KSampler.SAMPLERS) if comfy_samplers is not None else ["euler"]
        scheduler_options = list(comfy_samplers.KSampler.SCHEDULERS) if comfy_samplers is not None else ["normal"]
        return io.Schema(
            node_id="Flux2TTRControllerTrainer",
            display_name="Flux2TTRControllerTrainer",
            category="advanced/attention",
            description="Phase-2: train a TTR controller using dual-path sampling and REINFORCE updates.",
            inputs=[
                io.Model.Input("model_original"),
                io.Model.Input("model_ttr"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                _vae_input("vae"),
                TTRTrainingConfig.Input("training_config"),
                io.Int.Input("steps", default=20, min=1, max=10000, step=1),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, step=1),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=1e-3),
                io.Combo.Input("sampler_name", options=sampler_options, default="euler"),
                io.Combo.Input("scheduler", options=scheduler_options, default="normal"),
                io.String.Input("checkpoint_path", default="", multiline=False),
                io.Int.Input("training_iterations", default=100, min=1, max=100000, step=1),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=1e-3),
                io.Boolean.Input("sigma_aware_training", default=True),
            ],
            outputs=[
                io.Image.Output("IMAGE_ORIGINAL"),
                io.Image.Output("IMAGE_PATCHED"),
                TTRControllerType.Output("CONTROLLER"),
            ],
            is_experimental=True,
        )

    @staticmethod
    def _require_sampling_stack() -> None:
        if comfy_sample is None or comfy_samplers is None:
            raise RuntimeError("Flux2TTRControllerTrainer requires ComfyUI sampling modules (comfy.sample/comfy.samplers).")

    @staticmethod
    def _split_latent_batch(latent: dict) -> list[dict]:
        samples = latent.get("samples")
        if not torch.is_tensor(samples) or samples.ndim < 4:
            raise ValueError("Flux2TTRControllerTrainer: latent must contain tensor key 'samples' with batch dimension.")
        batch_size = int(samples.shape[0])
        items: list[dict] = []
        for idx in range(batch_size):
            item = dict(latent)
            item["samples"] = samples[idx : idx + 1]
            noise_mask = latent.get("noise_mask")
            if torch.is_tensor(noise_mask) and noise_mask.shape[0] == batch_size:
                item["noise_mask"] = noise_mask[idx : idx + 1]
            batch_index = latent.get("batch_index")
            if isinstance(batch_index, (list, tuple)) and len(batch_index) == batch_size:
                item["batch_index"] = [batch_index[idx]]
            items.append(item)
        return items

    @staticmethod
    def _sample_model(
        *,
        model,
        latent: dict,
        positive,
        negative,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
    ) -> dict:
        latent_image = latent["samples"]
        latent_image = comfy_sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))
        batch_inds = latent.get("batch_index", None)
        noise = comfy_sample.prepare_noise(latent_image, int(seed), batch_inds)
        samples = comfy_sample.sample(
            model,
            noise,
            int(steps),
            float(cfg),
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=float(denoise),
            noise_mask=latent.get("noise_mask", None),
            disable_pbar=True,
            seed=int(seed),
        )
        out = dict(latent)
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return out

    @staticmethod
    def _decode_vae(vae, latent_dict: dict) -> torch.Tensor:
        latent_t = latent_dict["samples"]
        if getattr(latent_t, "is_nested", False):
            latent_t = latent_t.unbind()[0]
        images = vae.decode(latent_t)
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    @staticmethod
    def _to_lpips_bchw(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("Flux2TTRControllerTrainer: expected decoded images with shape [B,H,W,C] or [B,C,H,W].")
        if images.shape[-1] == 3:
            x = images.permute(0, 3, 1, 2).contiguous()
        elif images.shape[1] == 3:
            x = images
        else:
            raise ValueError("Flux2TTRControllerTrainer: decoded images must have 3 channels.")
        x = x.float()
        if float(x.min().item()) >= 0.0 and float(x.max().item()) <= 1.0:
            x = x * 2.0 - 1.0
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def _maybe_unload_model(model) -> None:
        if comfy_model_management is None:
            return
        try:
            unload_fn = getattr(comfy_model_management, "unload_model_clones", None)
            if callable(unload_fn):
                unload_fn(model)
                return
            free_memory = getattr(comfy_model_management, "free_memory", None)
            if callable(free_memory):
                device = getattr(model, "load_device", None)
                if device is None and hasattr(comfy_model_management, "get_torch_device"):
                    device = comfy_model_management.get_torch_device()
                if device is not None:
                    free_memory(1e30, device)
        except Exception as exc:
            logger.debug("Flux2TTRControllerTrainer: failed to unload model from VRAM (%s).", exc)

    @staticmethod
    def _empty_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _representative_sigma(model, scheduler: str, steps: int) -> float:
        if comfy_samplers is None:
            return 0.0
        try:
            model_sampling = model.get_model_object("model_sampling")
            sigmas = comfy_samplers.calculate_sigmas(model_sampling, scheduler, int(steps))
            if torch.is_tensor(sigmas) and sigmas.numel() > 0:
                return float(sigmas[0].item())
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _flatten_comet_params(prefix: str, payload: dict[str, Any], out: dict[str, Any]) -> None:
        for key, value in payload.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                Flux2TTRControllerTrainer._flatten_comet_params(full_key, value, out)
            elif isinstance(value, (list, tuple)):
                out[full_key] = ",".join(str(x) for x in value)
            else:
                out[full_key] = value

    @staticmethod
    def _sanitize_metric(value: Any) -> Optional[float]:
        try:
            v = float(value)
            if math.isfinite(v):
                return v
        except Exception:
            return None
        return None

    @staticmethod
    def _build_comet_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            sanitized = Flux2TTRControllerTrainer._sanitize_metric(value)
            if sanitized is None:
                continue
            payload[f"flux2ttr_controller/{key}"] = sanitized

        # Backward/forward compatibility: ensure eligible full-attention ratio
        # is always present under a stable key in Comet dashboards.
        eligible_key = "flux2ttr_controller/full_attn_eligible"
        if eligible_key not in payload:
            fallback = Flux2TTRControllerTrainer._sanitize_metric(
                metrics.get("full_attn_ratio", metrics.get("actual_full_attn_ratio"))
            )
            if fallback is not None:
                payload[eligible_key] = fallback
        return payload

    @staticmethod
    def _safe_comet_log_metrics(experiment, payload: dict[str, float], step: int) -> bool:
        if experiment is None or not payload:
            return False
        try:
            experiment.log_metrics(payload, step=int(step))
            return True
        except Exception as exc:
            logger.warning("Flux2TTRControllerTrainer: failed to log Comet metrics at step %d (%s).", int(step), exc)
            return False

    @staticmethod
    def _inference_tensor_count(module: torch.nn.Module) -> int:
        count = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            checker = getattr(tensor, "is_inference", None)
            if callable(checker):
                try:
                    if bool(checker()):
                        count += 1
                except Exception:
                    pass
        return int(count)

    @staticmethod
    def _maybe_start_comet(
        logging_cfg: dict[str, Any],
        training_config: dict[str, dict[str, Any]],
        run_config: dict[str, Any],
    ):
        enabled = _bool_or(logging_cfg.get("comet_enabled"), False)
        if not enabled:
            logger.info("Flux2TTRControllerTrainer: Comet logging disabled (training_config.logging_config.comet_enabled=false).")
            return None
        try:
            import comet_ml  # type: ignore

            api_key = _string_or(logging_cfg.get("comet_api_key"), "").strip() or os.getenv("COMET_API_KEY", "")
            if not api_key:
                logger.warning(
                    "Flux2TTRControllerTrainer: Comet logging enabled but no API key found "
                    "(set training_config.comet_api_key or COMET_API_KEY)."
                )
                return None
            project_name = _string_or(logging_cfg.get("comet_project_name"), "ttr-distillation")
            workspace = _string_or(logging_cfg.get("comet_workspace"), "").strip() or None
            raw_experiment_key = _string_or(logging_cfg.get("comet_experiment"), "")
            experiment_key = sweep_utils.normalize_comet_experiment_key(raw_experiment_key)
            logging_cfg["comet_experiment"] = experiment_key
            if raw_experiment_key != experiment_key:
                logger.info(
                    "Flux2TTRControllerTrainer: normalized comet_experiment from %r to %r.",
                    raw_experiment_key,
                    experiment_key,
                )

            cached = _CONTROLLER_COMET_EXPERIMENTS.get(experiment_key)
            if cached is not None:
                logger.info("Flux2TTRControllerTrainer: reusing persistent Comet experiment %s.", experiment_key)
                return cached

            experiment = comet_ml.start(
                api_key=api_key or None,
                project_name=project_name,
                workspace=workspace,
                experiment_key=experiment_key,
            )
            params: dict[str, Any] = {
                "flux2ttr_phase": "controller_train",
                "project_name": project_name,
                "workspace": workspace or "",
                "comet_experiment": experiment_key,
            }
            Flux2TTRControllerTrainer._flatten_comet_params("training_config", training_config, params)
            Flux2TTRControllerTrainer._flatten_comet_params("run_config", run_config, params)
            if experiment_key not in _CONTROLLER_COMET_LOGGED_PARAM_KEYS:
                experiment.log_parameters(params)
                _CONTROLLER_COMET_LOGGED_PARAM_KEYS.add(experiment_key)
            _CONTROLLER_COMET_EXPERIMENTS[experiment_key] = experiment
            logger.info(
                "Flux2TTRControllerTrainer: Comet logging enabled (project=%s workspace=%s experiment=%s).",
                project_name,
                workspace or "<none>",
                experiment_key,
            )
            return experiment
        except Exception as exc:
            logger.warning("Flux2TTRControllerTrainer: failed to initialize Comet logging (%s).", exc)
            return None

    @classmethod
    def execute(
        cls,
        model_original,
        model_ttr,
        positive,
        negative,
        latent,
        vae,
        training_config: dict,
        steps: int,
        seed: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        checkpoint_path: str,
        training_iterations: int,
        denoise: float,
        sigma_aware_training: bool,
    ) -> io.NodeOutput:
        cls._require_sampling_stack()
        runtime, flux_cfg = _resolve_ttr_runtime_from_model(model_ttr)
        runtime.training_mode = False
        runtime.training_enabled = False
        runtime.steps_remaining = 0
        flux_cfg["training_mode"] = False
        flux_cfg["training"] = False

        layer_index_to_key: dict[int, str] = {}
        for layer_key in runtime.layer_specs:
            if not layer_key.startswith("single:"):
                continue
            idx = _extract_layer_idx(layer_key)
            if idx is not None and idx >= 0:
                layer_index_to_key[idx] = layer_key
        if not layer_index_to_key:
            for layer_key in runtime.layer_ready:
                idx = _extract_layer_idx(layer_key)
                if idx is not None and idx >= 0:
                    layer_index_to_key[idx] = layer_key
        layer_indices = sorted(layer_index_to_key.keys())
        if not layer_indices:
            raise ValueError("Flux2TTRControllerTrainer: unable to infer eligible TTR layer count from model_ttr.")
        num_layers = max(layer_indices) + 1

        ready_layer_indices: list[int] = []
        for idx in layer_indices:
            layer_key = layer_index_to_key[idx]
            if runtime._refresh_layer_ready(layer_key):
                ready_layer_indices.append(idx)
        if not ready_layer_indices:
            raise ValueError(
                "Flux2TTRControllerTrainer requires readiness-qualified TTR layers. "
                "Run Flux2TTRTrainer longer or lower readiness_threshold."
            )
        ttr_eligible_mask_cpu = torch.zeros(num_layers, dtype=torch.bool)
        ttr_eligible_mask_cpu[ready_layer_indices] = True
        forced_full_layer_count = int((~ttr_eligible_mask_cpu).sum().item())
        logger.info(
            "Flux2TTRControllerTrainer layer readiness: eligible_ttr=%d/%d forced_full=%d",
            int(len(ready_layer_indices)),
            int(num_layers),
            forced_full_layer_count,
        )

        normalized_cfg = _normalize_training_config(training_config)
        loss_cfg = normalized_cfg["loss_config"]
        schedule_cfg = normalized_cfg["schedule_config"]
        logging_cfg = normalized_cfg["logging_config"]
        gumbel_start = _float_or(
            schedule_cfg.get("gumbel_temperature_start"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["gumbel_temperature_start"],
        )
        gumbel_end = _float_or(
            schedule_cfg.get("gumbel_temperature_end"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["gumbel_temperature_end"],
        )
        lambda_entropy = _float_or(
            schedule_cfg.get("lambda_entropy"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["lambda_entropy"],
        )
        log_every = max(1, _int_or(logging_cfg.get("log_every"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["log_every"]))

        checkpoint_path = (checkpoint_path or "").strip()
        if checkpoint_path and os.path.isfile(checkpoint_path):
            controller = flux2_ttr_controller.load_controller_checkpoint(checkpoint_path, map_location="cpu")
        else:
            controller = flux2_ttr_controller.TTRController(num_layers=num_layers)
        if int(controller.num_layers) != int(num_layers):
            raise ValueError(
                f"Flux2TTRControllerTrainer: controller num_layers={controller.num_layers} does not match model layers={num_layers}."
            )

        device = getattr(model_ttr, "load_device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ComfyUI may execute nodes under inference_mode; normalize the loaded
        # controller into a trainable clone before constructing the trainer/wrapper.
        controller = flux2_ttr_controller.ControllerTrainer._rebuild_controller_trainable_copy(controller)
        sigma_aware_training = bool(sigma_aware_training)

        trainer = flux2_ttr_controller.ControllerTrainer(
            controller=controller,
            training_config=normalized_cfg,
            lambda_entropy=float(lambda_entropy),
            device=device,
        )
        # ControllerTrainer may rebuild a trainable controller if the incoming module
        # was created under inference mode. Ensure runtime/wrapper/checkpoint all
        # use the trainer-owned instance.
        controller = trainer.controller
        controller_device, _ = controller._input_device_dtype()
        training_wrapper = flux2_ttr_controller.TrainingControllerWrapper(
            controller,
            temperature=float(gumbel_start),
            ready_mask=ttr_eligible_mask_cpu.to(device=controller_device, dtype=torch.float32),
        )
        flux_cfg["controller"] = training_wrapper if sigma_aware_training else controller
        flux_cfg["controller_threshold"] = 0.5
        logger.info(
            "Flux2TTRControllerTrainer routing mode: sigma_aware_training=%s",
            sigma_aware_training,
        )
        if checkpoint_path and os.path.isfile(checkpoint_path):
            try:
                payload = flux2_ttr_controller.load_controller_training_state(checkpoint_path, map_location="cpu")
                trainer.restore_training_state(payload)
                has_optimizer_state = (
                    "optimizer_state_dict" in payload and payload["optimizer_state_dict"] is not None
                )
                has_baseline_state = (
                    ("reward_baseline" in payload and payload["reward_baseline"] is not None)
                    or ("reward_count" in payload and payload["reward_count"] is not None)
                )
                if has_optimizer_state or has_baseline_state:
                    logger.info(
                        "Flux2TTR controller: restored training state "
                        "(reward_baseline=%.4f, reward_count=%d, optimizer restored=%s)",
                        trainer.reward_baseline,
                        trainer.reward_count,
                        has_optimizer_state,
                    )
                else:
                    logger.info(
                        "Flux2TTR controller: no persisted training state in checkpoint. Starting fresh optimizer/baseline."
                    )
            except Exception as exc:
                logger.info(
                    "Flux2TTR controller: no training state to restore (%s). Starting fresh.",
                    exc,
                )
        latent_items = cls._split_latent_batch(latent)
        sigma_value = cls._representative_sigma(model_ttr, scheduler=scheduler, steps=int(steps))
        total_iterations = max(1, int(training_iterations))
        total_steps = total_iterations * max(1, len(latent_items))
        checkpoint_every = max(1, int(flux2_ttr_controller.DEFAULT_CONTROLLER_CHECKPOINT_EVERY))
        comet_experiment = cls._maybe_start_comet(
            logging_cfg,
            normalized_cfg,
            run_config={
                "steps": int(steps),
                "seed": int(seed),
                "cfg": float(cfg),
                "sampler_name": str(sampler_name),
                "scheduler": str(scheduler),
                "training_iterations": int(total_iterations),
                "latent_batch_size": int(len(latent_items)),
                "total_steps": int(total_steps),
                "denoise": float(denoise),
                "controller_num_layers": int(controller.num_layers),
                "eligible_ttr_layers": int(len(ready_layer_indices)),
                "forced_full_layers": int(forced_full_layer_count),
                "sigma_aware_training": bool(sigma_aware_training),
                "checkpoint_every": int(checkpoint_every),
                "checkpoint_path": checkpoint_path,
                "device": str(device),
                "comet_experiment": _string_or(logging_cfg.get("comet_experiment"), ""),
            },
        )
        lpips_enabled = _float_or(loss_cfg.get("lpips_weight"), 0.0) > 0

        global_step = 0
        last_original_images: Optional[torch.Tensor] = None
        last_patched_images: Optional[torch.Tensor] = None

        try:
            for iteration in range(total_iterations):
                progress = float(iteration) / float(max(1, total_iterations - 1))
                gumbel_temperature = float(gumbel_start + (gumbel_end - gumbel_start) * progress)

                iter_original_images: list[torch.Tensor] = []
                iter_patched_images: list[torch.Tensor] = []

                for latent_idx, latent_item in enumerate(latent_items):
                    sample_seed = int(seed) + int(global_step)
                    latent_sample = latent_item["samples"]
                    width = int(latent_sample.shape[-1] * 8)
                    height = int(latent_sample.shape[-2] * 8)

                    teacher_latent = cls._sample_model(
                        model=model_original,
                        latent=latent_item,
                        positive=positive,
                        negative=negative,
                        seed=sample_seed,
                        steps=int(steps),
                        cfg=float(cfg),
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        denoise=float(denoise),
                    )
                    teacher_image = cls._decode_vae(vae, teacher_latent)

                    cls._maybe_unload_model(model_original)
                    cls._empty_cuda_cache()

                    if sigma_aware_training:
                        training_wrapper.reset()
                        training_wrapper.temperature = float(gumbel_temperature)
                        flux_cfg.pop("controller_mask_override", None)
                        flux_cfg["controller"] = training_wrapper
                        student_latent = cls._sample_model(
                            model=model_ttr,
                            latent=latent_item,
                            positive=positive,
                            negative=negative,
                            seed=sample_seed,
                            steps=int(steps),
                            cfg=float(cfg),
                            sampler_name=sampler_name,
                            scheduler=scheduler,
                            denoise=float(denoise),
                        )
                        if not training_wrapper.step_records:
                            raise RuntimeError(
                                "Flux2TTRControllerTrainer: no per-step controller records collected during sigma-aware sampling."
                            )
                        actual_full_attn_ratio_eligible = float(training_wrapper.mean_full_attn_eligible())
                        actual_full_attn_ratio_overall = float(training_wrapper.mean_full_attn_overall())
                        expected_full_attn_ratio_eligible = float(training_wrapper.mean_expected_full_attn_eligible())
                        expected_full_attn_ratio_overall = float(training_wrapper.mean_expected_full_attn_overall())
                    else:
                        with torch.no_grad():
                            logits = controller(
                                sigma=sigma_value,
                                cfg_scale=float(cfg),
                                width=width,
                                height=height,
                            )
                            sampled_mask = controller.sample_training_mask(
                                logits,
                                temperature=gumbel_temperature,
                                hard=True,
                            )
                        sampled_mask = (sampled_mask.detach() > 0.5).to(dtype=torch.float32)
                        effective_mask = sampled_mask.clone()
                        ttr_eligible_mask = ttr_eligible_mask_cpu.to(device=effective_mask.device)
                        effective_mask[~ttr_eligible_mask] = 1.0
                        actual_full_attn_ratio_overall = float(effective_mask.mean().item())
                        eligible_count = int(ttr_eligible_mask.sum().item())
                        if eligible_count <= 0:
                            raise RuntimeError(
                                "Flux2TTRControllerTrainer: no eligible layers available for controller penalty computation."
                            )
                        actual_full_attn_ratio_eligible = float(effective_mask[ttr_eligible_mask].mean().item())
                        flux_cfg["controller_mask_override"] = effective_mask.detach().cpu()

                        try:
                            student_latent = cls._sample_model(
                                model=model_ttr,
                                latent=latent_item,
                                positive=positive,
                                negative=negative,
                                seed=sample_seed,
                                steps=int(steps),
                                cfg=float(cfg),
                                sampler_name=sampler_name,
                                scheduler=scheduler,
                                denoise=float(denoise),
                            )
                        finally:
                            flux_cfg.pop("controller_mask_override", None)

                    student_image = cls._decode_vae(vae, student_latent)
                    teacher_rgb = cls._to_lpips_bchw(teacher_image)
                    student_rgb = cls._to_lpips_bchw(student_image)
                    quality_loss, quality_metrics = trainer.compute_loss(
                        teacher_latent=teacher_latent["samples"],
                        student_latent=student_latent["samples"],
                        actual_full_attn_ratio=actual_full_attn_ratio_eligible,
                        teacher_rgb=teacher_rgb if lpips_enabled else None,
                        student_rgb=student_rgb if lpips_enabled else None,
                        include_efficiency_penalty=False,
                    )
                    reward = -float(quality_loss.detach().item())

                    if sigma_aware_training:
                        entropy_value = float(training_wrapper.mean_entropy())
                        target_ttr_ratio = float(trainer.target_ttr_ratio)
                        target_full_attn_ratio = float(
                            trainer._target_full_attn_ratio_from_ttr_ratio(target_ttr_ratio)
                        )
                        efficiency_penalty = max(0.0, actual_full_attn_ratio_eligible - target_full_attn_ratio)
                        efficiency_penalty_weighted = float(trainer.lambda_eff * efficiency_penalty)
                        entropy_bonus = float(trainer.lambda_entropy * entropy_value)
                        reward_value = float(reward - efficiency_penalty_weighted + entropy_bonus)
                        baselined_reward = float(reward_value - trainer.reward_baseline)
                        trainer._update_reward_baseline(reward_value)

                        sigma_max = max(
                            1e-8,
                            max(float(rec["sigma"]) for rec in training_wrapper.step_records),
                        )
                        with torch.inference_mode(False):
                            with torch.enable_grad():
                                policy_objective_t = training_wrapper.sigma_weighted_log_prob_recompute(
                                    cfg_scale=float(cfg),
                                    width=int(width),
                                    height=int(height),
                                    sigma_max=sigma_max,
                                    eligible_mask=ttr_eligible_mask_cpu,
                                )
                                policy_loss_t = -float(baselined_reward) * policy_objective_t
                        if not bool(policy_loss_t.requires_grad):
                            repaired = 0
                            with torch.inference_mode(False):
                                for p in trainer.controller.parameters():
                                    if not bool(p.requires_grad):
                                        p.requires_grad_(True)
                                        repaired += 1
                            if repaired > 0:
                                with torch.inference_mode(False):
                                    with torch.enable_grad():
                                        policy_objective_t = training_wrapper.sigma_weighted_log_prob_recompute(
                                            cfg_scale=float(cfg),
                                            width=int(width),
                                            height=int(height),
                                            sigma_max=sigma_max,
                                            eligible_mask=ttr_eligible_mask_cpu,
                                        )
                                        policy_loss_t = -float(baselined_reward) * policy_objective_t
                                if bool(policy_loss_t.requires_grad):
                                    logger.info(
                                        "Flux2TTRControllerTrainer: restored detached sigma-aware policy loss by re-enabling requires_grad on %d controller params.",
                                        int(repaired),
                                    )
                            param_list = list(trainer.controller.parameters())
                            trainable_params = int(sum(1 for p in param_list if bool(p.requires_grad)))
                            inference_params = cls._inference_tensor_count(trainer.controller)
                            probe_requires_grad: Optional[bool] = None
                            probe_error = ""
                            try:
                                probe_sigma = float(training_wrapper.step_records[0]["sigma"])
                                with torch.inference_mode(False):
                                    with torch.enable_grad():
                                        probe_logits = trainer.controller(
                                            sigma=probe_sigma,
                                            cfg_scale=float(cfg),
                                            width=int(width),
                                            height=int(height),
                                        )
                                if probe_logits.ndim > 1:
                                    probe_logits = probe_logits.reshape(-1)
                                probe_requires_grad = bool(probe_logits.requires_grad)
                            except Exception as exc:
                                probe_error = str(exc)
                            logger.warning(
                                "Flux2TTRControllerTrainer: sigma-aware policy loss still detached after recompute; "
                                "skipping optimizer step for this sample. "
                                "diag: trainable_params=%d/%d inference_tensors=%d probe_requires_grad=%s probe_error=%s repaired_params=%d policy_objective_requires_grad=%s policy_loss_requires_grad=%s recompute_debug=%s",
                                trainable_params,
                                int(len(param_list)),
                                inference_params,
                                probe_requires_grad,
                                probe_error or "<none>",
                                int(repaired),
                                bool(policy_objective_t.requires_grad),
                                bool(policy_loss_t.requires_grad),
                                str(getattr(training_wrapper, "last_recompute_debug", {})),
                            )
                        with torch.inference_mode(False):
                            with torch.enable_grad():
                                trainer.optimizer.zero_grad(set_to_none=True)
                                if bool(policy_loss_t.requires_grad):
                                    policy_loss_t.backward()
                                    if trainer.grad_clip_norm > 0:
                                        torch.nn.utils.clip_grad_norm_(trainer.controller.parameters(), trainer.grad_clip_norm)
                                    trainer.optimizer.step()

                        step_ttr = training_wrapper.per_step_ttr_ratio()
                        sigma_max_val = max((s for s, _ in step_ttr), default=1.0)
                        high_ttrs = [f for s, f in step_ttr if s > 0.5 * sigma_max_val]
                        low_ttrs = [f for s, f in step_ttr if s < 0.2 * sigma_max_val]
                        ttr_ratio_high = float(sum(high_ttrs) / max(1, len(high_ttrs)))
                        ttr_ratio_low = float(sum(low_ttrs) / max(1, len(low_ttrs)))
                        ttr_ratio_spread = float(ttr_ratio_high - ttr_ratio_low)
                        trajectory_sigma = (
                            float(sum(s for s, _ in step_ttr) / max(1, len(step_ttr)))
                            if step_ttr
                            else float(sigma_value)
                        )

                        reinforce_metrics = {
                            "policy_loss": float(policy_loss_t.detach().item()),
                            "total_loss": float(policy_loss_t.detach().item()),
                            "efficiency_penalty": float(efficiency_penalty),
                            "efficiency_penalty_weighted": float(efficiency_penalty_weighted),
                            "reward": float(reward_value),
                            "reward_quality": float(reward),
                            "reward_baseline": float(trainer.reward_baseline),
                            "baselined_reward": float(baselined_reward),
                            "entropy": float(entropy_value),
                            "entropy_bonus": float(entropy_bonus),
                            "lambda_eff": float(trainer.lambda_eff),
                            "lambda_entropy": float(trainer.lambda_entropy),
                            "target_ttr_ratio": float(target_ttr_ratio),
                            "target_full_attn_ratio": float(target_full_attn_ratio),
                            "actual_full_attn_ratio": float(actual_full_attn_ratio_eligible),
                            "actual_full_attn_ratio_overall": float(actual_full_attn_ratio_overall),
                            "expected_full_attn_ratio": float(expected_full_attn_ratio_eligible),
                            "expected_full_attn_ratio_overall": float(expected_full_attn_ratio_overall),
                            "ttr_ratio_high_sigma": float(ttr_ratio_high),
                            "ttr_ratio_low_sigma": float(ttr_ratio_low),
                            "ttr_ratio_spread": float(ttr_ratio_spread),
                            "steps_per_trajectory": float(len(step_ttr)),
                            "sigma": float(trajectory_sigma),
                        }
                    else:
                        reinforce_metrics = trainer.reinforce_step(
                            sigma=sigma_value,
                            cfg_scale=float(cfg),
                            width=width,
                            height=height,
                            sampled_mask=effective_mask,
                            reward=reward,
                            actual_full_attn_ratio=actual_full_attn_ratio_eligible,
                            eligible_layer_mask=ttr_eligible_mask,
                            actual_full_attn_ratio_overall=actual_full_attn_ratio_overall,
                        )
                        expected_full_attn_ratio_eligible = float(
                            reinforce_metrics.get("expected_full_attn_ratio", actual_full_attn_ratio_eligible)
                        )
                        expected_full_attn_ratio_overall = float(
                            reinforce_metrics.get(
                                "expected_full_attn_ratio_overall",
                                actual_full_attn_ratio_overall,
                            )
                        )
                        target_ttr_ratio = float(reinforce_metrics.get("target_ttr_ratio", trainer.target_ttr_ratio))
                        target_full_attn_ratio = float(
                            reinforce_metrics.get(
                                "target_full_attn_ratio",
                                trainer._target_full_attn_ratio_from_ttr_ratio(target_ttr_ratio),
                            )
                        )

                    full_attn_ratio_eligible = float(
                        reinforce_metrics.get("actual_full_attn_ratio", actual_full_attn_ratio_eligible)
                    )
                    full_attn_ratio_overall = float(
                        reinforce_metrics.get("actual_full_attn_ratio_overall", actual_full_attn_ratio_overall)
                    )
                    metrics = {
                        "loss": float(quality_loss.detach().item()),
                        "rmse": float(quality_metrics["rmse"]),
                        "cosine_distance": float(quality_metrics["cosine_distance"]),
                        "lpips": float(quality_metrics["lpips"]),
                        "efficiency_penalty": float(reinforce_metrics["efficiency_penalty"]),
                        "mask_mean": full_attn_ratio_eligible,
                        "probs_mean": expected_full_attn_ratio_eligible,
                        "full_attn_ratio": full_attn_ratio_eligible,
                        "ttr_ratio": float(max(0.0, 1.0 - full_attn_ratio_eligible)),
                        "expected_full_attn_ratio": expected_full_attn_ratio_eligible,
                        "expected_ttr_ratio": float(max(0.0, 1.0 - expected_full_attn_ratio_eligible)),
                        "full_attn_eligible": full_attn_ratio_eligible,
                        "ttr_eligible": float(max(0.0, 1.0 - full_attn_ratio_eligible)),
                        "full_attn_overall": full_attn_ratio_overall,
                        "ttr_overall": float(max(0.0, 1.0 - full_attn_ratio_overall)),
                        "expected_full_attn_eligible": expected_full_attn_ratio_eligible,
                        "expected_ttr_eligible": float(max(0.0, 1.0 - expected_full_attn_ratio_eligible)),
                        "expected_full_attn_overall": expected_full_attn_ratio_overall,
                        "expected_ttr_overall": float(max(0.0, 1.0 - expected_full_attn_ratio_overall)),
                        "reward": float(reinforce_metrics.get("reward", reward)),
                        "reward_quality": float(reinforce_metrics.get("reward_quality", reward)),
                        "baselined_reward": float(reinforce_metrics.get("baselined_reward", 0.0)),
                        "reward_baseline": float(reinforce_metrics["reward_baseline"]),
                        "policy_loss": float(reinforce_metrics["policy_loss"]),
                        "reinforce_total_loss": float(reinforce_metrics["total_loss"]),
                        "efficiency_penalty_weighted": float(reinforce_metrics.get("efficiency_penalty_weighted", 0.0)),
                        "lambda_eff": float(reinforce_metrics.get("lambda_eff", 1.0)),
                        "lambda_entropy": float(reinforce_metrics.get("lambda_entropy", lambda_entropy)),
                        "entropy": float(reinforce_metrics.get("entropy", 0.0)),
                        "entropy_bonus": float(reinforce_metrics.get("entropy_bonus", 0.0)),
                        "actual_full_attn_ratio": full_attn_ratio_eligible,
                        "actual_ttr_ratio": float(max(0.0, 1.0 - full_attn_ratio_eligible)),
                        "actual_full_attn_ratio_overall": full_attn_ratio_overall,
                        "actual_ttr_ratio_overall": float(max(0.0, 1.0 - full_attn_ratio_overall)),
                        "target_ttr_ratio": target_ttr_ratio,
                        "target_full_attn_ratio": target_full_attn_ratio,
                        "gumbel_temperature": float(gumbel_temperature),
                        "sigma": float(reinforce_metrics.get("sigma", sigma_value)),
                        "iteration": float(iteration + 1),
                        "latent_index": float(latent_idx),
                        "width": float(width),
                        "height": float(height),
                        "step": float(global_step + 1),
                        "eligible_ttr_layers": float(len(ready_layer_indices)),
                        "forced_full_layers": float(forced_full_layer_count),
                        "sigma_aware_training": float(1.0 if sigma_aware_training else 0.0),
                    }
                    if sigma_aware_training:
                        metrics["ttr_ratio_high_sigma"] = float(reinforce_metrics.get("ttr_ratio_high_sigma", 0.0))
                        metrics["ttr_ratio_low_sigma"] = float(reinforce_metrics.get("ttr_ratio_low_sigma", 0.0))
                        metrics["ttr_ratio_spread"] = float(reinforce_metrics.get("ttr_ratio_spread", 0.0))
                        metrics["steps_per_trajectory"] = float(reinforce_metrics.get("steps_per_trajectory", 0.0))

                    global_step += 1
                    if (
                        checkpoint_path
                        and global_step < total_steps
                        and flux2_ttr_controller.should_save_controller_checkpoint_step(
                            global_step,
                            checkpoint_every=checkpoint_every,
                        )
                    ):
                        flux2_ttr_controller.save_controller_checkpoint(controller, checkpoint_path, trainer=trainer)
                        logger.info(
                            "Flux2TTRControllerTrainer: periodic controller checkpoint saved at step %d/%d: %s",
                            global_step,
                            total_steps,
                            checkpoint_path,
                        )
                    if comet_experiment is not None and (global_step % log_every == 0 or global_step == total_steps):
                        cls._safe_comet_log_metrics(
                            comet_experiment,
                            cls._build_comet_metrics(metrics),
                            global_step,
                        )

                    if global_step % log_every == 0 or global_step == total_steps:
                        if sigma_aware_training:
                            logger.info(
                                (
                                    "Flux2TTRControllerTrainer step=%d/%d iter=%d loss=%.6g rmse=%.6g cosine=%.6g "
                                    "lpips=%.6g full_attn_eligible=%.4g full_attn_overall=%.4g "
                                    "target_full=%.4g eff_penalty=%.6g entropy=%.6g entropy_bonus=%.6g "
                                    "spread=%.4g high_ttr=%.4g low_ttr=%.4g traj_steps=%d reward_baseline=%.6g"
                                ),
                                global_step,
                                total_steps,
                                iteration + 1,
                                metrics["loss"],
                                metrics["rmse"],
                                metrics["cosine_distance"],
                                metrics["lpips"],
                                metrics["full_attn_eligible"],
                                metrics["full_attn_overall"],
                                metrics["target_full_attn_ratio"],
                                metrics["efficiency_penalty"],
                                metrics["entropy"],
                                metrics["entropy_bonus"],
                                metrics.get("ttr_ratio_spread", 0.0),
                                metrics.get("ttr_ratio_high_sigma", 0.0),
                                metrics.get("ttr_ratio_low_sigma", 0.0),
                                int(metrics.get("steps_per_trajectory", 0.0)),
                                metrics["reward_baseline"],
                            )
                        else:
                            logger.info(
                                (
                                    "Flux2TTRControllerTrainer step=%d/%d iter=%d loss=%.6g rmse=%.6g cosine=%.6g "
                                    "lpips=%.6g full_attn_eligible=%.4g full_attn_overall=%.4g "
                                    "target_full=%.4g eff_penalty=%.6g entropy=%.6g "
                                    "entropy_bonus=%.6g reward_baseline=%.6g"
                                ),
                                global_step,
                                total_steps,
                                iteration + 1,
                                metrics["loss"],
                                metrics["rmse"],
                                metrics["cosine_distance"],
                                metrics["lpips"],
                                metrics["full_attn_eligible"],
                                metrics["full_attn_overall"],
                                metrics["target_full_attn_ratio"],
                                metrics["efficiency_penalty"],
                                metrics["entropy"],
                                metrics["entropy_bonus"],
                                metrics["reward_baseline"],
                            )

                    iter_original_images.append(teacher_image.detach().cpu())
                    iter_patched_images.append(student_image.detach().cpu())

                    del teacher_latent
                    del student_latent
                    del teacher_image
                    del student_image
                    del teacher_rgb
                    del student_rgb
                    if not sigma_aware_training:
                        del sampled_mask
                        del effective_mask
                        del ttr_eligible_mask
                    cls._empty_cuda_cache()

                if iter_original_images:
                    last_original_images = torch.cat(iter_original_images, dim=0)
                if iter_patched_images:
                    last_patched_images = torch.cat(iter_patched_images, dim=0)

            if comet_experiment is not None:
                cls._safe_comet_log_metrics(
                    comet_experiment,
                    cls._build_comet_metrics(
                        {
                            "training_completed": 1.0,
                            "total_steps": float(total_steps),
                            "total_iterations": float(total_iterations),
                            "final_reward_baseline": float(trainer._reward_baseline),
                        }
                    ),
                    global_step if global_step > 0 else 0,
                )
        finally:
            pass

        if checkpoint_path:
            flux2_ttr_controller.save_controller_checkpoint(controller, checkpoint_path, trainer=trainer)

        if last_original_images is None or last_patched_images is None:
            raise RuntimeError("Flux2TTRControllerTrainer: no samples were produced.")

        return io.NodeOutput(last_original_images, last_patched_images, controller)


class Flux2TTRController(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Flux2TTRController",
            display_name="Flux2TTRController",
            category="advanced/attention",
            description="Inference node: load TTR + controller checkpoints and patch the model for dynamic layer routing.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("ttr_checkpoint_path", default="", multiline=False),
                io.String.Input("controller_checkpoint_path", default="", multiline=False),
                io.Float.Input("quality_speed", default=0.5, min=0.0, max=1.0, step=1e-3),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        ttr_checkpoint_path: str,
        controller_checkpoint_path: str,
        quality_speed: float,
    ) -> io.NodeOutput:
        ttr_checkpoint_path = (ttr_checkpoint_path or "").strip()
        controller_checkpoint_path = (controller_checkpoint_path or "").strip()
        if not ttr_checkpoint_path:
            raise ValueError("Flux2TTRController: ttr_checkpoint_path is required.")
        if not os.path.isfile(ttr_checkpoint_path):
            raise FileNotFoundError(f"Flux2TTRController: TTR checkpoint not found: {ttr_checkpoint_path}")
        if not controller_checkpoint_path:
            raise ValueError("Flux2TTRController: controller_checkpoint_path is required.")
        if not os.path.isfile(controller_checkpoint_path):
            raise FileNotFoundError(
                f"Flux2TTRController: controller checkpoint not found: {controller_checkpoint_path}"
            )

        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})
        prev_cfg = transformer_options.get("flux2_ttr")
        if isinstance(prev_cfg, dict):
            prev_runtime = prev_cfg.get("runtime_id")
            if isinstance(prev_runtime, str):
                flux2_ttr.unregister_runtime(prev_runtime)

        feature_dim = flux2_ttr.validate_feature_dim(int(prev_cfg.get("feature_dim", 256)) if isinstance(prev_cfg, dict) else 256)
        learning_rate = _float_or(
            prev_cfg.get("learning_rate", _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["learning_rate"]) if isinstance(prev_cfg, dict) else _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["learning_rate"],
            _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["learning_rate"],
        )
        query_chunk_size = _int_or(prev_cfg.get("query_chunk_size", 256) if isinstance(prev_cfg, dict) else 256, 256)
        key_chunk_size = _int_or(prev_cfg.get("key_chunk_size", 1024) if isinstance(prev_cfg, dict) else 1024, 1024)
        landmark_fraction = _float_or(
            prev_cfg.get("landmark_fraction", 0.08) if isinstance(prev_cfg, dict) else 0.08,
            0.08,
        )
        landmark_min = _int_or(prev_cfg.get("landmark_min", 64) if isinstance(prev_cfg, dict) else 64, 64)
        landmark_max = _int_or(prev_cfg.get("landmark_max", 512) if isinstance(prev_cfg, dict) else 512, 512)
        text_tokens_guess = _int_or(prev_cfg.get("text_tokens_guess", 77) if isinstance(prev_cfg, dict) else 77, 77)
        alpha_init = _float_or(prev_cfg.get("alpha_init", 0.1) if isinstance(prev_cfg, dict) else 0.1, 0.1)
        alpha_lr_multiplier = _float_or(
            prev_cfg.get("alpha_lr_multiplier", _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["alpha_lr_multiplier"]) if isinstance(prev_cfg, dict) else _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["alpha_lr_multiplier"],
            _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["alpha_lr_multiplier"],
        )
        phi_lr_multiplier = _float_or(
            prev_cfg.get("phi_lr_multiplier", _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["phi_lr_multiplier"]) if isinstance(prev_cfg, dict) else _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["phi_lr_multiplier"],
            _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["phi_lr_multiplier"],
        )
        training_query_token_cap = _int_or(prev_cfg.get("training_query_token_cap", 128) if isinstance(prev_cfg, dict) else 128, 128)
        replay_buffer_size = _int_or(prev_cfg.get("replay_buffer_size", 8) if isinstance(prev_cfg, dict) else 8, 8)
        replay_offload_cpu = _bool_or(prev_cfg.get("replay_offload_cpu", True) if isinstance(prev_cfg, dict) else True, True)
        replay_max_bytes = _int_or(prev_cfg.get("replay_max_bytes", 768 * 1024 * 1024) if isinstance(prev_cfg, dict) else 768 * 1024 * 1024, 768 * 1024 * 1024)
        train_steps_per_call = _int_or(prev_cfg.get("train_steps_per_call", 1) if isinstance(prev_cfg, dict) else 1, 1)
        huber_beta = _float_or(prev_cfg.get("huber_beta", 0.05) if isinstance(prev_cfg, dict) else 0.05, 0.05)
        grad_clip_norm = _float_or(prev_cfg.get("grad_clip_norm", 1.0) if isinstance(prev_cfg, dict) else 1.0, 1.0)
        readiness_threshold = _float_or(prev_cfg.get("readiness_threshold", 0.12) if isinstance(prev_cfg, dict) else 0.12, 0.12)
        readiness_min_updates = _int_or(prev_cfg.get("readiness_min_updates", 24) if isinstance(prev_cfg, dict) else 24, 24)
        enable_memory_reserve = _bool_or(prev_cfg.get("enable_memory_reserve", False) if isinstance(prev_cfg, dict) else False, False)
        layer_start = _int_or(prev_cfg.get("layer_start", -1) if isinstance(prev_cfg, dict) else -1, -1)
        layer_end = _int_or(prev_cfg.get("layer_end", -1) if isinstance(prev_cfg, dict) else -1, -1)
        cfg_scale = _float_or(prev_cfg.get("cfg_scale", 1.0) if isinstance(prev_cfg, dict) else 1.0, 1.0)
        min_swap_layers = _int_or(prev_cfg.get("min_swap_layers", 1) if isinstance(prev_cfg, dict) else 1, 1)
        max_swap_layers = _int_or(prev_cfg.get("max_swap_layers", -1) if isinstance(prev_cfg, dict) else -1, -1)
        inference_mixed_precision = _bool_or(
            prev_cfg.get("inference_mixed_precision", True) if isinstance(prev_cfg, dict) else True,
            True,
        )

        runtime = flux2_ttr.Flux2TTRRuntime(
            feature_dim=feature_dim,
            learning_rate=float(learning_rate),
            training=False,
            steps=0,
            scan_chunk_size=int(query_chunk_size),
            key_chunk_size=int(key_chunk_size),
            landmark_fraction=float(landmark_fraction),
            landmark_min=int(landmark_min),
            landmark_max=int(landmark_max),
            text_tokens_guess=int(text_tokens_guess),
            alpha_init=float(alpha_init),
            alpha_lr_multiplier=float(alpha_lr_multiplier),
            phi_lr_multiplier=float(phi_lr_multiplier),
            training_query_token_cap=int(training_query_token_cap),
            replay_buffer_size=int(replay_buffer_size),
            replay_offload_cpu=bool(replay_offload_cpu),
            replay_max_bytes=int(replay_max_bytes),
            train_steps_per_call=int(train_steps_per_call),
            huber_beta=float(huber_beta),
            grad_clip_norm=float(grad_clip_norm),
            readiness_threshold=float(readiness_threshold),
            readiness_min_updates=int(readiness_min_updates),
            enable_memory_reserve=bool(enable_memory_reserve),
            layer_start=int(layer_start),
            layer_end=int(layer_end),
            cfg_scale=float(cfg_scale),
            min_swap_layers=int(min_swap_layers),
            max_swap_layers=int(max_swap_layers),
            inference_mixed_precision=bool(inference_mixed_precision),
            training_preview_ttr=False,
        )
        runtime.register_layer_specs(flux2_ttr.infer_flux_single_layer_specs(m))
        runtime.load_checkpoint(ttr_checkpoint_path)
        runtime.training_mode = False
        runtime.training_enabled = False
        runtime.steps_remaining = 0
        runtime.training_updates_done = 0

        controller = flux2_ttr_controller.load_controller_checkpoint(controller_checkpoint_path, map_location="cpu")
        quality_speed = min(1.0, max(0.0, float(quality_speed)))
        threshold = 0.1 + 0.8 * quality_speed

        runtime_id = flux2_ttr.register_runtime(runtime)
        transformer_options["flux2_ttr"] = {
            "enabled": True,
            "training": False,
            "training_mode": False,
            "runtime_id": runtime_id,
            "controller": controller,
            "controller_threshold": float(threshold),
            "checkpoint_path": ttr_checkpoint_path,
            "controller_checkpoint_path": controller_checkpoint_path,
            "feature_dim": int(feature_dim),
            "query_chunk_size": int(query_chunk_size),
            "scan_chunk_size": int(query_chunk_size),
            "key_chunk_size": int(key_chunk_size),
            "landmark_fraction": float(landmark_fraction),
            "landmark_min": int(landmark_min),
            "landmark_max": int(landmark_max),
            "text_tokens_guess": int(text_tokens_guess),
            "alpha_init": float(alpha_init),
            "alpha_lr_multiplier": float(alpha_lr_multiplier),
            "phi_lr_multiplier": float(phi_lr_multiplier),
            "training_query_token_cap": int(training_query_token_cap),
            "replay_buffer_size": int(replay_buffer_size),
            "replay_offload_cpu": bool(replay_offload_cpu),
            "replay_max_bytes": int(replay_max_bytes),
            "train_steps_per_call": int(train_steps_per_call),
            "huber_beta": float(huber_beta),
            "grad_clip_norm": float(grad_clip_norm),
            "readiness_threshold": float(readiness_threshold),
            "readiness_min_updates": int(readiness_min_updates),
            "enable_memory_reserve": bool(enable_memory_reserve),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "cfg_scale": float(cfg_scale),
            "min_swap_layers": int(min_swap_layers),
            "max_swap_layers": int(max_swap_layers),
            "inference_mixed_precision": bool(inference_mixed_precision),
        }

        callback_key = "flux2_ttr"
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_PRE_RUN,
            callback_key,
            flux2_ttr.pre_run_callback,
        )
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_CLEANUP,
            callback_key,
            flux2_ttr.cleanup_callback,
        )

        logger.info(
            "Flux2TTRController configured: ttr_ckpt=%s controller_ckpt=%s quality_speed=%.4g threshold=%.4g",
            ttr_checkpoint_path,
            controller_checkpoint_path,
            quality_speed,
            threshold,
        )
        return io.NodeOutput(m)


class RandomSeedBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RandomSeedBatch",
            display_name="Random Seed Batch",
            category="advanced/scheduling",
            description="Generate a deterministic list of random seeds from a base seed.",
            inputs=[
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    tooltip="Base seed used to deterministically generate the output seed list.",
                ),
                io.Int.Input(
                    "count",
                    default=8,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Number of random seeds to generate.",
                ),
            ],
            outputs=[io.Int.Output("seeds", is_output_list=True)],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, seed: int, count: int) -> io.NodeOutput:
        seeds = sweep_utils.generate_seed_batch(int(seed), int(count))
        logger.info(
            "RandomSeedBatch generated %d seeds from base seed %d (first=%d).",
            len(seeds),
            int(seed),
            int(seeds[0]) if seeds else -1,
        )
        return io.NodeOutput(seeds)


class LoadPromptListFromJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadPromptListFromJSON",
            display_name="LoadPromptListFromJSON",
            category="advanced/scheduling",
            description="Load a JSON file containing an array of prompt strings.",
            inputs=[
                io.String.Input(
                    "json_path",
                    default="",
                    multiline=False,
                    tooltip="Path to JSON file containing a simple array of strings.",
                ),
            ],
            outputs=[io.String.Output("prompts", is_output_list=True)],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, json_path: str) -> io.NodeOutput:
        prompts = sweep_utils.load_prompt_list_from_json(json_path)
        logger.info(
            "LoadPromptListFromJSON loaded %d prompts from %s.",
            len(prompts),
            str(json_path or "").strip(),
        )
        return io.NodeOutput(prompts)


class ClockedSweepValues(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ClockedSweepValues",
            display_name="Clocked Sweep Values",
            category="advanced/scheduling",
            description="Map a clock list to evenly distributed sweep values.",
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "clock",
                        multiline=True,
                        placeholder="0, 1, 2 or 30",
                        tooltip="Clock list (length defines output length). Accepts JSON list, comma/space-separated values, a list input, or a single integer string to create 1..N.",
                    ),
                    [io.AnyType],
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "values",
                        multiline=True,
                        placeholder="0.1, 0.2, 0.3",
                        tooltip="Values to sweep across the clock (JSON list, comma/space-separated, or list input). If clock is blank, its length is inferred from values.",
                    ),
                    [io.AnyType],
                ),
            ],
            outputs=[io.Float.Output("values", is_output_list=True)],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clock, values) -> io.NodeOutput:
        clock_list, values_list = sweep_utils.parse_clock_and_values(clock, values)
        output = sweep_utils.build_clocked_sweep(clock_list, values_list)
        logger.info(
            "Clocked sweep built: clock=%d values=%d output=%d",
            len(clock_list),
            len(values_list),
            len(output),
        )
        return io.NodeOutput(output)


class Combinations(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Combinations",
            display_name="Combinations",
            category="advanced/scheduling",
            description="Generate repeated lists that cover all combinations of provided values.",
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "a",
                        multiline=True,
                        placeholder="1, 2, 3",
                        tooltip="Values for A (JSON list, comma/space-separated, or list input).",
                    ),
                    [io.AnyType],
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "b",
                        multiline=True,
                        placeholder="4, 5",
                        tooltip="Values for B (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "c",
                        multiline=True,
                        placeholder="",
                        tooltip="Values for C (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "d",
                        multiline=True,
                        placeholder="",
                        tooltip="Values for D (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
            ],
            outputs=[
                io.Float.Output("a_out", is_output_list=True),
                io.Float.Output("b_out", is_output_list=True),
                io.Float.Output("c_out", is_output_list=True),
                io.Float.Output("d_out", is_output_list=True),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, a, b=None, c=None, d=None) -> io.NodeOutput:
        values = []
        names = []
        a_list = sweep_utils.to_float_list(a, "a")
        if not a_list:
            raise ValueError("Combinations: 'a' must contain at least one value.")
        values.append(a_list)
        names.append("a")
        for label, item in (("b", b), ("c", c), ("d", d)):
            if item is None or item == "":
                continue
            parsed = sweep_utils.to_float_list(item, label)
            if parsed:
                values.append(parsed)
                names.append(label)

        outputs = sweep_utils.build_combinations(values)
        out_map = dict(zip(names, outputs))
        return io.NodeOutput(
            out_map.get("a", []),
            out_map.get("b", []),
            out_map.get("c", []),
            out_map.get("d", []),
        )


class TaylorAttentionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TaylorAttentionBackend,
            HybridTaylorAttentionBackend,
            Flux2TTRTrainingParameters,
            Flux2TTRTrainer,
            Flux2TTRControllerTrainer,
            Flux2TTRController,
            RandomSeedBatch,
            LoadPromptListFromJSON,
            ClockedSweepValues,
            Combinations,
        ]


async def comfy_entrypoint() -> TaylorAttentionExtension:
    return TaylorAttentionExtension()
