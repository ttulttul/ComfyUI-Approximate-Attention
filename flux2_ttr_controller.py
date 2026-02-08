from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from flux2_ttr_embeddings import ScalarSinusoidalEmbedding

logger = logging.getLogger(__name__)

_CONTROLLER_CHECKPOINT_FORMAT = "flux2_ttr_controller_v1"
_LPIPS_EPS = 1e-8
DEFAULT_CONTROLLER_CHECKPOINT_EVERY = 10


def should_save_controller_checkpoint_step(step: int, checkpoint_every: int = DEFAULT_CONTROLLER_CHECKPOINT_EVERY) -> bool:
    return int(step) > 0 and int(checkpoint_every) > 0 and (int(step) % int(checkpoint_every) == 0)

class _ResolutionEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.width_embed = ScalarSinusoidalEmbedding(embed_dim)
        self.height_embed = ScalarSinusoidalEmbedding(embed_dim)
        in_dim = self.width_embed.embed_dim + self.height_embed.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, width: torch.Tensor, height: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        emb = torch.cat([self.width_embed(width), self.height_embed(height)], dim=-1)
        return self.proj(emb.to(dtype=dtype))


class TTRController(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("TTRController: num_layers must be positive.")

        self.num_layers = int(num_layers)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.sigma_embed = ScalarSinusoidalEmbedding(self.embed_dim)
        self.cfg_embed = ScalarSinusoidalEmbedding(self.embed_dim)
        self.resolution_embed = _ResolutionEmbedding(self.embed_dim)

        mlp_in = self.sigma_embed.embed_dim + self.cfg_embed.embed_dim + self.embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_layers),
        )

    def _input_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.parameters())
        return param.device, param.dtype

    def forward(
        self,
        sigma: float | torch.Tensor,
        cfg_scale: float | torch.Tensor,
        width: int | float | torch.Tensor,
        height: int | float | torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = self._input_device_dtype()
        sigma_t = torch.as_tensor(sigma, device=device, dtype=torch.float32).reshape(1)
        cfg_t = torch.as_tensor(cfg_scale, device=device, dtype=torch.float32).reshape(1)
        width_t = torch.as_tensor(width, device=device, dtype=torch.float32).reshape(1)
        height_t = torch.as_tensor(height, device=device, dtype=torch.float32).reshape(1)

        sigma_e = self.sigma_embed(sigma_t).to(dtype=dtype)
        cfg_e = self.cfg_embed(cfg_t).to(dtype=dtype)
        res_e = self.resolution_embed(width_t, height_t, dtype=dtype)
        hidden = torch.cat([sigma_e, cfg_e, res_e], dim=-1)
        logits = self.mlp(hidden).reshape(-1)
        return logits.float()

    @staticmethod
    def sample_training_mask(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
        t = max(1e-4, float(temperature))
        u = torch.rand_like(logits).clamp(min=1e-6, max=1.0 - 1e-6)
        g = -torch.log(-torch.log(u))
        probs = torch.sigmoid((logits + g) / t)
        if not hard:
            return probs
        hard_mask = (probs > 0.5).to(dtype=probs.dtype)
        return hard_mask.detach() - probs.detach() + probs

    @staticmethod
    def inference_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(logits.float()) > float(threshold)).to(dtype=torch.float32)


def controller_checkpoint_state(
    controller: TTRController,
    trainer: Optional["ControllerTrainer"] = None,
) -> Dict[str, Any]:
    if not isinstance(controller, TTRController):
        raise TypeError("controller_checkpoint_state expects a TTRController instance.")
    return {
        "format": _CONTROLLER_CHECKPOINT_FORMAT,
        "num_layers": int(controller.num_layers),
        "embed_dim": int(controller.embed_dim),
        "hidden_dim": int(controller.hidden_dim),
        "state_dict": {k: v.detach().cpu() for k, v in controller.state_dict().items()},
        "reward_baseline": float(trainer.reward_baseline) if trainer is not None else None,
        "reward_count": int(trainer.reward_count) if trainer is not None else None,
        "lambda_entropy": float(trainer.lambda_entropy) if trainer is not None else None,
        "optimizer_state_dict": trainer.optimizer.state_dict() if trainer is not None else None,
    }


def save_controller_checkpoint(
    controller: TTRController,
    checkpoint_path: str,
    trainer: Optional["ControllerTrainer"] = None,
) -> None:
    path = (checkpoint_path or "").strip()
    if not path:
        raise ValueError("save_controller_checkpoint: checkpoint_path is required.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(controller_checkpoint_state(controller, trainer=trainer), path)
    logger.info("Flux2TTR controller: saved checkpoint to %s", path)


def load_controller_training_state(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    path = (checkpoint_path or "").strip()
    if not path:
        raise ValueError("load_controller_training_state: checkpoint_path is required.")
    payload = torch.load(path, map_location=map_location)
    fmt = payload.get("format")
    if fmt != _CONTROLLER_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {fmt!r}")
    return payload


def load_controller_checkpoint(checkpoint_path: str, map_location: str | torch.device = "cpu") -> TTRController:
    payload = load_controller_training_state(checkpoint_path, map_location=map_location)
    controller = TTRController(
        num_layers=int(payload["num_layers"]),
        embed_dim=int(payload.get("embed_dim", 64)),
        hidden_dim=int(payload.get("hidden_dim", 256)),
    )
    controller.load_state_dict(payload["state_dict"], strict=True)
    controller.eval()
    return controller


class ControllerTrainer:
    def __init__(
        self,
        controller: TTRController,
        *,
        training_config: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-3,
        rmse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        lpips_weight: float = 0.0,
        target_ttr_ratio: float = 0.7,
        lambda_eff: float = 1.0,
        lambda_entropy: float = 0.1,
        grad_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        if not isinstance(controller, TTRController):
            raise TypeError("ControllerTrainer expects a TTRController instance.")

        if training_config is not None:
            loss_cfg = training_config.get("loss_config", {}) if isinstance(training_config, dict) else {}
            opt_cfg = training_config.get("optimizer_config", {}) if isinstance(training_config, dict) else {}
            sched_cfg = training_config.get("schedule_config", {}) if isinstance(training_config, dict) else {}

            rmse_weight = float(loss_cfg.get("rmse_weight", rmse_weight))
            cosine_weight = float(loss_cfg.get("cosine_weight", cosine_weight))
            lpips_weight = float(loss_cfg.get("lpips_weight", lpips_weight))
            target_ttr_ratio = float(sched_cfg.get("target_ttr_ratio", target_ttr_ratio))
            lambda_eff = float(sched_cfg.get("lambda_eff", lambda_eff))
            lambda_entropy = float(sched_cfg.get("lambda_entropy", lambda_entropy))
            learning_rate = float(opt_cfg.get("learning_rate", learning_rate))
            grad_clip_norm = float(opt_cfg.get("grad_clip_norm", grad_clip_norm))

        if self._module_contains_inference_tensors(controller):
            logger.warning(
                "ControllerTrainer: controller parameters were created under inference mode; rebuilding trainable copy."
            )
            controller = self._rebuild_controller_trainable_copy(controller)

        self.controller = controller
        if device is not None:
            with torch.inference_mode(False):
                self.controller.to(device=device)

        self.optimizer = torch.optim.AdamW(self.controller.parameters(), lr=float(learning_rate))
        self.rmse_weight = float(rmse_weight)
        self.cosine_weight = float(cosine_weight)
        self.lpips_weight = float(lpips_weight)
        self.target_ttr_ratio = float(target_ttr_ratio)
        self.lambda_eff = max(0.0, float(lambda_eff))
        self.lambda_entropy = max(0.0, float(lambda_entropy))
        self.grad_clip_norm = max(0.0, float(grad_clip_norm))
        self._reward_baseline = 0.0
        self._reward_count = 0
        self._train_step_warned = False
        self.lpips_model = None
        if self.lpips_weight > 0:
            try:
                import lpips  # type: ignore

                self.lpips_model = lpips.LPIPS(net="alex")
                self.lpips_model.eval()
                controller_device, _ = self.controller._input_device_dtype()
                self.lpips_model.to(device=controller_device)
            except Exception as exc:
                raise RuntimeError(
                    "ControllerTrainer: lpips_weight > 0 requires the 'lpips' package."
                ) from exc
        logger.info(
            (
                "ControllerTrainer initialized: lr=%.6g rmse=%.4g cosine=%.4g lpips=%.4g "
                "target_ttr_ratio=%.4g target_full_attn_ratio=%.4g "
                "lambda_eff=%.4g lambda_entropy=%.4g grad_clip=%.4g"
            ),
            float(learning_rate),
            self.rmse_weight,
            self.cosine_weight,
            self.lpips_weight,
            self.target_ttr_ratio,
            self._target_full_attn_ratio_from_ttr_ratio(self.target_ttr_ratio),
            self.lambda_eff,
            self.lambda_entropy,
            self.grad_clip_norm,
        )

    @property
    def reward_baseline(self) -> float:
        return float(self._reward_baseline)

    @property
    def reward_count(self) -> int:
        return int(self._reward_count)

    def restore_training_state(self, payload: Dict[str, Any]) -> None:
        """Restore reward baseline and optimizer state from a checkpoint payload."""
        if not isinstance(payload, dict):
            return
        if "reward_baseline" in payload and payload["reward_baseline"] is not None:
            self._reward_baseline = float(payload["reward_baseline"])
        if "reward_count" in payload and payload["reward_count"] is not None:
            self._reward_count = int(payload["reward_count"])
        if "lambda_entropy" in payload and payload["lambda_entropy"] is not None:
            self.lambda_entropy = max(0.0, float(payload["lambda_entropy"]))
        if "optimizer_state_dict" in payload and payload["optimizer_state_dict"] is not None:
            try:
                with torch.inference_mode(False):
                    optimizer_state = self._clone_state_tensors(payload["optimizer_state_dict"])
                    self.optimizer.load_state_dict(optimizer_state)
                target_device = self.controller._input_device_dtype()[0]
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            with torch.inference_mode(False):
                                state[key] = value.detach().clone().to(device=target_device)
            except Exception as exc:
                logger.warning(
                    "ControllerTrainer: failed to restore optimizer state (%s). "
                    "Continuing with fresh optimizer.",
                    exc,
                )

    @staticmethod
    def _is_inference_tensor(t: torch.Tensor) -> bool:
        checker = getattr(t, "is_inference", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False

    @classmethod
    def _module_contains_inference_tensors(cls, module: nn.Module) -> bool:
        for tensor in list(module.parameters()) + list(module.buffers()):
            if cls._is_inference_tensor(tensor):
                return True
        return False

    @staticmethod
    def _clone_state_dict_tensors(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in state_dict.items()}

    @classmethod
    def _clone_state_tensors(cls, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().clone()
        if isinstance(obj, dict):
            return {k: cls._clone_state_tensors(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._clone_state_tensors(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(cls._clone_state_tensors(v) for v in obj)
        return obj

    @classmethod
    def _rebuild_controller_trainable_copy(cls, controller: TTRController) -> TTRController:
        with torch.inference_mode(False):
            rebuilt = TTRController(
                num_layers=int(controller.num_layers),
                embed_dim=int(controller.embed_dim),
                hidden_dim=int(controller.hidden_dim),
            )
            state = cls._clone_state_dict_tensors(controller.state_dict())
            rebuilt.load_state_dict(state, strict=True)
        return rebuilt

    @staticmethod
    def _cosine_distance(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = student.reshape(student.shape[0], -1)
        t = teacher.reshape(teacher.shape[0], -1)
        cos = F.cosine_similarity(s, t, dim=1, eps=_LPIPS_EPS)
        return (1.0 - cos).mean()

    @staticmethod
    def _prep_lpips_rgb(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("LPIPS tensors must be [B,C,H,W].")
        x = x.float()
        if x.shape[1] != 3:
            raise ValueError("LPIPS tensors must have 3 channels.")
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def _module_primary_device(module: nn.Module) -> Optional[torch.device]:
        for tensor in list(module.parameters()) + list(module.buffers()):
            return tensor.device
        return None

    def _ensure_lpips_model_device(self, target_device: torch.device) -> None:
        if self.lpips_model is None:
            return
        current_device = self._module_primary_device(self.lpips_model)
        if current_device != target_device:
            logger.debug(
                "ControllerTrainer: moving LPIPS model from %s to %s.",
                str(current_device),
                str(target_device),
            )
            with torch.inference_mode(False):
                self.lpips_model.to(device=target_device)

    @staticmethod
    def _ratio_tensor(
        actual_full_attn_ratio: float | torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if torch.is_tensor(actual_full_attn_ratio):
            ratio = actual_full_attn_ratio.to(device=device, dtype=dtype)
            if ratio.ndim > 0:
                ratio = ratio.mean()
            return ratio
        return torch.tensor(float(actual_full_attn_ratio), device=device, dtype=dtype)

    @staticmethod
    def _target_full_attn_ratio_from_ttr_ratio(target_ttr_ratio: float) -> float:
        # target_ttr_ratio is semantically "desired TTR usage fraction", so
        # full-attention usage should be constrained by (1 - target_ttr_ratio).
        target_ttr = max(0.0, min(1.0, float(target_ttr_ratio)))
        return 1.0 - target_ttr

    def compute_loss(
        self,
        *,
        teacher_latent: torch.Tensor,
        student_latent: torch.Tensor,
        actual_full_attn_ratio: float | torch.Tensor,
        teacher_rgb: Optional[torch.Tensor] = None,
        student_rgb: Optional[torch.Tensor] = None,
        include_efficiency_penalty: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher = teacher_latent.float()
        student = student_latent.float()

        rmse = torch.sqrt(torch.mean((student - teacher).square()) + _LPIPS_EPS)
        cosine = self._cosine_distance(student, teacher)
        loss = self.rmse_weight * rmse + self.cosine_weight * cosine

        lpips_term = torch.tensor(0.0, device=loss.device)
        if self.lpips_weight > 0:
            if self.lpips_model is None:
                raise RuntimeError("LPIPS requested but lpips_model is not initialized.")
            if teacher_rgb is None or student_rgb is None:
                raise ValueError("teacher_rgb and student_rgb are required when lpips_weight > 0.")
            teacher_rgb = self._prep_lpips_rgb(teacher_rgb)
            student_rgb = self._prep_lpips_rgb(student_rgb)

            lpips_device = teacher_rgb.device
            if student_rgb.device != lpips_device:
                student_rgb = student_rgb.to(device=lpips_device)
            self._ensure_lpips_model_device(lpips_device)

            lpips_term = self.lpips_model(student_rgb, teacher_rgb).mean().to(device=loss.device)
            loss = loss + self.lpips_weight * lpips_term

        ratio = self._ratio_tensor(
            actual_full_attn_ratio,
            device=loss.device,
            dtype=loss.dtype,
        )
        target_full_attn_ratio = self._target_full_attn_ratio_from_ttr_ratio(self.target_ttr_ratio)
        target_full_ratio = torch.tensor(target_full_attn_ratio, device=loss.device, dtype=loss.dtype)
        efficiency_penalty = torch.relu(ratio - target_full_ratio)
        if include_efficiency_penalty:
            loss = loss + efficiency_penalty

        actual_full_attn_ratio_value = float(ratio.detach().item())
        metrics = {
            "loss": float(loss.detach().item()),
            "rmse": float(rmse.detach().item()),
            "cosine_distance": float(cosine.detach().item()),
            "lpips": float(lpips_term.detach().item()),
            "efficiency_penalty": float(efficiency_penalty.detach().item()),
            "actual_full_attn_ratio": actual_full_attn_ratio_value,
            "actual_ttr_ratio": float(1.0 - actual_full_attn_ratio_value),
            "target_ttr_ratio": float(self.target_ttr_ratio),
            "target_full_attn_ratio": float(target_full_attn_ratio),
        }
        return loss, metrics

    def _update_reward_baseline(self, reward: float, decay: float = 0.95) -> None:
        self._reward_count += 1
        reward_f = float(reward)
        if self._reward_count == 1:
            self._reward_baseline = reward_f
        else:
            self._reward_baseline = decay * self._reward_baseline + (1.0 - decay) * reward_f

    def reinforce_step(
        self,
        *,
        sigma: float,
        cfg_scale: float,
        width: int,
        height: int,
        sampled_mask: torch.Tensor,
        reward: float,
        actual_full_attn_ratio: float,
        eligible_layer_mask: Optional[torch.Tensor] = None,
        actual_full_attn_ratio_overall: Optional[float] = None,
    ) -> Dict[str, float]:
        self.controller.train()
        reward_quality = float(reward)

        # ComfyUI often executes nodes in inference_mode; force grad-enabled training here.
        with torch.inference_mode(False):
            with torch.enable_grad():
                logits = self.controller(sigma=sigma, cfg_scale=cfg_scale, width=width, height=height)
                probs = torch.sigmoid(logits)
                probs_clamped = probs.clamp(min=1e-6, max=1.0 - 1e-6)
                per_layer_entropy = -(
                    probs_clamped * torch.log(probs_clamped)
                    + (1.0 - probs_clamped) * torch.log(1.0 - probs_clamped)
                )

                mask = sampled_mask.to(device=logits.device, dtype=logits.dtype).reshape(-1).detach().clone()
                if mask.numel() != probs.numel():
                    raise ValueError(
                        "ControllerTrainer.reinforce_step: sampled_mask length "
                        f"{int(mask.numel())} does not match controller layer count {int(probs.numel())}."
                    )

                if eligible_layer_mask is not None:
                    # ComfyUI can pass inference-mode tensors into nodes; clone here so
                    # boolean indexing works in autograd-tracked ops below.
                    eligible = (
                        eligible_layer_mask.to(device=logits.device, dtype=torch.bool)
                        .reshape(-1)
                        .detach()
                        .clone()
                    )
                    if eligible.numel() != probs.numel():
                        raise ValueError(
                            "ControllerTrainer.reinforce_step: eligible_layer_mask length "
                            f"{int(eligible.numel())} does not match controller layer count {int(probs.numel())}."
                        )
                else:
                    eligible = torch.ones_like(probs, dtype=torch.bool)

                eligible_count = int(eligible.sum().item())
                total_layer_count = int(probs.numel())
                if eligible_count <= 0:
                    raise ValueError("ControllerTrainer.reinforce_step: eligible_layer_mask must include at least one layer.")
                forced_full_layer_count = int(total_layer_count - eligible_count)
                entropy = per_layer_entropy[eligible].mean()
                entropy_bonus = float(self.lambda_entropy * entropy.detach().item())

                log_probs = mask * torch.log(probs + 1e-8) + (1.0 - mask) * torch.log(1.0 - probs + 1e-8)
                # Restrict policy gradient to controllable layers only.
                total_log_prob = log_probs[eligible].sum()

                target_full_attn_ratio = self._target_full_attn_ratio_from_ttr_ratio(self.target_ttr_ratio)
                actual_full_attn_eligible = float(mask[eligible].mean().item())
                efficiency_penalty_value = max(0.0, actual_full_attn_eligible - float(target_full_attn_ratio))
                efficiency_penalty_weighted = float(self.lambda_eff * efficiency_penalty_value)
                reward_value = reward_quality - efficiency_penalty_weighted + entropy_bonus
                baselined_reward = reward_value - self._reward_baseline
                self._update_reward_baseline(reward_value)

                policy_loss = -baselined_reward * total_log_prob
                probs_mean_eligible = probs[eligible].mean()
                loss = policy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                mask_mean_overall = float(mask.mean().item())
                mask_mean_eligible = float(mask[eligible].mean().item())
                probs_mean_overall = float(probs.detach().mean().item())
                probs_mean_eligible_value = float(probs_mean_eligible.detach().item())
                expected_full_attn_ratio_overall = float(
                    (probs_mean_eligible_value * float(eligible_count) + float(forced_full_layer_count))
                    / float(max(1, total_layer_count))
                )
                actual_full_attn_ratio_value = float(actual_full_attn_ratio)
                actual_full_attn_ratio_overall_value = (
                    float(actual_full_attn_ratio_overall)
                    if actual_full_attn_ratio_overall is not None
                    else mask_mean_overall
                )
                metrics = {
                    "policy_loss": float(policy_loss.detach().item()),
                    "efficiency_penalty": float(efficiency_penalty_value),
                    "efficiency_penalty_weighted": float(efficiency_penalty_weighted),
                    "total_loss": float(loss.detach().item()),
                    "reward": reward_value,
                    "reward_quality": reward_quality,
                    "reward_baseline": float(self._reward_baseline),
                    "baselined_reward": float(baselined_reward),
                    "entropy": float(entropy.detach().item()),
                    "entropy_bonus": float(entropy_bonus),
                    "mask_mean": mask_mean_eligible,
                    "mask_mean_overall": mask_mean_overall,
                    "full_attn_mask_mean": mask_mean_eligible,
                    "full_attn_mask_mean_overall": mask_mean_overall,
                    "ttr_mask_mean": float(1.0 - mask_mean_eligible),
                    "ttr_mask_mean_overall": float(1.0 - mask_mean_overall),
                    "probs_mean": probs_mean_eligible_value,
                    "probs_mean_overall": probs_mean_overall,
                    "expected_full_attn_ratio": probs_mean_eligible_value,
                    "expected_full_attn_ratio_overall": expected_full_attn_ratio_overall,
                    "expected_ttr_ratio": float(1.0 - probs_mean_eligible_value),
                    "expected_ttr_ratio_overall": float(1.0 - expected_full_attn_ratio_overall),
                    "actual_full_attn_ratio": actual_full_attn_ratio_value,
                    "actual_full_attn_ratio_overall": actual_full_attn_ratio_overall_value,
                    "actual_ttr_ratio": float(1.0 - actual_full_attn_ratio_value),
                    "actual_ttr_ratio_overall": float(1.0 - actual_full_attn_ratio_overall_value),
                    "target_ttr_ratio": float(self.target_ttr_ratio),
                    "target_full_attn_ratio": float(target_full_attn_ratio),
                    "lambda_eff": float(self.lambda_eff),
                    "lambda_entropy": float(self.lambda_entropy),
                    "eligible_layer_count": float(eligible_count),
                    "total_layer_count": float(total_layer_count),
                    "forced_full_layer_count": float(forced_full_layer_count),
                }
        return metrics

    def train_step(
        self,
        *,
        sigma: float,
        cfg_scale: float,
        width: int,
        height: int,
        teacher_latent: torch.Tensor,
        student_forward_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, Any]],
        teacher_rgb: Optional[torch.Tensor] = None,
        gumbel_temperature: float = 1.0,
        hard_mask: bool = True,
    ) -> Dict[str, float]:
        if not self._train_step_warned:
            logger.warning(
                "ControllerTrainer.train_step is deprecated; prefer reinforce_step for policy-gradient training."
            )
            self._train_step_warned = True
        if not callable(student_forward_fn):
            raise TypeError("ControllerTrainer.train_step requires a callable student_forward_fn(mask, logits).")
        self.controller.train()
        with torch.inference_mode(False):
            with torch.enable_grad():
                logits = self.controller(sigma=sigma, cfg_scale=cfg_scale, width=width, height=height)
                mask = self.controller.sample_training_mask(logits, temperature=gumbel_temperature, hard=hard_mask)

                forward_out = student_forward_fn(mask, logits)
                if not isinstance(forward_out, dict):
                    raise TypeError("student_forward_fn must return a dict with at least 'student_latent'.")
                student_latent = forward_out.get("student_latent")
                if not torch.is_tensor(student_latent):
                    raise ValueError("student_forward_fn output must include tensor 'student_latent'.")
                actual_full_attn_ratio = forward_out.get("actual_full_attn_ratio", mask.mean())
                student_rgb = forward_out.get("student_rgb")

                loss, metrics = self.compute_loss(
                    teacher_latent=teacher_latent,
                    student_latent=student_latent,
                    actual_full_attn_ratio=actual_full_attn_ratio,
                    teacher_rgb=teacher_rgb,
                    student_rgb=student_rgb,
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                metrics["mask_mean"] = float(mask.detach().mean().item())
                metrics["layers_full_ratio"] = float(metrics.get("actual_full_attn_ratio", metrics["mask_mean"]))
        return metrics
