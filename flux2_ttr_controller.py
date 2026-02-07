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


def controller_checkpoint_state(controller: TTRController) -> Dict[str, Any]:
    if not isinstance(controller, TTRController):
        raise TypeError("controller_checkpoint_state expects a TTRController instance.")
    return {
        "format": _CONTROLLER_CHECKPOINT_FORMAT,
        "num_layers": int(controller.num_layers),
        "embed_dim": int(controller.embed_dim),
        "hidden_dim": int(controller.hidden_dim),
        "state_dict": {k: v.detach().cpu() for k, v in controller.state_dict().items()},
    }


def save_controller_checkpoint(controller: TTRController, checkpoint_path: str) -> None:
    path = (checkpoint_path or "").strip()
    if not path:
        raise ValueError("save_controller_checkpoint: checkpoint_path is required.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(controller_checkpoint_state(controller), path)
    logger.info("Flux2TTR controller: saved checkpoint to %s", path)


def load_controller_checkpoint(checkpoint_path: str, map_location: str | torch.device = "cpu") -> TTRController:
    path = (checkpoint_path or "").strip()
    if not path:
        raise ValueError("load_controller_checkpoint: checkpoint_path is required.")
    payload = torch.load(path, map_location=map_location)
    fmt = payload.get("format")
    if fmt != _CONTROLLER_CHECKPOINT_FORMAT:
        raise ValueError(f"Flux2TTR controller: unsupported checkpoint format: {fmt!r}")

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
            learning_rate = float(opt_cfg.get("learning_rate", learning_rate))
            grad_clip_norm = float(opt_cfg.get("grad_clip_norm", grad_clip_norm))

        self.controller = controller
        if device is not None:
            self.controller.to(device=device)

        self.optimizer = torch.optim.AdamW(self.controller.parameters(), lr=float(learning_rate))
        self.rmse_weight = float(rmse_weight)
        self.cosine_weight = float(cosine_weight)
        self.lpips_weight = float(lpips_weight)
        self.target_ttr_ratio = float(target_ttr_ratio)
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
            "ControllerTrainer initialized: lr=%.6g rmse=%.4g cosine=%.4g lpips=%.4g target_ratio=%.4g grad_clip=%.4g",
            float(learning_rate),
            self.rmse_weight,
            self.cosine_weight,
            self.lpips_weight,
            self.target_ttr_ratio,
            self.grad_clip_norm,
        )

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
            teacher_rgb = self._prep_lpips_rgb(teacher_rgb).to(device=loss.device)
            student_rgb = self._prep_lpips_rgb(student_rgb).to(device=loss.device)
            lpips_term = self.lpips_model(student_rgb, teacher_rgb).mean()
            loss = loss + self.lpips_weight * lpips_term

        ratio = self._ratio_tensor(
            actual_full_attn_ratio,
            device=loss.device,
            dtype=loss.dtype,
        )
        efficiency_penalty = torch.relu(ratio - float(self.target_ttr_ratio))
        if include_efficiency_penalty:
            loss = loss + efficiency_penalty

        metrics = {
            "loss": float(loss.detach().item()),
            "rmse": float(rmse.detach().item()),
            "cosine_distance": float(cosine.detach().item()),
            "lpips": float(lpips_term.detach().item()),
            "efficiency_penalty": float(efficiency_penalty.detach().item()),
            "actual_full_attn_ratio": float(ratio.detach().item()),
            "target_ttr_ratio": float(self.target_ttr_ratio),
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
    ) -> Dict[str, float]:
        self.controller.train()
        reward_value = float(reward)

        # ComfyUI often executes nodes in inference_mode; force grad-enabled training here.
        with torch.inference_mode(False):
            with torch.enable_grad():
                logits = self.controller(sigma=sigma, cfg_scale=cfg_scale, width=width, height=height)
                probs = torch.sigmoid(logits)

                mask = sampled_mask.to(device=logits.device, dtype=logits.dtype).reshape(-1).detach().clone()
                if mask.numel() != probs.numel():
                    raise ValueError(
                        "ControllerTrainer.reinforce_step: sampled_mask length "
                        f"{int(mask.numel())} does not match controller layer count {int(probs.numel())}."
                    )

                log_probs = mask * torch.log(probs + 1e-8) + (1.0 - mask) * torch.log(1.0 - probs + 1e-8)
                total_log_prob = log_probs.sum()

                baselined_reward = reward_value - self._reward_baseline
                self._update_reward_baseline(reward_value)

                policy_loss = -baselined_reward * total_log_prob
                efficiency_penalty = torch.relu(probs.mean() - float(self.target_ttr_ratio))
                loss = policy_loss + efficiency_penalty

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                metrics = {
                    "policy_loss": float(policy_loss.detach().item()),
                    "efficiency_penalty": float(efficiency_penalty.detach().item()),
                    "total_loss": float(loss.detach().item()),
                    "reward": reward_value,
                    "reward_baseline": float(self._reward_baseline),
                    "baselined_reward": float(baselined_reward),
                    "mask_mean": float(mask.mean().item()),
                    "probs_mean": float(probs.detach().mean().item()),
                    "actual_full_attn_ratio": float(actual_full_attn_ratio),
                    "target_ttr_ratio": float(self.target_ttr_ratio),
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
