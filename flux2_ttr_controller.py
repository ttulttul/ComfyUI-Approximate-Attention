from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

_CONTROLLER_CHECKPOINT_FORMAT = "flux2_ttr_controller_v1"
_LPIPS_EPS = 1e-8


class _ScalarSinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        dim = max(2, int(embed_dim))
        if dim % 2 != 0:
            dim += 1
        self.embed_dim = dim
        half = dim // 2
        exponents = torch.arange(half, dtype=torch.float32) / max(1, half - 1)
        inv_freq = torch.exp(-math.log(10000.0) * exponents)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            x = x.reshape(1)
        x = x.reshape(-1).float()
        angles = x[:, None] * self.inv_freq[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class _ResolutionEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.width_embed = _ScalarSinusoidalEmbedding(embed_dim)
        self.height_embed = _ScalarSinusoidalEmbedding(embed_dim)
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

        self.sigma_embed = _ScalarSinusoidalEmbedding(self.embed_dim)
        self.cfg_embed = _ScalarSinusoidalEmbedding(self.embed_dim)
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
        learning_rate: float = 1e-3,
        rmse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        lpips_weight: float = 0.0,
        target_ttr_ratio: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        if not isinstance(controller, TTRController):
            raise TypeError("ControllerTrainer expects a TTRController instance.")
        self.controller = controller
        if device is not None:
            self.controller.to(device=device)

        self.optimizer = torch.optim.AdamW(self.controller.parameters(), lr=float(learning_rate))
        self.rmse_weight = float(rmse_weight)
        self.cosine_weight = float(cosine_weight)
        self.lpips_weight = float(lpips_weight)
        self.target_ttr_ratio = float(target_ttr_ratio)
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

    def compute_loss(
        self,
        *,
        teacher_latent: torch.Tensor,
        student_latent: torch.Tensor,
        actual_full_attn_ratio: float,
        teacher_rgb: Optional[torch.Tensor] = None,
        student_rgb: Optional[torch.Tensor] = None,
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

        efficiency_penalty = torch.relu(
            torch.tensor(float(actual_full_attn_ratio), device=loss.device) - float(self.target_ttr_ratio)
        )
        loss = loss + efficiency_penalty

        metrics = {
            "loss": float(loss.detach().item()),
            "rmse": float(rmse.detach().item()),
            "cosine_distance": float(cosine.detach().item()),
            "lpips": float(lpips_term.detach().item()),
            "efficiency_penalty": float(efficiency_penalty.detach().item()),
            "actual_full_attn_ratio": float(actual_full_attn_ratio),
            "target_ttr_ratio": float(self.target_ttr_ratio),
        }
        return loss, metrics

    def train_step(
        self,
        *,
        sigma: float,
        cfg_scale: float,
        width: int,
        height: int,
        teacher_latent: torch.Tensor,
        student_latent: torch.Tensor,
        actual_full_attn_ratio: float,
        teacher_rgb: Optional[torch.Tensor] = None,
        student_rgb: Optional[torch.Tensor] = None,
        gumbel_temperature: float = 1.0,
    ) -> Dict[str, float]:
        self.controller.train()
        logits = self.controller(sigma=sigma, cfg_scale=cfg_scale, width=width, height=height)
        mask = self.controller.sample_training_mask(logits, temperature=gumbel_temperature, hard=True)

        loss, metrics = self.compute_loss(
            teacher_latent=teacher_latent,
            student_latent=student_latent,
            actual_full_attn_ratio=actual_full_attn_ratio,
            teacher_rgb=teacher_rgb,
            student_rgb=student_rgb,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        metrics["mask_mean"] = float(mask.detach().mean().item())
        metrics["layers_full_ratio"] = float(mask.detach().mean().item())
        return metrics
