import sys

import pytest
import torch

import flux2_ttr_controller


def test_ttr_controller_forward_shape():
    controller = flux2_ttr_controller.TTRController(num_layers=6, embed_dim=16, hidden_dim=32)
    logits = controller(sigma=0.5, cfg_scale=4.0, width=128, height=128)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()


def test_ttr_controller_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    path = tmp_path / "controller.pt"
    flux2_ttr_controller.save_controller_checkpoint(controller, str(path))

    loaded = flux2_ttr_controller.load_controller_checkpoint(str(path))
    out_a = controller(sigma=0.25, cfg_scale=3.0, width=64, height=64)
    out_b = loaded(sigma=0.25, cfg_scale=3.0, width=64, height=64)
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_controller_trainer_compute_loss_without_lpips():
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0)

    teacher = torch.randn(1, 4, 8, 8)
    student = teacher + 0.05 * torch.randn_like(teacher)
    loss, metrics = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.4,
    )
    assert loss.item() > 0
    assert metrics["actual_full_attn_ratio"] == pytest.approx(0.4)
    assert "rmse" in metrics
    assert "cosine_distance" in metrics


def test_controller_trainer_lpips_requires_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "lpips", None)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    with pytest.raises(RuntimeError):
        flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=1.0)


def test_controller_trainer_train_step_uses_mask_for_gradient_flow():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        rmse_weight=1.0,
        cosine_weight=0.0,
        lpips_weight=0.0,
        target_ttr_ratio=2.0,  # Disable efficiency penalty so RMSE path must drive gradients.
    )
    teacher = torch.zeros(1, 4, 4, 4)
    before = [p.detach().clone() for p in controller.parameters()]

    def _student_forward(mask: torch.Tensor, logits: torch.Tensor):
        del logits
        scale = mask.mean()
        student = scale * torch.ones_like(teacher)
        return {
            "student_latent": student,
            "actual_full_attn_ratio": scale,
        }

    metrics = trainer.train_step(
        sigma=0.5,
        cfg_scale=4.0,
        width=64,
        height=64,
        teacher_latent=teacher,
        student_forward_fn=_student_forward,
    )
    assert metrics["loss"] > 0.0
    assert metrics["mask_mean"] >= 0.0
    assert metrics["layers_full_ratio"] >= 0.0

    after = list(controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))


def test_controller_trainer_train_step_requires_student_latent():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0)
    teacher = torch.zeros(1, 4, 2, 2)

    def _bad_forward(mask: torch.Tensor, logits: torch.Tensor):
        del mask, logits
        return {"actual_full_attn_ratio": 0.5}

    with pytest.raises(ValueError):
        trainer.train_step(
            sigma=0.3,
            cfg_scale=2.0,
            width=32,
            height=32,
            teacher_latent=teacher,
            student_forward_fn=_bad_forward,
        )
