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
