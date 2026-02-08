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
    assert metrics["actual_ttr_ratio"] == pytest.approx(0.6)
    assert metrics["target_ttr_ratio"] == pytest.approx(0.7)
    assert metrics["target_full_attn_ratio"] == pytest.approx(0.3)
    assert "rmse" in metrics
    assert "cosine_distance" in metrics


def test_controller_trainer_efficiency_penalty_uses_ttr_target_semantics():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0, target_ttr_ratio=0.7)
    teacher = torch.zeros(1, 4, 4, 4)
    student = torch.ones_like(teacher) * 0.25
    _, metrics = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.9,
        include_efficiency_penalty=False,
    )

    assert metrics["target_ttr_ratio"] == pytest.approx(0.7)
    assert metrics["target_full_attn_ratio"] == pytest.approx(0.3)
    assert metrics["actual_ttr_ratio"] == pytest.approx(0.1)
    assert metrics["efficiency_penalty"] == pytest.approx(0.6)


def test_controller_trainer_lpips_requires_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "lpips", None)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    with pytest.raises(RuntimeError):
        flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=1.0)


def test_controller_trainer_training_config_overrides_defaults():
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    training_config = {
        "loss_config": {"rmse_weight": 2.5, "cosine_weight": 0.75, "lpips_weight": 0.0},
        "optimizer_config": {"learning_rate": 3e-4, "grad_clip_norm": 0.25},
        "schedule_config": {"target_ttr_ratio": 0.4},
    }
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        training_config=training_config,
        learning_rate=1e-2,
        rmse_weight=7.0,
        cosine_weight=7.0,
        lpips_weight=0.0,
        target_ttr_ratio=0.9,
        grad_clip_norm=3.0,
    )
    assert trainer.rmse_weight == pytest.approx(2.5)
    assert trainer.cosine_weight == pytest.approx(0.75)
    assert trainer.target_ttr_ratio == pytest.approx(0.4)
    assert trainer.grad_clip_norm == pytest.approx(0.25)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_controller_trainer_compute_loss_can_skip_efficiency_penalty():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0, target_ttr_ratio=0.2)
    teacher = torch.zeros(1, 4, 4, 4)
    student = torch.ones_like(teacher) * 0.25

    loss_with_penalty, metrics_with_penalty = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.9,
        include_efficiency_penalty=True,
    )
    loss_without_penalty, metrics_without_penalty = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.9,
        include_efficiency_penalty=False,
    )

    assert metrics_with_penalty["efficiency_penalty"] == pytest.approx(metrics_without_penalty["efficiency_penalty"])
    assert loss_with_penalty.item() > loss_without_penalty.item()


def test_controller_trainer_reinforce_step_updates_parameters():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        target_ttr_ratio=0.6,
        grad_clip_norm=1.0,
    )
    before = [p.detach().clone() for p in controller.parameters()]

    metrics_a = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        reward=1.5,
        actual_full_attn_ratio=0.5,
    )
    metrics_b = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        reward=-0.5,
        actual_full_attn_ratio=0.25,
    )

    after = list(controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert "policy_loss" in metrics_a
    assert "total_loss" in metrics_a
    assert metrics_a["reward"] == pytest.approx(1.5)
    assert metrics_a["reward_baseline"] == pytest.approx(1.5)
    assert metrics_b["reward_baseline"] < metrics_a["reward_baseline"]


def test_controller_trainer_reinforce_penalty_uses_full_attention_budget():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)
    with torch.no_grad():
        for param in controller.parameters():
            param.zero_()
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=0.0,
        target_ttr_ratio=0.7,
        grad_clip_norm=1.0,
    )
    metrics = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 1.0, 1.0]),
        reward=0.0,
        actual_full_attn_ratio=1.0,
    )

    assert metrics["target_ttr_ratio"] == pytest.approx(0.7)
    assert metrics["target_full_attn_ratio"] == pytest.approx(0.3)
    assert metrics["probs_mean"] == pytest.approx(0.5)
    assert metrics["expected_full_attn_ratio"] == pytest.approx(0.5)
    assert metrics["expected_ttr_ratio"] == pytest.approx(0.5)
    assert metrics["efficiency_penalty"] == pytest.approx(0.2, abs=1e-6)


def test_controller_trainer_reinforce_step_works_under_inference_mode():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, learning_rate=1e-2, target_ttr_ratio=0.5)
    before = [p.detach().clone() for p in controller.parameters()]

    with torch.inference_mode():
        metrics = trainer.reinforce_step(
            sigma=0.6,
            cfg_scale=3.5,
            width=64,
            height=64,
            sampled_mask=torch.tensor([1.0, 0.0, 1.0]),
            reward=0.8,
            actual_full_attn_ratio=2.0 / 3.0,
        )

    after = list(controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert "total_loss" in metrics
    assert metrics["reward"] == pytest.approx(0.8)


def test_controller_trainer_rebuilds_controller_created_in_inference_mode():
    with torch.inference_mode():
        controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)

    trainer = flux2_ttr_controller.ControllerTrainer(controller, learning_rate=1e-2, target_ttr_ratio=0.5)
    assert not any(
        flux2_ttr_controller.ControllerTrainer._is_inference_tensor(p)
        for p in trainer.controller.parameters()
    )

    before = [p.detach().clone() for p in trainer.controller.parameters()]
    metrics = trainer.reinforce_step(
        sigma=0.7,
        cfg_scale=2.5,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 0.0, 1.0]),
        reward=0.4,
        actual_full_attn_ratio=2.0 / 3.0,
    )
    after = list(trainer.controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert "total_loss" in metrics


def test_controller_trainer_train_step_uses_mask_for_gradient_flow():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        rmse_weight=1.0,
        cosine_weight=0.0,
        lpips_weight=0.0,
        target_ttr_ratio=0.0,  # Disable efficiency penalty so RMSE path must drive gradients.
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
