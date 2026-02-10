# Flux2TTR v2 Nodes for ComfyUI

This repository provides ComfyUI nodes for Flux2TTR v2 distillation and controller training.

Retired nodes:
- `TaylorAttentionBackend`
- `HybridTaylorAttentionBackend`

These legacy Taylor/hybrid nodes and their support modules are intentionally removed.

## Install

1. Copy this folder into your ComfyUI `custom_nodes` directory.
2. Restart ComfyUI.

## Available Nodes

- `Flux2TTRTrainingParameters`
- `Flux2TTRTrainer` (Phase 1: distill TTR layers)
- `Flux2TTRControllerTrainer` (Phase 2: train routing controller)
- `Flux2TTRController` (inference-time patching/routing)
- `RandomSeedBatch`
- `LoadPromptListFromJSON`
- `ClockedSweepValues`
- `Combinations`

## Flux2TTR v2 Workflow

Training:
- `Flux2TTRTrainingParameters -> Flux2TTRTrainer`
- `Flux2TTRTrainingParameters -> Flux2TTRControllerTrainer`

Inference:
- `MODEL -> Flux2TTRController -> KSampler`

## Flux2TTR Notes

- `Flux2TTRControllerTrainer` supports `sigma_aware_training` (default `true`) for per-step sigma-dependent routing policy updates.
- Controller checkpoints persist trainer state (`reward_baseline`, `reward_count`, optimizer state) for stable resume behavior.
- `Flux2TTRController` exposes `quality_speed` to trade quality and speed through controller thresholding.
- Flux2TTR landmark selection always includes all conditioning tokens as landmarks; the dynamic landmark budget is now applied only to image/spatial tokens.
- Runtime accepts conditioning token hints through `transformer_options` keys: `conditioning_token_count`, `cond_token_count`, or `prefix_token_count`.
- Comet logging now emits latest per-layer metrics for all tracked layers at each log tick, plus `flux2ttr/global/pareto_frontier` for ready-layer quality/coverage tracking.
- Phase-1 TTR Comet logging now supports persistent experiments across ComfyUI sampling runs when `comet_experiment` is set, reusing the same Comet run key instead of ending at each cleanup.
- Layer readiness now uses hysteresis (`exit = readiness_threshold * 1.2`) so layers do not flap at the readiness boundary.
- Phase-1 EMA updates for `ema_loss` and `ema_cosine_dist` are now flushed once per sampling run (run-mean on sigma boundary) instead of every train step, reducing prompt-to-prompt readiness oscillation.
- Phase-1 EMA accumulation also has a periodic fallback flush every 20 training updates so readiness/EMA progress continues even if sigma boundary detection does not fire.
- Phase-1 replay training now optimizes a composite objective `smooth_l1 + (1 - cosine_similarity)` so the student is trained directly on both magnitude and directional alignment.
- Controller inference now logs per-step routing summaries (extracted sigma, controller threshold, and student-routed layer set) once per step.
- `Flux2TTRController` supports `policy_mode` (`stochastic` or `threshold`). The default `stochastic` mode samples one controller mask per diffusion step (cached for all layer calls in that step) to match sigma-aware policy training behavior.
- The HKR phi feature map MLP now uses split Q/K networks by default and a 3-layer shape (`head_dim -> hidden -> hidden -> feature_dim`) with two SiLU activations, where `hidden = max(head_dim, 2 * feature_dim)`, increasing kernel expressivity versus the previous 2-layer mapping.

## Utility Nodes

### ClockedSweepValues

Maps a clock list to evenly distributed sweep values. Output length matches the clock length.

### RandomSeedBatch

Generates a deterministic list of integer seeds from a base seed.

### LoadPromptListFromJSON

Loads a JSON `array<string>` file into a prompt list output.

### Combinations

Builds repeated float-list outputs across up to four input value lists to cover Cartesian-style combinations.

## Tests

```bash
pytest -q
```

Using `uv`:

```bash
uv sync --extra test
uv run pytest
```

## Build Paper

Build `docs/flux2ttr_v2_paper.tex` into `docs/flux2ttr_v2_paper.pdf`:

```bash
scripts/build_paper.sh
```

Helpful options:

```bash
scripts/build_paper.sh --dry-run
scripts/build_paper.sh --engine latexmk
scripts/build_paper.sh --engine pdflatex --clean
```
