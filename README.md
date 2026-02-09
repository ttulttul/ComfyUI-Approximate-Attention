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
