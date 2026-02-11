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

## Network Overview

### Phase 1: TTR Attention Network (`Flux2HKRAttnLayer`)

Each patched Flux single-attention block gets a trainable TTR layer with two branches:

- Kernel-regression branch (`KernelRegressorAttention`):
  - Learns positive feature maps with a 3-layer MLP (`head_dim -> hidden -> hidden -> feature_dim`, `hidden=max(head_dim, 2*feature_dim)`).
  - Uses split Q/K phi networks by default.
  - Streams K/V in chunks to build linear-time statistics (`Ksum`, `K^T V`) and evaluates Q in chunks.
- Landmark softmax branch:
  - Always keeps all conditioning tokens as landmarks.
  - Adds image-token landmarks from a dynamic budget (`landmark_fraction`, `landmark_min`, `landmark_max`; `landmark_max=0` means unlimited).
- Sigma/CFG conditioner:
  - Small MLP predicts per-branch scale/bias modulation (`kernel` and `landmark` branches).
  - Initialized to identity behavior for backward-compatible startup.

Branch fusion is residual and strictly convex by construction:

```text
out = out_kernel + alpha * (out_land - out_kernel)
```

- Non-adaptive mode: `alpha = alpha_max * sigmoid(base_logit)`
- Adaptive mode: `alpha_token = alpha_max * sigmoid(base_logit + gamma * (d_norm - 0.5))`
- `d_norm` comes from cosine disagreement (`1 - cosine_similarity(kernel, landmark)`), which avoids pure magnitude bias.
- `alpha` is always in `(0, alpha_max)` (default `alpha_max=0.2`), so the landmark branch stays a bounded correction.

### Phase 1 Training Mechanics (Online Distillation)

Training is performed online during sampling:

- Teacher targets are native Flux attention outputs on real prompts/sampling states.
- Replay stores query-subsampled examples with full keys/values:
  - `(q_sub, k_full, v_full, teacher_sub, masks, sigma, cfg)`
  - Query-only subsampling keeps global context while reducing training cost.
- Replay is CPU-offload friendly with configurable storage dtype and a global byte budget + eviction.
- Per-step layer selection is randomized (`min_swap_layers` / `max_swap_layers`) so training coverage remains broad.
- Readiness is fail-closed:
  - Inference uses teacher attention until a layer has enough updates and low EMA cosine distance.
  - Hysteresis (`exit = threshold * 1.2`) reduces ready/not-ready flapping.

Distillation loss (per replay update) is uncertainty-weighted multi-task optimization:

```text
huber = SmoothL1(student, teacher; beta=huber_beta)
cosine_term = 1 - cosine_similarity(student, teacher, dim=-1).mean()

loss =
  huber / (2 * exp(log_var_huber)) + log_var_huber / 2
  + cosine_term / (2 * exp(log_var_cosine)) + log_var_cosine / 2
```

- `log_var_huber` and `log_var_cosine` are learned scalars (Kendall-style weighting).
- Layer weights and loss-weight scalars have persisted optimizer state in checkpoints for stable resume behavior.

### Phase 2: Controller Network (`TTRController`)

The controller predicts a per-layer routing logit for each diffusion step:

- Inputs: `sigma`, `cfg_scale`, latent `width`, latent `height`.
- Embeddings:
  - Sinusoidal scalar embeddings for `sigma` and `cfg`.
  - Learned resolution embedding from sinusoidal `width`/`height`.
- MLP head: 3 linear layers to `num_layers` logits.
- Semantics:
  - High logit/probability means full attention.
  - Low logit/probability means route that layer through Phase-1 TTR.

### Phase 2 Training Mechanics (`Flux2TTRControllerTrainer`)

Controller training uses policy gradients with quality-driven rewards:

- Teacher path: sample with original model.
- Student path: sample with TTR model + controller routing.
- Quality objective (`compute_loss`) on latent/image outputs:
  - `rmse_weight * RMSE`
  - `+ cosine_weight * cosine_distance`
  - `+ lpips_weight * LPIPS` (optional)
- Reward shaping:
  - `reward_quality = -quality_loss`
  - `reward = reward_quality - lambda_eff * efficiency_penalty + lambda_entropy * entropy_bonus`
  - `efficiency_penalty = relu(actual_full_attn_ratio - target_full_attn_ratio)` where `target_full_attn_ratio = 1 - target_ttr_ratio`.
- REINFORCE update:
  - Policy loss is `-(reward - baseline) * log_prob(actions)` over eligible (ready) layers only.
  - Entropy bonus keeps policies from collapsing to saturated Bernoulli decisions.
  - Reward baseline + AdamW optimizer state are checkpointed/restored.

Sigma-aware mode uses a trajectory wrapper that logs per-step actions and recomputes sigma-weighted log-probs under grad-enabled context, matching how routing is actually used during denoising.

## Unique Implementation Ideas

- Fail-closed readiness gates at layer granularity, with EMA cosine-distance hysteresis.
- Query-only replay subsampling with full K/V context for better learnability.
- Learned uncertainty weighting between robust regression and directional alignment losses.
- Strictly bounded adaptive blend (`alpha_max`) with logit-space modulation from cosine disagreement.
- Dynamic landmark policy that always preserves conditioning tokens and scales image landmarks with resolution.
- Inference-mode-safe training: explicit `torch.inference_mode(False)` / grad-enabled guards plus inference-tensor rebuild paths.
- Controller penalties and routing ratios computed over eligible layers (with forced-full layers tracked separately), avoiding biased policy pressure.
- Rich observability: per-layer + cross-layer quantile metrics, Pareto-style readiness frontier, persistent Comet experiment handling.

## Flux2TTR Notes

- `Flux2TTRControllerTrainer` supports `sigma_aware_training` (default `true`) for per-step sigma-dependent routing policy updates.
- Controller checkpoints persist trainer state (`reward_baseline`, `reward_count`, optimizer state) for stable resume behavior.
- `Flux2TTRController` exposes `quality_speed` to trade quality and speed through controller thresholding.
- Flux2TTR landmark selection always includes all conditioning tokens as landmarks; the dynamic landmark budget is applied only to image/spatial tokens, and `landmark_max=0` now means unlimited (the trainer node default).
- Runtime accepts conditioning token hints through `transformer_options` keys: `conditioning_token_count`, `cond_token_count`, or `prefix_token_count`.
- Comet logging now emits latest per-layer metrics for all tracked layers at each log tick, plus `flux2ttr/global/pareto_frontier` for ready-layer quality/coverage tracking.
- Phase-1 TTR Comet logging now supports persistent experiments across ComfyUI sampling runs when `comet_experiment` is set, reusing the same Comet run key instead of ending at each cleanup.
- Layer readiness now gates on `layer_ema_cosine_dist` with hysteresis (`exit = readiness_threshold * 1.2`) so layers do not flap at the readiness boundary.
- Phase-1 EMA updates for `ema_loss` and `ema_cosine_dist` are now flushed once per sampling run (run-mean on sigma boundary) instead of every train step, reducing prompt-to-prompt readiness oscillation.
- Phase-1 EMA accumulation also has a periodic fallback flush every 20 training updates so readiness/EMA progress continues even if sigma boundary detection does not fire.
- Each run-EMA flush now logs per-layer post-flush values (`ema_loss`, `ema_cosine_dist`) to the console for easier readiness debugging.
- Phase-1 replay training now uses learned uncertainty weighting across Smooth L1 and cosine-alignment tasks (`log_var_huber`, `log_var_cosine`) so neither objective dominates and the balance adapts during training.
- Phase-1 loss-weight initialization is now asymmetric (`log_var_huber=0.0`, `log_var_cosine=-1.0`) so cosine alignment starts with higher effective weight before Kendall balancing converges.
- Phase-1 cosine alignment now computes per-token/per-head similarity over head dimension (`dim=-1`) for both loss and logged metrics, preventing high-dimensional flattening from hiding hard-token directional errors.
- Phase-1 checkpoints now persist both per-layer AdamW optimizer states and the loss-weight optimizer state (`log_var_huber`/`log_var_cosine`) so cross-run resume preserves momentum/variance estimates instead of restarting optimizer warmup each run.
- Layer/optimizer restoration now strips inference-tensor metadata and auto-rebuilds stale inference-backed layers before replay training, preventing AdamW "inplace update to inference tensor" failures at run boundaries.
- Phase-1 Comet logging now emits learned loss-balance parameters as `flux2ttr/global/log_var_huber` and `flux2ttr/global/log_var_cosine` at each log tick.
- Phase-1 Comet logging now emits per-layer `flux2ttr/<layer>/alpha_sigmoid` and cross-layer aggregates for adaptive alpha monitoring.
- Distill snapshot logs now include the current learned scalar loss-weight parameters (`log_var_huber`, `log_var_cosine`) for quick console-side monitoring.
- Loss-balance parameters are now rebuilt as normal trainable tensors if created under `torch.inference_mode()`, preventing Adam in-place update failures in ComfyUI runtime contexts.
- Layer blend `alpha` is now stored in logit-space and interpreted through `sigmoid`; forward pass supports error-driven adaptive per-token alpha gating, and old raw-alpha checkpoints are auto-migrated on load.
- Landmark/kernel blending now uses residual gating (`out_kernel + alpha * (out_land - out_kernel)`) so alpha controls interpolation toward landmark attention instead of adding an unconditional landmark residual.
- Adaptive alpha now uses cosine disagreement with logit-space shifts and an explicit cap (`alpha_max`, default `0.2`), so per-token blend weights stay strictly convex (`0 < alpha < alpha_max`) in both adaptive and non-adaptive paths.
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
