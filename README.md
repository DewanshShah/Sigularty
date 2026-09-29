# Sigularty - Reference

sigularty is a PyTorch model compression toolkit. It runs any PyTorch model through
a configurable pipeline of compression techniques and produces a smaller, faster
model with a full evaluation report. Model-agnostic - works on CNNs, Transformers,
and NLP models, including architectures it's never seen.

This file covers what's specific to sigularty's own implementation - pipeline
mechanics, gating, search algorithms, CQI, and per-technique config. It assumes
you already know what pruning, quantization, distillation, BN fusion, etc. are.

For installation and learning the "sigularty" package refer to it's [PyPI page](https://pypi.org/project/sigularty/)

Here are compression results on some models (The results are achieved by the algorithm itself, no human fine-tuning was involved): 

Constants used for each one (Better results can be achieved by using a better learning rate): `train_sample = 3000, accuracy_drop_threshold = 10`

| Model | Acc Drop | Compression Ratio | CQI | Speedup | Size (MB) | Accuracy (%) | Latency (ms) |
| :--- | ---: | ---: | ---: | :---: | :---: | :---: | :---: |
| `Resnet50` | **2.20%** | **8.00x** | **9.897** | **1.28x** | 90.66 -> 11.34 | 66.60 -> 64.40 | 6.977 -> 5.451 |
| `vit_b_16` | **0.00%** | **4.51×** | **4.526**| **1.00x** | 327.59 -> 72.67 | 77.00 -> 77.00 | 14.96 -> 14.90 |
| `efficientnet_b0` | **2.80%** | **2.61x** | **2.284**| **0.90x** | 15.95 -> 6.11 | 91.80 -> 89.00 | 8.238 -> 9.124|
| `bert_base` | **4.20%** | **6.61x** | **6.449** | **1.02x** | 417.66 -> 67.17 | 86.20 -> 82.00 | 10.393 -> 10.209 |
| `distilbert` | **5.60%** | **9.55x** | **7.049** | **0.78x** | 255.42 -> 26.75 | 82.40 -> 76.80 | 5.441 -> 6.957 |
| `distilgpt2` | **0.00%** | **2.00x** | **2.903** | **1.45x** | 312.48 -> 156.24 | 87.20 -> 87.20 | 8.559 -> 5.896 |

---

## File Map

```
python main.py                  ← ALWAYS the entry point. Never run anything else.

main.py                         ← Hyperparameter constants + main() wiring only.
                                   Never grows beyond ~200 lines.

compression.py                  ← All 9 compression algorithms, pure logic.
                                   No pipeline orchestration. No plotting.

optimization.py                 ← Hyperparameter search (epsilon + pruning).
                                   Anchor sampling + ternary search + CQI scoring.

helper_functions.py             ← Everything else: data loading, model loading,
                                   training, accuracy/latency measurement, ONNX
                                   export, pipeline orchestration, CLI parsing.

visualization.py                ← All PNG report generation.
                                   Never runs models. Only renders data it receives.

model_registry.py               ← 20 model definitions.
                                   Loader functions + metadata + recommended configs.
```

- All plotting lives exclusively in `visualization.py`.
- All compression logic lives exclusively in `compression.py`.
- `helper_functions.py` owns everything else (data, training, evaluation, orchestration).
- `main.py` owns only constants and is the sole source of truth for every
  hyperparameter - changing a constant there always takes effect.

---

## Pipeline Order - Fixed

```
BatchNorm Fusion                                              [never gated - lossless]
    ↓
Structured Pruning  (activation-statistics importance via Torch-Pruning)  [gated]
    ↓
Low-Rank Factorization  (standard global epsilon or adaptive per-layer SVD energy)  [gated]
    ↓
Weight Clustering  (GPU k-means, stays on-device)              [gated]
    ↓
GPTQ INT4/INT8  (Hessian-corrected, Linear layers only)         [gated]
    ↓
Quantization (fp16 / dynamic INT8)                              [gated]
    ↓
Knowledge Distillation Fine-tuning  (original model as frozen teacher)  [gated + extra
                                                                          recovery check]
```

**[gated]** = measured before/after; reverted to the pre-technique state if that
one technique's own marginal accuracy drop exceeds `ACCURACY_DROP_THRESHOLD` - see
[Accuracy-Drop Gating](#accuracy-drop-gating--per-technique-impact) below.

**Why this order:**
- BN Fusion first - its parameters would otherwise get pruned/factorized/clustered
  for nothing, since they disappear after fusion.
- Pruning before LRF - LRF wraps layers in `nn.Sequential`, which breaks
  Torch-Pruning's dependency-graph trace.
- LRF and Clustering before Quantization - SVD and k-means both need float32 weights.
- KD runs last, after quantization/GPTQ - it needs the final architecture and
  precision fixed before training against the teacher, so it recovers accuracy
  lost to every prior step at once.

BN Fusion, Pruning, LRF, and Clustering are gated inside `apply_compression_pipeline`;
GPTQ, Quantization, and the final KD step are gated separately in
`run_compression_pipeline`, since dynamic INT8 has no CUDA kernel and needs to run
after the float32 stages for latency comparisons to stay fair.

All final metrics (accuracy, size, latency) are measured on the actually-deployed
compressed model, never an intermediate one.

---

## Accuracy-Drop Gating & Per-Technique Impact

`ACCURACY_DROP_THRESHOLD` (main.py) does two things:
1. Selects the winning config in both hyperparameter searches.
2. Gates the real pipeline: after every mutating technique runs, accuracy is
   measured before/after; if the marginal drop exceeds the threshold, that
   technique's result is discarded and the pipeline reverts to its pre-technique
   state.

Each technique has an **independent** budget - an earlier technique's cost doesn't
eat into a later one's allowance. BN Fusion and Sensitivity Analysis are never
gated (lossless / read-only respectively). A reverted technique doesn't appear in
the final report's technique list.

**Per-technique impact reporting** exists because a technique can be a structural
no-op and still show a real accuracy *improvement* in the gate log - the
improvement came entirely from that technique's own bundled recovery fine-tune,
not the technique itself. For every gated technique, an `[Impact]` block reports:
- structural summary (what actually changed, from that technique's own report dict)
- algorithm-vs-fine-tune accuracy split (Pruning/LRF/Clustering only - these bundle
  a fine-tune inside the same call the gate measures around)
- marginal (vs. immediately-prior state) and cumulative (vs. absolute original)
  accuracy/size/latency
- an explicit warning when the structural change was zero

```
[Impact] Low-Rank Factorization
  Structural : 0/7 eligible layers factorized (kernel>1×1, skip_large_kernels=True)
  Algorithm  : 71.00% → 71.00%  (0.00pp — raw structural effect)
  Fine-tune  : 71.00% → 73.40%  (+2.40pp — recovery fine-tune's own contribution)
  Result     : kept
  ⚠️ Zero structural effect — the accuracy gain above is entirely the fine-tune.
```

```python
ACCURACY_DROP_THRESHOLD = 10.0   # one value, used everywhere above
```

---

## Early-Abort for Recovery Fine-Tunes

A separate, earlier check than the gate above: after epoch 1 of Pruning's or
(Adaptive) LRF's recovery fine-tune, if the drop vs. the *original* model already
exceeds `FINE_TUNE_ABORT_THRESHOLD` (a direct pp value), the remaining epochs are skipped - the gate would very
likely revert it anyway. Applies to both the real pipeline and the epsilon/pruning
searches, since all four share `_kd_recovery_fine_tune()`. Not applied to
Clustering's or the final KD step's fine-tunes.


```python
FINE_TUNE_ABORT_THRESHOLD = 40.0   # If the accuracy drop after the first epoch is greater than 40, all remaining epochs will be  skipped
```

---

## Major Fine-Tune Epoch-Zero Safety Check

Every fine-tuning loop in the toolkit shares one check: if epoch 1 reports
**exactly 0.0%** training accuracy, a warning prints and the remaining epochs are
skipped - a mathematically dead gradient signal won't improve with more epochs, so
continuing would only waste compute. The model is still returned (broken or not);
the accuracy-drop gate is what actually discards it.

`fine_tune_with_distillation` always tracks training accuracy regardless of
whether `test_loader` is supplied, and tracks test accuracy when it is, attaching
both as `._kd_history = [{epoch, loss, train_acc, test_acc}, ...]` - this is what
the final KD step's recovery-quality retry reads (see
[KD's Cumulative Recovery Check](#kds-cumulative-recovery-check)).

---

## measure_accuracy: Structural vs. Genuine Failure

`measure_accuracy()` distinguishes two outcomes:
- **Structural failure** - every batch raised an exception. Raises `RuntimeError`
  with `"No samples were evaluated"` in the message, rather than returning a fake
  `0.0%`. Search loops detect this substring to trigger the
  [structural-failure abort](#epsilon-search-lrf) instead of grinding through more
  guaranteed-broken trials. Ordinary per-batch tolerance for isolated shape issues
  is unchanged - this only changes what happens when *everything* fails.
- **Genuine zero** - at least one batch succeeded but nothing was predicted
  correctly. Returned normally as `0.0%`, with a loud warning, since this is a
  real (bad) result rather than a structural break.

---

## LRF + nn.MultiheadAttention Safety

LRF cannot factorize `nn.MultiheadAttention`'s `out_proj`: its `forward()` reaches
directly into `.weight`/`.bias` as raw tensors
(`F.multi_head_attention_forward(..., self.out_proj.weight, ...)`) rather than
calling `out_proj.forward()`, so wrapping it in `nn.Sequential` (what factorization
does) breaks every forward pass.

`apply_low_rank_factorization()` / `apply_adaptive_lrf()` detect any
`nn.MultiheadAttention` in the model and return it completely unchanged.
`run_compression_pipeline()` checks this before the pipeline starts and
force-disables `USE_LOW_RANK` + `FIND_OPTIMAL_EPSILON` for the run, avoiding a
wasted epsilon search entirely rather than running it only to have every trial
no-op.

Doesn't affect Swin-T (plain Q/K/V `nn.Linear`, not `nn.MultiheadAttention`) or any
HuggingFace NLP model in the registry (separate Q/K/V projections, same reason).

---

## Compression Techniques

### BatchNorm Fusion

Folds `Conv2d/Linear → BatchNorm` into a single layer:
```
W' = W · γ/√(σ²+ε)
b' = (b-μ) · γ/√(σ²+ε) + β
```
BN is replaced with `nn.Identity()`. Bit-for-bit identical output - never gated, no
fine-tune, no hyperparameters beyond the on/off switch. Requires BN in eval mode
(populated running stats); the function calls `model.eval()` internally.

Note: This only works if the model already has bn_layers, or it will do nothing.

```python
USE_BN_FUSION = True   # no reason to disable
```

---

### Structured Pruning

Removes low-importance output channels from Conv2d layers via Torch-Pruning's
dependency-graph tracing, which propagates channel removal through skip
connections, SE blocks, and multi-path branches without architecture-specific code.

**Importance metric:**
```
importance(filter_i) = mean over calibration batches of mean(|activation_i(x)|) over spatial positions
```
Activation statistics are used instead of L1/L2 weight magnitude because magnitude
reflects the learned parameter, not how much it actually matters on real data - a
metric that has to work across arbitrary, unknown architectures can't assume
anything about their regularization or initialization.

**Global pruning + per-group protection.** One target sparsity is set across the
whole network; importance scores decide per-layer ratios. Left unprotected, this
can collapse an individual layer or coupled group to a handful of channels even at
a modest overall ratio - most dangerously for groups whose channel count forms the
residual stream for an entire network stage. Sigularty detects these via
Torch-Pruning's own dependency graph: a group is flagged as residual-coupled when
it contains a genuine elementwise-addition node, identified by the node's
`grad_fn` class name (`'Add...'`) rather than just its `OPTYPE.ELEMENTWISE`
category, which ReLU and other activations also share. Flagged groups - and all
SE-attention layers (`ratio=0`, always; pruning SE's fc1/fc2 breaks the attention
mechanism entirely) - get an explicit per-module cap via `pruning_ratio_dict`.
`MetaPruner`'s own `max_pruning_ratio` kwarg does not reliably hold a specific
group under a ceiling on its own.

**Safety clamp by `model_type`:**

| model_type | max ratio | why |
|---|---|---|
| `classifier` | 0.5 | tolerates aggressive pruning |
| `embedding` | 0.2 | geometry breaks silently downstream (e.g. cosine-similarity retrieval) |
| `generative` | 0.1 | extremely sensitive |
| `unknown` | 0.2 | conservative default |

**Recovery:** always knowledge distillation against the original (pre-pruning)
model, via the shared `_kd_recovery_fine_tune()` helper.

**Behavioral probe:** compares original vs. pruned outputs on calibration data -
KL divergence for classifiers, cosine similarity otherwise. `<0.01` negligible ·
`<0.05` acceptable · `<0.15` moderate · `≥0.15` high.

Gated like every technique - see
[Accuracy-Drop Gating](#accuracy-drop-gating--per-technique-impact).

```python
PRUNING_RATIO              = 0.3    # global fraction of channels removed
PRUNING_MODEL_TYPE         = 'classifier'
PRUNING_FINE_TUNE_EPOCHS   = 3       # 0 = skip recovery
PRUNING_FINE_TUNE_LR       = 0.0001
PRUNING_CAL_BATCHES        = 50      # activation-statistics calibration batches
PRUNING_ITERATIVE_STEPS    = 1       # >1 = prune in N incremental rounds, more stable/slower
PRUNING_ROUND_TO           = None    # e.g. 8/16 for Tensor Core alignment
PRUNING_ISOMORPHIC         = False   # force identical structure across coupled groups
PRUNING_MAX_RATIO          = 0.5     # per-group ceiling, via pruning_ratio_dict
PRUNING_RESIDUAL_MAX_RATIO = None    # ceiling for residual-coupled groups; None = falls back to PRUNING_MAX_RATIO
PRUNING_REPORT_PATH        = 'pruning_report.png'
```

---

### Low-Rank Factorization

Replaces `Linear(in, out)` with `Linear(in, rank, bias=False) → Linear(rank, out,
bias=True)` via truncated SVD, and `Conv2d` similarly via its reshaped kernel.
`rank = int(min(in_dim, out_dim) * epsilon)`.

Skipped: depthwise/grouped convs (`groups>1` - block-diagonal weight, no
cross-channel mixing to decompose), any layer with `in`/`out` ≤
`LRF_MIN_LAYER_SIZE`, any layer whose computed rank < `LRF_MIN_RANK`, the output
head (always - it's typically well under 0.1% of parameters, and every learned
feature funnels through it, making it disproportionately sensitive to rank
reduction), and the whole model if it contains `nn.MultiheadAttention` (see
above).

Factorizing replaces one kernel launch with two - on 3×3+ convolutions the extra
launch overhead outweighs the smaller compute, so latency regresses even as
parameters drop; on 1×1 convolutions both matmuls are genuinely smaller and it's a
net win.

```
LRF_SKIP_LARGE_KERNELS = False   # EfficientNet, MobileNet — mostly 1×1 already
                       = True    # ResNet, VGG, DenseNet, Inception — heavy 3×3
```

Recovery: optional, always KD against the original model via
`_kd_recovery_fine_tune()` (reuses `KD_TEMPERATURE`/`KD_ALPHA`); `0` epochs = skip.

```python
USE_LOW_RANK           = True
LRF_EPSILON            = 0.5    # rank ratio kept; only used when LRF_ADAPTIVE=False
LRF_MIN_LAYER_SIZE     = 64
LRF_MIN_RANK           = 2
LRF_SKIP_LARGE_KERNELS = True
LRF_FINE_TUNE_EPOCHS   = 3      # 0 = no recovery
LRF_FINE_TUNE_LR       = 0.00001
```

#### Adaptive LRF

Computes epsilon per layer analytically instead of one global value:
`energy_fraction(r) = sum(S[:r]²) / sum(S²)` (S = singular values); finds the
minimum `r*` with `energy_fraction(r*) ≥ LRF_ENERGY_THRESHOLD`, then
`epsilon* = r* / min(in_dim, out_dim)`. No data or forward passes needed - weight
tensors only. Same output-head protection, `nn.MultiheadAttention` skip, and
recovery fine-tune as standard LRF.

```python
LRF_ADAPTIVE         = True
LRF_ENERGY_THRESHOLD = 0.99   # fraction of SVD energy retained per layer
```

---

### Weight Clustering

Runs k-means per layer, replaces it with `_ClusteredLinear`/`_ClusteredConv2d`: `k`
trainable centroids + a fixed, packed per-weight assignment (4-bit for k≤16, uint8
otherwise). Weight is reconstructed each forward as `centroids[assignment]`.
`get_model_size_mb` sums these as real parameters/buffers, so the compression
ratio (~8× at k≤16, ~4× at k=256) is measured, not aspirational. Doesn't speed up
inference by itself - it's a storage format change; both layer types still fully
dequantize before the same float op.

`centroids` is the only trainable piece - `assignment` is a fixed buffer - so
recovery fine-tuning genuinely only moves centroid values; backprop through
`centroids[assignment]` sums gradients from every weight sharing a centroid (same
mechanism as `nn.Embedding`).

Clustering runs before GPTQ in the fixed pipeline order; `_ClusteredLinear`
exposes `.weight`/`.bias`/`.in_features`/`.out_features` like a real `nn.Linear`,
so GPTQ reads a real dense weight from it and replaces the whole layer with
`_Int4Linear` - clustering becomes pre-conditioning for GPTQ's Linear layers, and
stays the final storage form for Conv2d (GPTQ never touches Conv2d).

```python
USE_CLUSTERING           = True
CLUSTER_NUM_CLUSTERS     = 16     # ≤16 → 4-bit packed; >16 → uint8
CLUSTER_FINE_TUNE_EPOCHS = 5      # 0 = skip
CLUSTER_FINE_TUNE_LR     = 0.0001
```

---

### Knowledge Distillation Fine-tuning

The sole fine-tuning method used everywhere in the toolkit - Pruning, LRF,
Clustering, and the final post-quantization step all use this against the
original model as a frozen teacher. Plain cross-entropy only survives as a
fallback for direct/library callers who omit a teacher.

```
loss = α · CE(student, hard_labels) + (1-α) · T² · KL(softmax(teacher/T) ‖ softmax(student/T))
```

```python
USE_KD_FINETUNE = True
KD_EPOCHS       = 3
KD_LR           = 0.0001
KD_TEMPERATURE  = 4.0   # 2–6 typical; higher = softer, more inter-class signal
KD_ALPHA        = 0.7   # hard-label weight; 1-alpha = distillation weight
```
Reused by Pruning's and LRF's own recovery fine-tunes - one T/α pair governs every
KD-based fine-tune in the toolkit.

#### KD's Cumulative Recovery Check

The final KD step is gated like every other technique, but that marginal check
rarely catches anything meaningful for a step whose whole purpose is recovery. A
second, separate check runs after KD survives its own gate: if the **cumulative**
drop vs. the absolute original baseline still exceeds `ACCURACY_DROP_THRESHOLD`,
the per-epoch test-accuracy trend (`._kd_history`) is inspected for exactly one
bounded retry, restarting fresh from the pre-KD checkpoint:
- **Stagnant** (avg epoch-to-epoch change within ±0.5pp) → halve `KD_LR`, retry once.
- **Small but consistent improvement** (every delta positive, averaging <2pp/epoch)
  → raise `KD_LR` by 50%, retry once.
- **Anything else** (erratic, net-negative, or already a large jump) → accepted
  as-is, no retry.

The retry's result is accepted unconditionally regardless of outcome - no second
retry, no gate re-applied to it.

---

### Quantization

**fp16** - casts everything to `torch.float16`; GPU/Tensor-Core speedup, no
calibration needed.

**dynamic INT8** - Linear layers only (no CUDA INT8 kernel for Conv2d in
PyTorch); dequantizes just before each matmul. Uses a manual tree-walking
implementation (`_manual_dynamic_quantize`) rather than
`torch.ao.quantization.quantize_dynamic`, since that dispatcher can't handle the
`nn.Sequential` wrappers LRF leaves behind.

**static INT8** - auto-switched to dynamic: residual additions (`tensor + tensor`
→ `aten::add`) have no QuantizedCPU kernel and crash after static conversion.

Gated like every technique; on revert the model correctly stays on whatever
device it was on before quantization ran (dynamic quant's CPU move never happened
to the model that gets kept).

```python
USE_QUANTIZATION              = True
QUANT_MODE                    = "fp16"     # "dynamic" for CPU; "static" auto-switches to dynamic
QUANT_NUM_CALIBRATION_BATCHES = 100        # static only
```

---

### GPTQ

Hessian-corrected INT4/INT8 on Linear layers: estimates `H ≈ 2·XᵀX` from
calibration activations, quantizes column-by-column, and propagates each column's
quantization error into the remaining columns via `H⁻¹` (Cholesky) - this is what
keeps INT4 viable (~0.5-2% accuracy drop vs. 5-15% for independent rounding).
Weights are packed as real INT4 (`_Int4Linear`, 2 values/byte, genuine 8×
reduction) plus a per-block fp16 scale; compute still dequantizes to float before
each forward - a storage win today, kernel-ready for a true INT4 compute path
later.

```python
USE_GPTQ         = True
GPTQ_BITS        = 4     # 4=INT4 (8×), 8=INT8 (4×)
GPTQ_CAL_BATCHES = 16    # Hessian estimation batches
GPTQ_BLOCK_SIZE  = 128
```

#### GPTQ + KD: trainable weights via STE

`_Int4Linear` stores everything as non-trainable buffers, so a plain Adam
optimizer can't touch GPTQ'd weights - which would leave the final KD recovery
step unable to recover anything GPTQ itself cost, on exactly the architectures
(ViT, BERT-family) where GPTQ covers most of the Linear layers.

`apply_gptq_quantization(..., qat=True)` (requested whenever `USE_KD_FINETUNE=True`)
produces `_Int4LinearQAT` instead: a real float32 `shadow_weight` parameter plus a
straight-through estimator -
```
w = shadow_weight + (fake_quantize(shadow_weight) - shadow_weight).detach()
```
- so the forward value is INT4-grid-snapped, but gradients land entirely on
`shadow_weight`. `collapse_qat_layers()` runs once after the final KD step (kept
or reverted either way) and converts every `_Int4LinearQAT` back to a frozen,
compact `_Int4Linear` built from the fine-tuned shadow values. While QAT is
active, size legitimately grows (the shadow weight is a real dense fp32 copy) -
the impact report's structural summary says so explicitly for exactly this
reason. Scoped to `bits=4` only; `bits=8` already leaves a plain trainable
`nn.Linear` behind.

---

## Search Algorithms

### Epsilon Search (LRF)

Anchor sampling (5 fixed epsilons: 0.1/0.25/0.5/0.75/0.9) + ternary-search
refinement around the best anchor, then a final selection pass that pools the
entire cache and picks the highest-scoring candidate within
`accuracy_drop_threshold` (ternary, not binary, because the goal is a maximum,
not a zero-crossing, and ternary's one-third/two-thirds split is what makes each
discard step valid for unimodal maximization).

Every trial measures accuracy, size, and latency (10 iterations, 3 warmup) - LRF
adds a kernel launch, so an epsilon can shrink the model and still score below
1.0 if it's launch-bound; the search then disables LRF for the run rather than
use a net-negative config. No INT8 proxy is used here: it would run on CPU, which
for a model like EfficientNet-B0 is roughly 10× slower than GPU float32 - using
it would make the search slower, not faster.

Two **consecutive** structural failures (`"No samples were evaluated"`) abort the
remaining trials across every phase; final selection still runs on whatever's
cached, and returns `None` only if literally nothing ever succeeded.

Cached to `epsilon_cache.json`, keyed by epsilon value only - delete it after
changing CQI weights or latency settings, or old and new scores mix under the
same key.

```python
FIND_OPTIMAL_EPSILON      = True
EPSILON_SEARCH_NUM_TRIALS = 15   # 5 anchors + (remaining/2) ternary iterations
```

### Pruning Hyperparameter Search

Searches `pruning_ratio` (anchor + ternary), `pruning_max_ratio` (grid at the best
ratio), and `iterative_steps` (1 vs 2) - every phase selects by the same tiered
rule: highest score within `accuracy_drop_threshold`, falling back to `+10pp`
relaxed, falling back to pure score-max only as a last resort. A final phase pools
every evaluation from the *entire* search (not just the winning branch of each
phase) and picks the highest-scoring candidate within threshold across all of it.

Scoring includes KL divergence from pruning's own behavioral probe
(`(1/(1+KL))^w_kl`), so the search also avoids configs that shift the output
distribution even when raw accuracy looks fine. Shares the same
2-consecutive-structural-failure abort as the epsilon search.

```python
FIND_OPTIMAL_PRUNING      = True
PRUNING_SEARCH_NUM_TRIALS = 15
PRUNING_SEARCH_FT_EPOCHS  = 3     # per-trial fine-tune epochs (fast approximation)
PRUNING_SEARCH_FT_LR      = 1e-4
```

---

## Compression Quality Index (CQI)

Single scoring metric used by both searches and the compression report:

```
CQI = accuracy_factor(x, accuracy_drop_threshold)
    × (baseline_size / size_mb)^w_size
    × (baseline_lat / lat)^w_latency      [when latency available]
    × (1 / (1 + KL))^w_kl                 [pruning only]
```

`accuracy_factor` is a three-piece piecewise function (`_accuracy_barrier_factor()`) - a saturating reward for
accuracy improvements, a gentle linear decline while still within the
accuracy-drop budget, and an unbounded exponential fall once past it, joined
continuously and driven by `ceiling` / `penalty_slope` / `penalty_steepness` -
replacing the older `acc_ratio ** w_accuracy` term (kept as a fallback for
legacy callers that don't pass `accuracy_drop_threshold`). The exact per-piece
formulas need the real `optimization.py` source before this section can state
them precisely - flagging rather than guessing at the equations.

Size, latency, and KL factors are unaffected by the above - still plain ratios
raised to their weight.

```python
CQI_W_SIZE     = 1.0
CQI_W_LATENCY  = 1.0
CQI_W_KL       = 1.0   # pruning only
```

Both searches measure real latency on every trial, so `CQI_W_LATENCY` affects
which config wins, not just the final report - a config that shrinks the model
but slows it down is scored accordingly. `ACCURACY_DROP_THRESHOLD` (not these
weights) is the hard gate that keeps a high `CQI_W_SIZE` from picking an
accuracy-destroying config regardless of how the weights are tuned.

---

## Other main.py Constants

Constants that don't belong to any single technique above.

```python
ACTIVE_MODEL = 'efficientnet_b0'   # see Model Registry below for all 20 options

PRETRAIN_MODEL_PATH    = f'models/{ACTIVE_MODEL}.pth'
REPORT_SAVE_PATH       = 'compression_report.png'
EPSILON_LANDSCAPE_PATH = 'epsilon_landscape.png'
ONNX_SAVE_PATH         = 'compressed_model.onnx'

TRAIN_SAMPLE_SIZE = 2000   # 0 = full training set
TEST_SAMPLE_SIZE  = 500    # 0 = full test set; also what every accuracy gate reads from
BATCH_SIZE        = 32
NUM_CLASSES       = 102    # fallback only — the registry sets this per model

PRETRAIN_EPOCHS = 10       # only used when no checkpoint exists
PRETRAIN_LR     = 0.001
FORCE_RETRAIN   = False

EXPORT_ONNX = False
RUN_ONNX    = False
ONNX_OPSET  = 17
```

---

## Model Registry - 20 Models

All models in `model_registry.py`. Use `python main.py --list-models` to print them.

| ACTIVE_MODEL | Architecture | Dataset | Params | Notes |
|---|---|---|---|---|
| **`efficientnet_b0`** | NAS/MBConv | flowers102 | 5.3M | Default. Depthwise convs skipped by LRF |
| `resnet18` | ResNet | cifar100 | 11.2M | Smallest ResNet. Skip 3×3 for LRF |
| **`resnet50`** | ResNet | cifar100 | 25.6M | Bottleneck blocks. Skip 3×3 for LRF |
| `resnext50_32x4d` | ResNeXt | cifar100 | 25.0M | Grouped convs auto-skipped by LRF |
| `wide_resnet50_2` | WideResNet | cifar100 | 68.9M | Clustering very effective |
| `vgg16` | VGG | cifar100 | 138.4M | ALL convs 3×3 - must skip for LRF |
| `densenet121` | DenseNet | cifar100 | 8.0M | Dense connections, careful pruning |
| `convnext_tiny` | ConvNeXt | cifar100 | 28.6M | 7×7 kernels - must skip for LRF |
| `mobilenet_v3_large` | MobileNet | cifar100 | 5.5M | Already optimized |
| `regnet_y_400mf` | RegNet | cifar100 | 4.3M | Small model, floor test |
| `shufflenet_v2_x1_0` | ShuffleNet | cifar100 | 2.3M | Tiny. LRF disabled in recommended |
| `squeezenet1_1` | SqueezeNet | cifar100 | 1.2M | Smallest. Only clustering+fp16 |
| `inception_v3` | Inception | cifar100 | 27.2M | Requires 299×299 input |
| **`vit_b_16`** | ViT | cifar100 | 86.6M | nn.MultiheadAttention - LRF and epsilon search auto-disabled at runtime, regardless of    USE_LOW_RANK/FIND_OPTIMAL_EPSILON |
| `swin_t` | Swin Transformer | cifar100 | 28.3M | Mostly Linear (NOT nn.MultiheadAttention). LRF very effective |
| **`bert_base`** | BERT | sst2 | 110.0M | Requires transformers+datasets |
| **`distilbert`** | DistilBERT | sst2 | 66.4M | Already distilled - does more help? |
| `roberta_base` | RoBERTa | sst2 | 125.0M | Uniform weights → clustering effective |
| `albert_base` | ALBERT | sst2 | 11.7M | Shared weights across layers |
| **`distilgpt2`** | GPT-2 | sst2 | 81.9M | Causal attention, classification |

Registry `recommended` settings can supply `pretrain_epochs`/`pretrain_lr` - every
other setting comes from `main.py`'s constants. For those two, precedence is:
explicit CLI flag (`--pretrain-epochs`/`--pretrain-lr`) > registry recommendation >
`main.py`'s own `PRETRAIN_EPOCHS`/`PRETRAIN_LR` constants. A flag value equal to its
argparse default (`10` epochs / `0.001` lr) can't be told apart from no flag at all,
so the registry value still applies in that case. The `nn.MultiheadAttention` check
is a separate runtime check, not part of `recommended` and not overridable by it.

---

## CLI Flags

All flags override the corresponding main.py constant for a single run.

```bash
# Enable / configure pruning
python main.py --pruning
python main.py --pruning --pruning-ratio 0.2
python main.py --pruning --pruning-model-type embedding
python main.py --pruning --pruning-fine-tune-epochs 5
python main.py --pruning --pruning-iterative-steps 2
python main.py --pruning --pruning-round-to 8
python main.py --pruning --pruning-max-ratio 0.3

# Disable individual techniques
python main.py --no-low-rank
python main.py --no-clustering
python main.py --no-quantization

# Configure LRF
python main.py --lrf-epsilon 0.3
python main.py --lrf-min-layer-size 32
python main.py --lrf-min-rank 4
python main.py --lrf-skip-large-kernels
python main.py --lrf-fine-tune-epochs 3
python main.py --lrf-fine-tune-lr 0.00001

# Configure clustering
python main.py --num-clusters 32
python main.py --cluster-fine-tune-epochs 10
python main.py --cluster-fine-tune-lr 0.00005

# Quantization
python main.py --quant-mode dynamic
python main.py --quant-mode fp16

# Data and training
python main.py --train-samples 0        # full dataset
python main.py --test-samples 0         # full test set
python main.py --force-retrain
python main.py --pretrain-epochs 20

# Model selection
python main.py --active-model vit_b_16
python main.py --list-models

# Search
python main.py --find-optimal-epsilon
python main.py --find-optimal-epsilon --epsilon-search-num-trials 30
python main.py --find-optimal-pruning
python main.py --find-optimal-pruning --pruning-search-num-trials 30
python main.py --find-optimal-pruning --fine-tune-abort-threshold 20

# ONNX
python main.py --export-onnx
python main.py --export-onnx --run-onnx
python main.py --export-onnx --onnx-path models/v2.onnx

# Device
python main.py --device cpu
```

---

## Things That Will Break and Why

**`aten::add.out has no QuantizedCPU kernel`**
Static quantization. The pipeline auto-switches to dynamic. If you call
`apply_quantization(model, mode='static')` manually: residual additions use
`tensor + tensor` which becomes `aten::add` after static quantization. This
op has no QuantizedCPU kernel. Fix: use `mode='dynamic'` or `mode='fp16'`.

**`apply_dynamic is not implemented for this packed parameter type`**
PyTorch 2.x `quantize_dynamic` crashes on `nn.Sequential` wrappers created by LRF.
The toolkit uses `_manual_dynamic_quantize` internally. If you call
`torch.ao.quantization.quantize_dynamic` directly after running LRF, you'll hit this.
Fix: use `apply_quantization(model, mode='dynamic')` instead.

**`Input type (FloatTensor) and weight type (HalfTensor) should be the same`**
You're passing float32 data to an fp16 model. `measure_accuracy` and `measure_latency`
auto-cast inputs. If you're calling `model(X)` directly:
`X = X.to(dtype=next(model.parameters()).dtype)`

**A technique you enabled doesn't show up in `techniques_used` / the report**
It was reverted by the global per-technique accuracy gate - check the console
output for a `❌ [TechniqueName] SKIPPED - accuracy dropped Xpp ...` line near
where that technique ran. This is the gating behaviour, not a bug - see
[Accuracy-Drop Gating](#accuracy-drop-gating--per-technique-impact). Raise
`ACCURACY_DROP_THRESHOLD` if you want to permit larger drops, or investigate
why that specific technique is causing such a large drop (wrong hyperparameter,
incompatible architecture, etc.) before raising the threshold blindly.

**`measure_accuracy: No samples were evaluated`**
Every batch in this evaluation raised an exception - almost always a
structural break, not ordinary bad accuracy. The most common cause is LRF
attempting to factorize a layer that some OTHER module accesses by reaching
directly into `.weight`/`.bias` (the exact `nn.MultiheadAttention.out_proj`
pattern described above). If you see this on an architecture NOT already
covered by the `nn.MultiheadAttention` check, it likely means some other
module in that architecture has the same "direct attribute access bypassing
forward()" pattern - inspect the architecture's source for similar direct
`.weight`/`.bias` access on a sub-module before assuming this is a generic bug.

**Epsilon or pruning search aborted after only 2 trials with "🛑 SEARCH ABORTED"**
Two consecutive structural failures. The search still completes (using whatever
succeeded before the abort), but if this happens on every run for a given model,
that model likely has a fundamental incompatibility with the technique being
searched (check for patterns like the `nn.MultiheadAttention` case, or verify
`num_classes` / input shape are actually correct for this model).

**Search results don't change after editing CQI weights or accuracy factor knobs**
Both searches cache trial results (`epsilon_cache.json`,
`pruning_search_cache.json`, `.sigularty_cache/*.json` via `compress()`) keyed by
hyperparameter value only - not by which CQI formula or knob values produced the
score. A cache written before a CQI-affecting change will silently feed
old-formula scores into a new-formula search under the same keys. Delete the
relevant cache file after any such change.

**LRF or pruning gets disabled by its search with a score below 1.0**
The search judged the technique worse than not compressing once size, accuracy,
and latency are combined. On launch-bound models LRF often shrinks the model
while slowing it down; the warning lists the winner's size and latency against
the baseline. Raise `CQI_W_SIZE` or lower `CQI_W_LATENCY` only if you are
deliberately optimizing for size over speed.

**Shape mismatch after pruning then LRF**
If you prune AFTER LRF (wrong order), the dependency graph trace fails because
Torch-Pruning can't propagate through nn.Sequential wrappers. Pruning must always
run before LRF. The pipeline enforces this order - only occurs if you call
functions directly out of order.

**`Could not derive example_input from dataloader`**
The dataloader is empty or its first batch raised an exception. Check that the
dataset downloaded correctly and the DataLoader returns `(X, y)` tuples. Also
check that `data/` directory has write permissions.

**`ImportError: torch-pruning is required for structured pruning`**
`pip install torch-pruning --break-system-packages` and restart.

**NLP models: `ImportError: No module named 'transformers'`**
`pip install transformers datasets --break-system-packages`

**OOM during clustering**
`apply_weight_clustering` deep-copies the model (2× memory) plus gradient memory
during fine-tuning. Fixes (try in order):
1. Reduce TRAIN_SAMPLE_SIZE
2. Reduce BATCH_SIZE
3. Set CLUSTER_FINE_TUNE_EPOCHS = 0

**GPTQ: `RuntimeError: linalg.cholesky: The factorization could not be completed`**
The Hessian matrix is near-singular - the calibration data doesn't provide enough
diversity to estimate all directions. Fix: increase GPTQ_CAL_BATCHES (try 32 or 64)
or add `percdamp=0.1` (higher damping regularizes the Hessian better).

**Pruning behavioral probe shows HIGH severity after search finds "optimal" config**
The search uses reduced fine_tune_epochs (`PRUNING_SEARCH_FT_EPOCHS`) for speed.
The final pipeline uses `PRUNING_FINE_TUNE_EPOCHS`. HIGH severity in the search
doesn't necessarily mean HIGH in the final run - and even if it does, the
pipeline's accuracy gate (not just the behavioral probe) will catch and revert
a genuinely bad result automatically. If severity is still HIGH in the final
run AND it somehow passed the accuracy gate (e.g. the KL drift is severe but
the raw accuracy number looks acceptable), reduce `PRUNING_RATIO` or increase
`PRUNING_FINE_TUNE_EPOCHS`.

**Accuracy is wrong after loading checkpoint**
Wrong NUM_CLASSES or wrong dataset transform. If you changed ACTIVE_MODEL or
NUM_CLASSES, set FORCE_RETRAIN = True to rebuild the checkpoint.

**Latency INCREASES after LRF on ResNet/VGG**
Expected if LRF_SKIP_LARGE_KERNELS = False. Two small kernel launches have more
overhead than one larger launch. Set LRF_SKIP_LARGE_KERNELS = True for ResNet/VGG.
The model registry `recommended` config sets this automatically.

---

## Output Files

| File | Generated when |
|---|---|
| `compression_report.png` | Every run |
| `pruning_report.png` | `USE_PRUNING = True` AND pruning survived its accuracy gate |
| `epsilon_landscape.png` | `FIND_OPTIMAL_EPSILON = True` AND `USE_LOW_RANK` wasn't auto-disabled |
| `epsilon_cache.json` | `FIND_OPTIMAL_EPSILON = True` (crash recovery) |
| `pruning_search_cache.json` | `FIND_OPTIMAL_PRUNING = True` (crash recovery) |
| `models/{ACTIVE_MODEL}.pth` | First run or `FORCE_RETRAIN = True` |
| `compressed_model.onnx` | `EXPORT_ONNX = True` |

---

## Installation

```bash
pip install torch torchvision torchmetrics scikit-learn tqdm \
            matplotlib numpy scipy onnx onnxruntime torch-pruning \
            --break-system-packages

# For NLP models (bert_base, distilbert, roberta_base, albert_base, distilgpt2):
pip install transformers datasets --break-system-packages
```

---