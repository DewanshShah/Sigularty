# Model Compression Toolkit - Complete Reference

sigularty is a PyTorch model compression toolkit. It takes any PyTorch model,
runs it through a configurable pipeline of compression techniques, and produces
a smaller, faster model with a full evaluation report. The pipeline is
completely model-agnostic - it works on CNNs, Transformers, NLP models, and
architectures it has never seen before.

Everything needed to understand, configure, debug, or extend the toolkit is
in this file. It assumes familiarity with PyTorch basics but explains the
compression theory behind every decision from scratch.

For installation and learning the "sigularty" package refer to it's [Pypi page](https://pypi.org/project/sigularty/)

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

**Rules:**
- All plotting lives exclusively in `visualization.py`.
- All compression logic lives exclusively in `compression.py`.
- `helper_functions.py` owns everything else (data, training, evaluation, orchestration).
- `main.py` owns only constants. It is the SOLE source of truth for every hyperparameter.
  Changing a constant in `main.py` always takes effect - no other file needs touching.

---

## Pipeline Order - Fixed, Cannot Be Changed

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

**"[gated]" means:** every one of these techniques is measured before and after it
runs (when a test set is available). If that ONE technique's own marginal
accuracy drop exceeds `ACCURACY_DROP_THRESHOLD`, its result is discarded and the
pipeline reverts to the state immediately before it ran. See
[Global Accuracy-Drop Threshold & Per-Technique Gating](#global-accuracy-drop-threshold--per-technique-gating)
below

**Why this exact order:**

BatchNorm Fusion must go first so BN parameters don't get pruned, factorized, or
clustered - that would waste budget on parameters that disappear after fusion.

Structured Pruning must run before LRF because LRF wraps layers in `nn.Sequential`
containers. Torch-Pruning traces a computational dependency graph via a real
forward pass - once layers are wrapped in Sequential, the graph trace breaks and
channel propagation fails with shape mismatch errors.

LRF and Clustering must run before Quantization. SVD decomposition and k-means
both require float32 weight values. Quantizing first converts weights to INT8 or
FP16, which breaks both algorithms.

Knowledge Distillation Fine-tuning runs after all structural changes (pruning,
LRF, clustering) AND after quantization/GPTQ are complete. It needs the final
student architecture and precision to be fixed before training against the
teacher, so it can recover accuracy lost from every prior step at once.

Quantization is always last among the structural/precision techniques (KD runs
after it specifically to recover what quantization cost).

**Two-phase optimization:**
The pipeline internally uses two phases during optimization: Phase A generates
a float32 baseline for hyperparameter search, and Phase B applies quantization
on top for final compression.

- **Phase A**: BN Fusion + Pruning + LRF + Clustering → float32 (gated internally
  by `apply_compression_pipeline`)
- **Phase B**: GPTQ → Quantization → final KD recovery, each gated individually
  in `run_compression_pipeline` (these run outside `apply_compression_pipeline`
  entirely, so they need their own gating calls)

All evaluation metrics (accuracy, size, and latency) are measured on the final
model (Phase B output). This ensures fair comparison: you're evaluating the
actual model that will be deployed, not an intermediate representation.

---

## Global Accuracy-Drop Threshold & Per-Technique Gating

`ACCURACY_DROP_THRESHOLD` (in `main.py`) is a single, global value used in
two distinct ways:

1. **Hyperparameter search selection** - `find_optimal_epsilon_smart`
   and `find_optimal_pruning_params` use it to decide which searched
   configuration to recommend.
2. **A hard per-technique gate in the actual pipeline** - after EVERY
   technique that can mutate the model runs (Structured Pruning, Low-Rank
   Factorization, Weight Clustering, GPTQ, standard Quantization, and the
   final KD recovery fine-tune), accuracy is measured immediately before and
   immediately after that one technique. If the marginal drop exceeds the
   threshold, that technique's output is discarded and the model reverts to
   its pre-technique state.

**Each technique has an INDEPENDENT budget.** If Structured Pruning alone drops
accuracy by 8pp (kept, since that's under a 10pp threshold), LRF is then judged
purely on ITS OWN marginal drop from that already-8pp-lower starting point -
not against the original baseline, and not against "10pp minus 8 already
spent." A technique cannot be penalized for budget some earlier technique used.

**What "reverts" means mechanically:** the post-technique model is discarded
(`del` + `torch.cuda.empty_cache()` on CUDA) and the model reference from
immediately before that technique ran is returned instead. This is cheap
because every technique already deep-copies internally before mutating -
reverting never requires an extra defensive copy beyond what the technique
itself already made.

**Which techniques are NEVER gated:**
- **BN Fusion** - mathematically lossless by construction. Checking it is pure
  overhead with zero chance of ever tripping.
- **Sensitivity Analysis** - read-only; it produces a map for pruning to consume
  but never mutates the model that flows downstream.

**Reporting:** a technique that gets reverted is **not** listed in the final
report's "Techniques Applied" - the report only ever describes what the
returned model actually contains. Console output explicitly states when a
technique was reverted and why (e.g.
`❌ [Weight Clustering] SKIPPED - accuracy dropped 12.3pp ... exceeding the global threshold of 10.0pp. Reverting.`).

**Final KD step gets an EXTRA check beyond the standard gate** - see
[KD's Cumulative Recovery Check & Bounded Retry](#kds-cumulative-recovery-check--bounded-retry)
below, since KD's entire job is recovery, so the standard marginal gate rarely
catches anything interesting for it specifically.

**Where this lives in code:**
- `helper_functions.py`: `gate_technique_accuracy()` - the shared gate function.
  Measures (or accepts a known) pre-accuracy, measures post-accuracy, compares
  the drop, reverts or keeps.
- `compression.py`: `apply_compression_pipeline()` calls this around Structured
  Pruning, LRF, Weight Clustering, and its own (dormant in the real product,
  see below) in-pipeline KD/GPTQ/Quantization steps. Sets `._gating_report`
  (dict of `{technique_label: bool_kept}`) and `._gating_accuracy` on the
  returned model.
- `helper_functions.py`: `run_compression_pipeline()` calls the same gate
  function around GPTQ, standard Quantization, and the final KD step, since
  those run OUTSIDE `apply_compression_pipeline` in the two-phase design above.

**Configuration constants in main.py:**
```python
ACCURACY_DROP_THRESHOLD = 10.0
# Applies to EVERY technique now: hyperparameter search selection AND the
# pipeline's per-technique gate. One number, one meaning, everywhere.
```

---

## Per-Technique Impact Reporting

**This solves a real ambiguity the gate above cannot see on its own: a
technique can be structurally a complete no-op on a given architecture and
still show a healthy-looking accuracy *improvement* in the gate log.**

Concretely: on the `custom_cnn` architecture, every `Conv_block` is a 3×3
kernel and `lrf_skip_large_kernels=True` (required for this model), so LRF's
factorization walk skips every single eligible layer - zero weights change.
Yet the gate's own log line for that run reads
`✅ [Low-Rank Factorization] kept - accuracy 71.00% → 73.40% (improved by
2.40pp, within 10.0pp threshold)`. Read on its own, that line credits LRF
with 2.40pp of recovery. It didn't do anything - the 2.40pp came entirely
from the 3 epochs of KD fine-tuning LRF's own recovery step bundles in
alongside it. The same thing happens with GPTQ whenever the only Linear
layer in a model is smaller than `min_layer_size`: `Layers quantized: 0/0`
prints once and is easy to miss, while the gate reports a perfectly
reasonable-looking `0.00pp drop, kept`.

Per-technique impact reporting exists specifically so this failure mode is
impossible to miss: for every technique the pipeline gates (Structured
Pruning, LRF, Weight Clustering, GPTQ, standard Quantization, the final KD
step), an `[Impact]` block is printed immediately after that technique's
gate decision, showing:

- **Structural summary** - what the technique's own report says actually
  changed (`"5/6 Conv2d layers pruned"`, `"0/7 eligible layers factorized"`,
  `"0/0 eligible Linear layer(s) quantized"`). Built from each technique's
  own report dict (`._pruning_report`, `._lrf_report`,
  `._clustering_report`, `._gptq_report`) since every technique's structural
  story is genuinely different - there's no single schema that fits all of
  them.
- **Algorithm vs. fine-tune split** (accuracy only, Pruning/LRF/Clustering
  only) - these three techniques bundle their own recovery fine-tune
  *inside* the same function call the gate measures around, so a plain
  before/after conflates "what the compression algorithm did to the
  weights" with "what a few epochs of KD gradient descent recovered
  afterward." An accuracy snapshot taken after the structural change but
  before that internal fine-tune starts splits the two apart:
  ```
  Algorithm  : 71.00% → 71.00%  (dropped 0.00pp - raw structural effect, before recovery fine-tune)
  Fine-tune  : 71.00% → 73.40%  (improved by 2.40pp - recovery fine-tune's own contribution)
  ```
  GPTQ and standard Quantization have no internal fine-tune loop at all
  (their only recovery happens later, in the shared post-quantization KD
  step), so their number is already a clean read with nothing to split. The
  final KD step *is* the fine-tune - there's no separate algorithm phase
  preceding it - so it's excluded from this split for the same reason; it
  reduces to marginal + cumulative only.
- **Marginal accuracy/size/latency** - this technique's own before/after,
  exactly what the gate already measures for accuracy, extended to size and
  latency too. Answers "what did THIS step change, given whatever state the
  pipeline was already in."
- **Cumulative accuracy/size/latency vs. original** - after this technique
  vs. the *absolute* original model, measured once at the very start of the
  run. Answers "how much of the pipeline's total compression so far can
  this technique take credit for" - which for a technique with marginal ≈
  1.00× is usually "none of it; cumulative is entirely inherited from
  earlier steps." **This distinction matters concretely**: if technique ♤
  compresses a 100 MB model to 50 MB (2.0×) and technique ◇ that runs next
  is a genuine no-op, ◇'s cumulative-vs-original ratio still reads 2.0× -
  that number was never claiming to be ◇'s own work, only marginal (1.00×
  for ◇) answers that question. Reading cumulative alone, under a
  technique's own heading, is the exact way to misattribute an earlier
  technique's compression to whichever technique happens to run next.
- **An explicit warning** when the caller's structural check comes back
  zero (`structural_zero=True`): *"Zero structural effect - \[technique]
  did not change a single weight this run. Any accuracy movement shown
  above is entirely the recovery fine-tune, not \[technique] itself.
  Consider disabling this technique for this architecture."* This is
  deliberately louder than a plain "marginal accuracy near zero" check,
  because - as the LRF example above shows - a technique can be
  structurally inert while accuracy *still moves*, entirely via its bundled
  fine-tune.

**Scope: pipeline-only, real numbers only, never applied during
hyperparameter search.** This only ever fires from inside
`apply_compression_pipeline()` and `run_compression_pipeline()`, using each
technique's real, full-fidelity settings (real `fine_tune_epochs`, real
`num_calibration_batches`). `optimization.py`'s search functions
(`find_optimal_epsilon_smart`, `find_optimal_pruning_params`) call the same
underlying technique functions with `test_loader` already set (for the
early-abort mechanism - see above), but never set the
`capture_pre_finetune_accuracy` flag those functions now accept, so search
trials never pay for or produce the extra accuracy snapshot this feature
needs. Search-phase numbers are explicitly reduced-fidelity by design and
would not mean the same thing here.

**Cost:** always one `get_model_size_mb()` call each for pre/post per
technique (no forward passes, negligible). Latency costs one
`measure_latency()` call each for pre/post, at **reduced fidelity** (10
iterations, 3 warmup - the same convention `optimization.py`'s search
functions already use for intermediate, non-final measurements) rather than
the 100/10 the final compression report uses, and is wrapped in try/except
so a forward-pass failure in a reporting feature can never take down the
actual compression run - it prints `N/A` and moves on. No accuracy is ever
re-measured for a *kept* technique - the gate has already measured it, and
that same number is reused for free. Only a *reverted* technique costs one
extra `measure_accuracy()` call, since the gate's own return value no
longer reflects the discarded technique's actual output in that case, and
reporting honestly on what was discarded (not what was kept instead)
requires knowing it. Additionally, Pruning/LRF/Clustering each pay one more
`measure_accuracy()` call (only when `capture_pre_finetune_accuracy=True`)
to capture the pre-fine-tune snapshot the algorithm/fine-tune split needs.

**Example (the exact LRF case above):**
```
  [Impact] Low-Rank Factorization
    Structural : 0/7 eligible layers factorized (rest: kernel>1×1, skip_large_kernels=True; output head)
    Algorithm  : 71.00% → 71.00%  (dropped 0.00pp - raw structural effect, before recovery fine-tune)
    Fine-tune  : 71.00% → 73.40%  (improved by 2.40pp - recovery fine-tune's own contribution)
    Accuracy   : 71.00% → 73.40%   marginal improved by 2.40pp   |   cumulative improved by 2.40pp vs. original
    Size       : 0.100 MB → 0.100 MB   marginal 1.00×   |   cumulative 1.00× vs. original
    Latency    : 0.640 ms → 0.635 ms   marginal 1.01×   |   cumulative 1.01× vs. original   (10 iter, reduced fidelity)
    Result     : kept
    ⚠️  Zero structural effect - Low-Rank Factorization did not change a single
        weight this run. Any accuracy movement shown above is entirely the
        recovery fine-tune, not Low-Rank Factorization itself. Consider
        disabling this technique for this architecture.
```

**Where this lives in code:**
- `helper_functions.py`: `report_technique_impact()` - the shared, technique-
  agnostic reporting function. Computes and prints marginal/cumulative
  accuracy/size/latency and the optional algorithm/fine-tune split; returns
  a dict of every computed number for future programmatic use (e.g. a
  results database).
- `compression.py`: `apply_structured_pruning()`, `apply_low_rank_
  factorization()`, `apply_adaptive_lrf()`, and `apply_weight_clustering()`
  each accept a new opt-in `capture_pre_finetune_accuracy` parameter
  (default `False`); when set, they measure accuracy once right after their
  own structural change and before their internal recovery fine-tune,
  attaching it as `._pre_finetune_accuracy` on the returned model (and
  inside their own technique-specific report dict -
  `._pruning_report['pre_finetune_accuracy']`, `._lrf_report`,
  `._clustering_report`). `apply_low_rank_factorization` and
  `apply_adaptive_lrf` also now return `._lrf_report` unconditionally
  (`{factorized_count, skipped_count, mha_skip, pre_finetune_accuracy}`) -
  previously this data was only ever printed, never returned.
  `apply_weight_clustering` contains `test_loader` parameter, used only
  for this snapshot (it has no early-abort mechanism of its own to share it
  with). `apply_gptq_quantization`'s existing `._gptq_report` has one key, `total_eligible_layers` (the denominator for `layers_quantized`
  - previously computed internally but never returned, so a `0/0` case was
  only ever visible in console output, not to any caller). `apply_
  compression_pipeline()` calls `report_technique_impact()` right after
  every gate decision.
  (`original_size_mb`, `original_latency_ms`, `input_shape`, `input_dtype`)
  so every impact report's "cumulative vs. original" numbers use the same
  true-original reference throughout.
- `helper_functions.py`: `run_compression_pipeline()` measures
  `orig_size_mb`/`orig_latency_ms` once, immediately alongside its existing
  once-only `orig_accuracy` measurement, and threads all three into
  `apply_compression_pipeline()` and its own GPTQ/Quantization/final-KD
  impact reports (which run outside `apply_compression_pipeline` in the
  two-phase design - see above).

---

## Early-Abort Threshold for Recovery Fine-Tunes

**This is a separate, earlier-firing check from the per-technique gate above.**
The accuracy-drop gate decides whether to *keep or revert* a technique **after**
its recovery fine-tune has fully finished. The early-abort threshold decides
whether to **cut a fine-tune short mid-flight** - after just epoch 1 - when
the result already looks hopeless. It exists purely to save compute: if a
config's drop is already enormous after one epoch, running the remaining
epochs almost never claws enough of it back, and the gate would revert the
technique anyway once it did finish.

Applies to Structured Pruning's and (Adaptive) LRF's recovery fine-tunes -
in **both** the real pipeline (`apply_compression_pipeline`) and their
hyperparameter searches (`find_optimal_pruning_params`,
`find_optimal_epsilon_smart`) - since all four share the same underlying
`_kd_recovery_fine_tune()` helper in `compression.py`. It does **not** apply
to Weight Clustering's or the final post-quantization KD step's fine-tunes
(`fine_tune_with_distillation`) - those are out of scope for this check.

**Mechanism:** after epoch 1 of any covered fine-tune, if the epoch-1-exactly-
0.0%-accuracy safety check (below) didn't already fire, `_kd_recovery_fine_tune`
measures accuracy on a genuinely held-out `test_loader` and compares the drop
against the **absolute original model's** accuracy (not a marginal
pre-technique value). If that drop already exceeds `FINE_TUNE_ABORT_THRESHOLD`,
the remaining epochs are skipped and the fine-tune returns whatever it has so
far - the per-technique gate (or the search's own scoring) then decides that
result's fate normally.

**`FINE_TUNE_ABORT_THRESHOLD` is a direct percentage-point value, not a
multiplier.** `FINE_TUNE_ABORT_THRESHOLD = 40.0` means exactly "abort after
epoch 1 if the drop already exceeds 40pp" - full stop, independent of
whatever `ACCURACY_DROP_THRESHOLD` happens to be set to.

**Cost:** exactly one extra `measure_accuracy()` pass over `test_loader`,
only after epoch 1, only when this check is active - on a few-hundred-sample
test set that's a couple of seconds, negligible next to a real training epoch.

**A real limitation worth knowing:** this check only has something to abort
*out of* when `fine_tune_epochs > 1`. With `EPSILON_SEARCH_FT_EPOCHS = 1`
(the default), there's no "remaining epoch" left after epoch 1 finishes, so
this threshold has essentially no effect on the epsilon search specifically -
it matters far more for `PRUNING_SEARCH_FT_EPOCHS = 3` and for the main
pipeline's own `PRUNING_FINE_TUNE_EPOCHS` / `LRF_FINE_TUNE_EPOCHS`, wherever
more than one epoch is actually configured.

```
⏭️  [Pruning Fine-tune] EARLY ABORT: after epoch 1, held-out test accuracy is
61.20% vs. the original model's 92.66% - a 31.46pp drop, exceeding the 30.0pp
early-abort threshold. Recovering that much ground in the remaining 2
epoch(s) is judged implausible, so skipping them rather than spending compute
on a result the accuracy-drop gate would very likely revert anyway.
```
*(Example above predates the 30.0 → 40.0 change - a 31.46pp drop would now
survive to keep training, not abort.)*

**Configuration constants in main.py:**
```python
FINE_TUNE_ABORT_THRESHOLD = 40.0
# Direct pp value. None disables this entirely (every fine-tune always runs
# to completion). Use a very large number (e.g. 1000)
# instead of None if you want to keep the CLI flag's float type happy while
# effectively never triggering it.
```

**CLI flag:** `--fine-tune-abort-threshold 25` (any float; overrides the
main.py constant for a single run).

---

## LRF + nn.MultiheadAttention Safety

**ViT-B/16 (and any architecture built on `nn.MultiheadAttention`) cannot
safely run through Low-Rank Factorization.** This used to fail silently and
expensively - every epsilon trial would report a confidently-wrong **0.0%
accuracy**, and the search would conclude "LRF makes this model worse" after
burning the entire trial budget on a model that was never actually evaluated.

**The mechanical reason:** `_decompose_linear_layer` replaces a target
`nn.Linear` with `nn.Sequential(layer1, layer2)`. This works fine for ordinary
`nn.Linear` usage, where the parent module calls `child(x)` (i.e. the
Sequential's own `forward()`). But `nn.MultiheadAttention.forward()` does NOT
do this for its `out_proj` sub-layer - it calls
`F.multi_head_attention_forward(..., self.out_proj.weight, self.out_proj.bias, ...)`,
reaching directly into `.weight` / `.bias` as raw tensors. `nn.Sequential` has
neither attribute, so every forward pass after factorization raised
`AttributeError: 'Sequential' object has no attribute 'weight'`.

**Why this used to produce 0.0% instead of crashing visibly:** `measure_accuracy`
had a per-batch exception handler intended to tolerate occasional shape
mismatches, written broadly enough that it also caught and silently skipped
this `AttributeError` on **every single batch**. With zero batches counted,
the function fell through to `return 0.0 if total == 0 else ...` - a fake,
confident-looking accuracy number indistinguishable from "the model is
genuinely bad."

**The fix, in two layers:**

1. **`measure_accuracy` now raises** instead of silently returning `0.0` when
   every batch failed (`total == 0`). The error message contains the
   substring `"No samples were evaluated"`, which `optimization.py`'s search
   loops specifically detect to trigger the consecutive-failure abort protocol
   (see [Structural-Failure Abort Protocol](#structural-failure-abort-protocol-search-loops)
   below) instead of grinding through more guaranteed-broken trials.
   Per-batch tolerance for occasional, genuinely isolated shape issues is
   unchanged - what changed is what happens when literally everything fails.

2. **`apply_low_rank_factorization()` and `apply_adaptive_lrf()` proactively
   detect `nn.MultiheadAttention` anywhere in the model** and return it
   **completely unchanged** (a deepcopy, but zero factorization applied
   anywhere) rather than attempting partial factorization elsewhere and
   crashing on the attention layer. **`run_compression_pipeline()` ALSO checks
   for this before the pipeline even starts**, and if found, force-disables
   both `USE_LOW_RANK` and `FIND_OPTIMAL_EPSILON` for the run - this is the
   layer that actually saves you the wasted compute, since it skips the
   epsilon search's 15+ trials entirely instead of running them only to have
   each one no-op. `compression.py`'s own check is a defensive backstop for
   anyone calling these functions directly outside the pipeline.

This is checked via `model_has_multihead_attention()` in `helper_functions.py`
- a simple `isinstance(m, nn.MultiheadAttention)` walk. It does **not** affect
Swin-T (`ShiftedWindowAttention` uses plain `qkv`/`proj` `nn.Linear` layers,
not `nn.MultiheadAttention`) or any of the HuggingFace NLP models (BERT,
RoBERTa, DistilBERT, ALBERT, DistilGPT-2 - all use separate Q/K/V `nn.Linear`
projections called through ordinary `forward()`). Their existing
`use_low_rank: False` registry recommendations are quality-based, not
crash-prevention - this check only fires on the exact pattern that actually
breaks.

**Console output when this triggers:**
```
⚠️  [LRF Safety] Detected 12 nn.MultiheadAttention module(s) in 'vit_b_16'.
LRF cannot safely factorize out_proj (...). Disabling use_low_rank and
find_optimal_epsilon for this run.
```

---

## LRF Output-Head Protection

The final `nn.Linear` in the model (the classification head) is now always
skipped by LRF, regardless of `LRF_EPSILON` / `LRF_MIN_LAYER_SIZE` /
`LRF_MIN_RANK`, matching the protection Structured Pruning already had via
`ignored_layers`. This layer is typically well under 0.1% of total parameters
(e.g. `Linear(768, 100)` on ViT-B/16 is 76,800 params out of 86.6M) -
factorizing it saves essentially nothing but every feature the network has
learned funnels through this one decision boundary, making it disproportionately
sensitive to rank reduction. `compute_adaptive_epsilons()` also skips analyzing
this layer, since `apply_adaptive_lrf()` will protect it regardless.

---

## LRF Recovery Fine-Tuning

Low-Rank Factorization previously had **no** recovery mechanism at all - every
other major technique (Pruning, Clustering) could fine-tune afterward, but LRF
just applied the SVD truncation and stopped. Under the global gating
system, this meant LRF would get reverted far more often than necessary,
since it had no way to claw back accuracy before the gate checked it.

**LRF now supports an optional recovery fine-tune, always knowledge
distillation against the original (pre-factorization) model - never plain
cross-entropy**, matching the "every major fine-tune in this pipeline uses KD"
principle that Pruning and Clustering already followed.

```python
LRF_FINE_TUNE_EPOCHS = 3        # 0 = no fine-tune
LRF_FINE_TUNE_LR     = 0.00001
```

When `LRF_FINE_TUNE_EPOCHS > 0`, the factorized model is fine-tuned via the
shared `_kd_recovery_fine_tune()` helper in `compression.py` (the same one
Structured Pruning now uses - see below), against the original model as a
frozen teacher, reusing `KD_TEMPERATURE` / `KD_ALPHA` (no separate
LRF-specific KD hyperparameters were introduced). Setting either to `0`
epochs preserves the no-fine-tune behaviour exactly, for backward
compatibility with direct/library use of `apply_low_rank_factorization()`.

---

## Structured Pruning's Recovery Fine-Tune Is Now Also KD-Based

Structured Pruning's post-prune fine-tune previously used plain cross-entropy
(`_fine_tune_after_pruning`, now removed). It has been replaced with the same
shared `_kd_recovery_fine_tune()` helper LRF uses above, teaching against
`model` (the function's own pre-pruning parameter - pruning is always the
first technique in the pipeline, so this genuinely is the original,
uncompressed reference). `PRUNING_FINE_TUNE_EPOCHS` / `PRUNING_FINE_TUNE_LR`
control the same thing they always did (epochs and learning rate); the loss
formula itself now matches KD's (`α·CE + (1-α)·T²·KL(teacher‖student)`) using
`KD_TEMPERATURE` / `KD_ALPHA` rather than pure CE.

Weight Clustering's fine-tune already used KD in practice - the real pipeline
always supplies `kd_teacher=model` to `apply_weight_clustering()`, so its
plain cross-entropy fallback path (`_fine_tune_after_clustering`) only ever
runs for direct/library callers who explicitly omit a teacher.

---

## Major-Fine-Tune Epoch-Zero Safety Check (Everywhere)

Every fine-tuning loop in this toolkit - `_kd_recovery_fine_tune` (Pruning,
LRF), `fine_tune_with_distillation` (Clustering, the in-pipeline KD step, the
final post-quantization KD step), and the plain-CE fallback
`_fine_tune_after_clustering` - now shares the same safety check:

**If epoch 1 reports EXACTLY 0.0% accuracy, a major warning is printed and ALL
remaining epochs are skipped.** This is almost always a structural break
(shape mismatch, dead/NaN gradients, wrong `num_classes`) rather than ordinary
slow convergence - continuing to train for the remaining epochs would just
waste compute on a fine-tune that mathematically cannot recover (e.g. if every
gradient is zero or NaN, more epochs change nothing). The model is still
returned (broken or not) - the responsibility for actually discarding it
belongs to the per-technique accuracy gate described above, which will see the
terrible post-fine-tune accuracy and revert automatically.

```
🚨 [Pruning Fine-tune] MAJOR WARNING: epoch 1 produced EXACTLY 0.0% accuracy.
This almost always means a structural break (shape mismatch, dead/NaN
gradients, wrong num_classes) rather than slow convergence. Skipping
remaining 2 epoch(s) to avoid wasting compute - the accuracy-drop gate will
revert this technique.
```

The "training accuracy" used for this check is the same metric these loops
already compute and print per-epoch (via `torchmetrics.Accuracy` on the
training/calibration batches being fine-tuned on) - not a separate
measurement, and not necessarily the same as held-out test accuracy.

`fine_tune_with_distillation` (the generic KD function, used by Clustering and
both deferred/final KD steps) previously only tracked accuracy when
`test_loader` was supplied - meaning the in-pipeline KD step (which doesn't
pass `test_loader`) had **no** per-epoch accuracy signal at all. It now always
tracks training accuracy regardless of `test_loader`, and additionally tracks
test accuracy when available, attaching both to the returned model as
`._kd_history = [{epoch, loss, train_acc, test_acc}, ...]` - this history is
what powers the final KD step's recovery-quality retry check below.

---

## measure_accuracy(): Structural Failure vs. Genuine Zero

`measure_accuracy()` in `helper_functions.py` now distinguishes two
previously-conflated outcomes:

**1. Structural failure - every batch raised an exception.** Previously this
silently returned a fake `0.0%`. It now **raises** `RuntimeError` with a
message containing `"No samples were evaluated"`. This propagates through
`optimization.py`'s search evaluation functions (`_evaluate_epsilon`,
`_evaluate_pruning_config`), which re-raise it specifically (rather than
swallowing it into a generic "trial failed, score 0" result), so the search
loop's structural-failure counter can see it and trigger the abort protocol
below. Individual-batch tolerance for ordinary shape quirks is unchanged -
this only changes what happens when ALL batches fail.

**2. Genuine zero - at least one batch succeeded, but literally nothing was
predicted correctly.** This is a real, valid (if bad) result, so it is still
**returned normally** - but a major warning is printed, since 0.0% on a real
evaluation almost always indicates something is badly wrong even though it
technically ran:

```
🚨 MAJOR WARNING: measure_accuracy returned EXACTLY 0.0% (500 samples
evaluated, 0 correct). This is usually a real, severe failure (wrong
num_classes, NaN/dead weights, mismatched output head) rather than ordinary
poor performance - investigate before trusting downstream results.
```

---

## Structural-Failure Abort Protocol (Search Loops)

Both `find_optimal_epsilon_smart()` and `find_optimal_pruning_params()` track
**consecutive** structural failures (the "No samples were evaluated" case
above) across their **entire run** - one shared counter spanning every phase
(anchor sampling, ternary refinement, grid search, iterative-steps comparison
for pruning), not reset between phases. Any successful trial resets the
counter to zero. **On the second consecutive structural failure, the search
aborts immediately** - no further trials are attempted in any remaining
phase - and falls straight through to final selection using whatever
succeeded before the abort.

This is intentionally conservative: one isolated failure is tolerated and the
search just moves to the next trial, but two in a row is treated as strong
evidence the whole architecture is fundamentally incompatible with this
technique (the `nn.MultiheadAttention` case above is exactly this, though it's
now caught earlier by the dedicated check and never reaches this fallback in
practice). Continuing to grind through a full trial budget when every
remaining trial is virtually certain to fail the same way wastes exactly the
compute this protocol exists to save.

Both search functions' final-selection logic (`find_optimal_epsilon_smart`'s
Phase 3, `find_optimal_pruning_params`'s rewritten Phase 4 - see below)
operate by pooling **everything cached so far**, so an abort never discards
genuinely good results that succeeded before the failures started. If
**zero** trials ever succeeded, both functions return `(None, {'warning': ...}, history)`
rather than crashing on an empty result set.

**Mechanically:** the shared kernel `_ternary_search()` now accepts a
`(None, None)` sentinel return from its `score_fn` callback to mean "stop
searching, the caller has decided to abort" - this keeps the abort logic
entirely inside each search function's own evaluation wrapper (where the
counter lives) rather than requiring `_ternary_search()` itself to understand
exceptions or counters.

---

## Pruning Search - Phase 4 Rewritten (Bug Fix)

This fixes the exact bug pattern that originally prompted this round of
work: the pruning hyperparameter search could select a catastrophically
destructive configuration (e.g. 96% global pruning, ~2% final accuracy) over
a clearly-better, low-risk configuration (e.g. 20% global pruning, ~68%
final accuracy, only an 8.5pp drop) that was sitting in the search's own cache
the entire time.

**Two compounding root causes, both fixed:**

1. **Cache-key mismatch in anchor accuracy-drop checking.** The function that
   was supposed to gate anchor selection by accuracy drop
   (`_anchor_acc_drop`) looked up `cache.get(str(round(ratio, 4)), {})` - but
   `_evaluate_pruning_config()`'s actual cache key is a five-field compound
   string (`"{ratio}|{max_ratio}|{epochs}|{lr}|{steps}"`). The simple
   single-field lookup **always missed**, silently falling back to a default
   of "0.0pp drop" for every single anchor regardless of its real accuracy.
   This made the strict-tier accuracy gate a no-op - every anchor "passed,"
   so anchor selection collapsed to pure score-maximization. Now fixed by
   reading directly from the search's own in-memory results
   (`anchor_results`, built from the exact dicts just computed in Phase 1)
   instead of re-deriving a cache key.

2. **No accuracy gating anywhere past Phase 1.** Ratio ternary refinement
   (Phase 1b), the max-ratio grid (Phase 2), and the iterative-steps
   comparison (Phase 3) were all pure `max(..., key=score)` with zero
   awareness of accuracy drop - so even with bug #1 fixed, these phases would
   still climb toward whatever region had the highest raw CQI score, which
   (especially with `CQI_W_SIZE > CQI_W_ACCURACY`, as in this toolkit's
   default `CQI_W_SIZE = 2.0`) is often the most heavily-pruned, most
   accuracy-destroyed region - a squared size-ratio factor can outweigh a
   linear accuracy-ratio factor even when the accuracy outcome is a near-total
   loss. Every phase now uses a shared `_select_best_by_tier()` helper:
   prefer candidates within `accuracy_drop_threshold` (ranked by score), fall
   back to `accuracy_drop_threshold + 10pp` if nothing qualifies, fall back
   to pure score-max only as a last resort.

**Phase 4 itself was rewritten from "validate one funnel winner, binary
accept/reject" to "pool every cached evaluation from the entire search,
filter to candidates within `accuracy_drop_threshold`, pick the highest score
among survivors"** - mirroring `find_optimal_epsilon_smart`'s Phase 3 exactly.
This means a low-ratio configuration sitting in the cache with a small,
acceptable accuracy drop can now win outright over a high-ratio
configuration with a much higher raw score but a catastrophic accuracy drop
- previously, only the single config that emerged from the (also-broken)
funnel above could ever be considered, with no path back to a better
candidate that was computed but never selected as the "current best" along
the way.

**Incidental fix in the same area:** the "abort search early if every anchor
pruned 0 layers" check (intended for transformer architectures with no
prunable Conv2d groups) compared a field, `layers_pruned`, that
`_evaluate_pruning_config()`'s result dict never actually populated - so the
comparison (`-1 == 0`) was always `False` for any real result, making this
early-exit dead code in practice (it would only fire, incorrectly, if zero
anchors were cached at all, for any reason). `_evaluate_pruning_config()` now
populates a real `layers_pruned_count` field, and the check reads that
correctly.

---

## KD's Cumulative Recovery Check & Bounded Retry

The final post-quantization KD step is gated like every other technique (see
[Global Accuracy-Drop Threshold](#global-accuracy-drop-threshold--per-technique-gating)
above) - if KD itself somehow makes accuracy worse, it reverts to the
pre-KD state. But KD's entire purpose is *recovery*, so that marginal check
rarely catches anything meaningful for it specifically (worst case, it does
nothing useful; it essentially never makes things actively worse).

**A second, separate check exists specifically for KD:** after KD finishes
(and survives its own marginal gate), the **cumulative** drop is checked -
original baseline accuracy vs. the model's accuracy now, after every
compression technique AND the recovery fine-tune. If that cumulative drop
still exceeds `ACCURACY_DROP_THRESHOLD`, the per-epoch test-accuracy trend
(from `._kd_history`, tracked by `fine_tune_with_distillation` - see above)
is inspected for exactly one bounded retry:

- **Stagnant** (average epoch-to-epoch test-accuracy change is within
  ±0.5pp): the trend suggests the learning rate is too high to make further
  progress (oscillating around a point) or too low for any signal to move
  the needle in either direction within a few epochs - halve `KD_LR` and
  retry once, restarting fresh from the pre-KD checkpoint (not continuing
  the already-completed trajectory).
- **Small but consistent improvement** (every epoch-to-epoch delta is
  positive, averaging under 2pp/epoch): the trend is real but slow - raise
  `KD_LR` by 50% and retry once, same fresh-restart approach.
- **Anything else** (erratic, net-negative, or already a large jump):
  accepted as-is, no retry - the trend doesn't match either pattern this
  heuristic is designed to correct.

**The retry's result is accepted unconditionally, regardless of outcome -
there is no second retry, and no gate re-applied to the retry's result.**
If the retry still isn't acceptable, the printed guidance is to re-run with
less aggressive compression settings rather than looping indefinitely inside
a single pipeline run.

```
[KD Recovery] Cumulative accuracy drop 12.40pp still exceeds the 10.0pp
threshold after KD. Trend is stagnant (avg epoch-to-epoch change +0.12pp,
|Δ|<0.5pp) - retrying ONCE with lr 1.00e-05 → 5.00e-06, restarting fresh from
the pre-KD checkpoint. This result is accepted unconditionally - no further
retries.
```

This entire mechanism is intentionally NOT a general-purpose hyperparameter
search (it is not what `find_optimal_epsilon_smart`/`find_optimal_pruning_params`
are) - it's a narrow, bounded, single-shot correction for the two most common
"KD didn't quite get there" failure shapes, on the principle that a one-line
LR adjustment is worth trying automatically before asking you to reconfigure
and re-run the whole pipeline.

---

## Model and Dataset

**EfficientNet-B0** on **Oxford Flowers-102** is the default. The model registry
includes 20 models total - see `ACTIVE_MODEL` in main.py.

**Why EfficientNet-B0 as the default:**
- 5.3M parameters: small enough that compression effects are sharp and visible.
  ResNet-50 at 25M has so much redundancy that you can prune 40% and barely notice.
  EfficientNet-B0 is already lean, so every compression technique has to earn its
  compression ratio.
- MBConv blocks (mobile inverted bottleneck convolutions) use depthwise-separable
  convolutions. These respond differently to LRF than standard convolutions because
  the depthwise layers are already rank-1 (groups == in_channels). The toolkit
  automatically skips them and factorizes only the pointwise (1×1) convolutions.
- Compound scaling (width × depth × resolution co-optimized by NAS) means the
  model is already at a Pareto-optimal point on the accuracy-FLOPs tradeoff.
  Compressing it further requires techniques that actually understand structure,
  not just uniform shrinking.

**Why Flowers-102:**
- 102 fine-grained flower categories that are visually very similar (different
  species of roses, tulips, etc.). A model that compresses poorly will immediately
  lose the ability to distinguish similar classes.
- Forces fine-tuning from ImageNet weights - tests the full load-or-train path.
  If you used ImageNet directly, you could skip the training step.
- Small dataset (2040 training images) - fits in RAM, fast to iterate on.
- Test set has 6149 images with 500 used by default for speed.

---

## Compression Techniques

### BatchNorm Fusion

**What it does:**
Eliminates BatchNorm layers by mathematically folding them into the preceding
Conv2d or Linear layer. The result is a single layer that computes the same thing
as Conv→BN in one operation. It has one limitation - It assumes the fact that the model already
has Batch Norm layers added and trained that is to be fused.

**The math:**
Conv2d output: `y = W * x + b`
BatchNorm output: `z = γ * (y - μ) / sqrt(σ² + ε) + β`

Since both are linear in x, they compose into:
```
W' = W * (γ / sqrt(σ² + ε))     # one scale factor per output channel
b' = (b - μ) * (γ / sqrt(σ² + ε)) + β
```

The fused Conv2d has these weights and bias. The BatchNorm layer is replaced
with `nn.Identity()` (a no-op). The computation is now one kernel launch instead
of two.

**Why it's always safe (and never gated):**
This is a mathematical identity - the output is bit-for-bit identical before and
after fusion (up to floating point rounding). Zero accuracy change. Zero need for
fine-tuning. Zero point in measuring accuracy before/after - there's nothing to
revert to. This is the first optimization every production inference engine
(TensorRT, ONNX Runtime, TorchScript) performs automatically.

**When it helps:**
- Reduces kernel launches (one fewer GPU operation per Conv-BN pair)
- Reduces memory traffic (BN parameters no longer loaded from VRAM)
- Shrinks model size (BN parameters eliminated: γ, β, running_mean, running_var)
- On EfficientNet-B0 with 17 Conv-BN pairs: ~2% latency improvement, small size reduction

**Requirements:**
- BatchNorm must be in eval mode (running statistics populated). The function
  calls `model.eval()` internally before fusing.
- Conv2d output channels must match BatchNorm num_features (always true in valid models).

**Configuration constants in main.py:**
```python
USE_BN_FUSION = True   # Default True. No reason to ever disable this.
```
There are no other hyperparameters. Fusion is always correct or it doesn't apply.

---

### Structured Pruning

**What it does:**
Permanently removes entire output channels (filters) from Conv2d layers. This
reduces the number of filters from, say, 160 to 120, which makes every weight
tensor and activation tensor smaller. Unlike unstructured pruning (zeroing
individual weights), structured pruning produces smaller dense tensors that run
faster on any hardware without sparse matrix support.

**Importance metric - activation statistics:**
```
importance(filter_i) = mean over calibration batches of:
    mean(|activation_i(x)|) over spatial positions
```
Filters that consistently produce small activations on the calibration data are
contributing little to the representation. They get removed first. (See
`_ActivationImportance` in `compression.py` for why this beats L1/L2 weight
magnitude for unknown architectures - magnitude is a property of the
learned parameter, not of how much that parameter actually matters on real data.)

**Why Torch-Pruning instead of custom code:**
Every new architecture required new heuristics, and skip connections, SE blocks, and
multi-path branches each caused new shape mismatch crashes. Torch-Pruning traces
a proper computational dependency graph from a real forward pass, then propagates
channel removals atomically through all coupled layers. Skip connections,
SqueezeExcitation blocks, and arbitrary architectures all work without
any architecture-specific code.

**Why global pruning instead of per-layer uniform:**
Uniform pruning (e.g., remove 30% from every layer) hits critical early layers
equally hard as redundant later layers. The first few convolutions in a CNN learn
low-level edge detectors - removing 30% of them destroys the basis for all
downstream processing. Late layers often contain a lot of redundancy. Global
pruning sets one sparsity target and lets importance scores distribute the budget
automatically. Redundant layers get heavily pruned; critical layers are left
mostly intact.

**The max_pruning_ratio / residual-group protection (critical):**
Without protection, global pruning can send an individual layer to 1-5
channels even when the global ratio is only 0.2-0.3. This happens because
low-activation layers attract the entire budget under global optimization.
Concretely, this toolkit found - on a real trained ResNet-50 - that the
`conv3`/`downsample` group at the end of `layer4` (the output-facing convs
whose width IS the residual stream for that entire stage) lost 92% of its
channels at just a 20% global target, while every other layer lost 2-25%.

**Two things had to be true for the fix to actually work, both confirmed
empirically (not just assumed):**
1. Torch-Pruning's own `max_pruning_ratio` KWARG to `MetaPruner` does **not**
   reliably hold a specific vulnerable group under a ceiling by itself - a
   deliberately-weakened test group still got pruned all the way to the
   overall global target, not held near the requested cap.
2. The mechanism that DOES reliably work is `pruning_ratio_dict`: explicitly
   setting a target ratio per module so Torch-Pruning's global solver treats
   that ratio as fixed for those modules and redistributes the rest of the
   budget elsewhere - confirmed to hold a deliberately-weakened group to
   exactly its requested ratio while the network still reached a healthy
   overall reduction.

Which modules get an entry in `pruning_ratio_dict` is now determined by
**architecture-agnostic graph analysis**, not name matching - see
[Architecture-Agnostic Residual Group Detection](#architecture-agnostic-residual-group-detection)
below for the full mechanism and the empirical verification behind it.

SE (SqueezeExcitation) attention layers are set to ratio=0 always - they cannot
be pruned at all. SE blocks implement cross-channel attention that tells the model
which feature maps to emphasize. Pruning their fc1/fc2 layers breaks the
attention mechanism completely.

**Model type safety clamping:**

| model_type   | max allowed ratio | Reason |
|---|---|---|
| `'classifier'` | 0.5 | Standard classifiers tolerate aggressive pruning |
| `'embedding'`  | 0.2 | Embedding geometry breaks silently in downstream tasks |
| `'generative'` | 0.1 | Diffusion/GAN models extremely sensitive |
| `'unknown'`    | 0.2 | Conservative default for unfamiliar models |

The embedding model cap (0.2) exists because embedding models feed downstream
tasks like cosine similarity search or retrieval. Removing filters changes the
embedding space geometry. The pruned model's own accuracy can look fine while
downstream cosine similarity collapses completely - a silent, hard-to-detect failure.
Use `'unknown'` for any model you haven't verified.

**Recovery fine-tune - now ALWAYS knowledge distillation:**
Previously plain cross-entropy. Now uses the shared `_kd_recovery_fine_tune()`
helper, teaching against the original (pre-pruning) model - same KD loss
formula as every other major fine-tune in this toolkit (see
[Structured Pruning's Recovery Fine-Tune Is Now Also KD-Based](#structured-prunings-recovery-fine-tune-is-now-also-kd-based)
above). Includes the epoch-1-exactly-0.0% safety check.

**Behavioral probe:**
After every pruning run, the toolkit compares original vs pruned model outputs
on 10 calibration batches and computes KL divergence (for classifiers) or cosine
similarity (for embeddings/other). This catches silent degradation that accuracy
metrics miss - a model can still get 85% top-1 accuracy while outputting
completely wrong confidence distributions.

KL divergence thresholds:
- < 0.01: Negligible. Deploy as-is.
- < 0.05: Acceptable. Run evaluation to confirm.
- < 0.15: Moderate. Reduce pruning_ratio or add more fine-tuning.
- ≥ 0.15: High. Do not deploy. Reduce ratio or increase fine-tune epochs.

**Pipeline-level accuracy gate:** beyond the behavioral probe (which only
*reports* severity), pruning's actual result is now measured before/after and
reverted automatically if its own marginal accuracy drop exceeds
`ACCURACY_DROP_THRESHOLD` - see
[Global Accuracy-Drop Threshold](#global-accuracy-drop-threshold--per-technique-gating).

**Configuration constants in main.py:**
```python
USE_PRUNING               = True
PRUNING_RATIO             = 0.3   # Global fraction of channels to remove (0.0–1.0)
PRUNING_MODEL_TYPE        = 'classifier'
PRUNING_FINE_TUNE_EPOCHS  = 3
PRUNING_FINE_TUNE_LR      = 0.0001
PRUNING_CAL_BATCHES       = 50
PRUNING_ITERATIVE_STEPS   = 1
PRUNING_ROUND_TO          = None
PRUNING_ISOMORPHIC        = False
PRUNING_MAX_RATIO         = 0.5
PRUNING_RESIDUAL_MAX_RATIO = None  # None = fall back to PRUNING_MAX_RATIO above
PRUNING_REPORT_PATH       = 'pruning_report.png'
```

`PRUNING_RATIO`: The target fraction of total channels to remove globally. 0.3
means roughly 30% of channels go away. The actual per-layer ratios vary - some
layers lose 50%, critical ones lose 5%. At 0.3 on EfficientNet-B0, you get
~20-30% fewer parameters and ~10-20% faster inference.

`PRUNING_MODEL_TYPE`: Controls the safety cap. Use `'classifier'` for any image
classifier or text classifier. Use `'embedding'` for models that produce feature
vectors for similarity search. Use `'generative'` for anything generating images
or text. When in doubt, use `'unknown'`.

`PRUNING_FINE_TUNE_EPOCHS`: Short KD-based training after pruning to recover
lost accuracy. 3 epochs typically recovers 3-7% accuracy. Use 0 to skip
(faster, but accuracy stays degraded). Never use a high LR here - you're nudging
weights to adapt to the removed channels, not re-learning from scratch.

`PRUNING_FINE_TUNE_LR`: Keep this well below the original training LR (0.001).
1e-4 is the standard. If accuracy is still low after fine-tuning, halve this and
double the epochs before raising it.

`PRUNING_CAL_BATCHES`: Number of calibration batches for activation-statistics
importance collection. More batches = more reliable importance rankings. 50
batches of 32 images = 1600 samples. For very noisy data, increase to 200.

`PRUNING_ITERATIVE_STEPS`: Whether to prune in one shot (1) or multiple incremental
rounds (>1). `steps=2` with `ratio=0.3` removes ~15% per round then re-evaluates
importance. More steps = more stable accuracy but slower. Single-shot (1) is fine
for most cases. Use 2-3 steps when you see very high behavioral severity scores.

`PRUNING_ROUND_TO`: Round pruned channel counts to multiples of this number. Set
to 8 or 16 for GPU Tensor Core alignment (channels that are multiples of 8 or 16
enable faster matrix multiply hardware paths). Set to None for maximum compression.
The accuracy impact of rounding is negligible; the latency impact can be 5-15%.

`PRUNING_ISOMORPHIC`: Force the same channel mask across all layers in a dependency
group. Normally False - each layer gets its own optimal ratio. Set to True for
parallel-branch architectures (ResNeXt groups, multi-head attention) where the
same channels must be removed from all parallel paths.

`PRUNING_MAX_RATIO`: Hard cap per layer/group, enforced via `pruning_ratio_dict`
(see [Architecture-Agnostic Residual Group Detection](#architecture-agnostic-residual-group-detection)
below for why this - not Torch-Pruning's own `max_pruning_ratio` kwarg - is
what actually holds). No single layer will ever be pruned more than this
fraction regardless of global budget. Default 0.5 means at most 50% of any
one layer is removed. Decrease this (e.g. 0.2) if you see individual layers
collapsing to single-digit channel counts. Also the fallback value for
`PRUNING_RESIDUAL_MAX_RATIO` below when that constant is `None`.

`PRUNING_RESIDUAL_MAX_RATIO`: Ceiling specifically for auto-detected residual/
skip-connection-coupled groups - the layers whose channel count IS the
residual stream for an entire network stage, so collapsing them damages every
downstream block in that stage, not just one layer's worth of capacity.
`None` (default) falls back to `PRUNING_MAX_RATIO` above. Set an explicit
number here only if you want residual-critical layers held to a *different*
ceiling than everything else. Applies to both the pruning search and the
final pipeline run.

---

### Architecture-Agnostic Residual Group Detection

**This replaces the previous `'block.3'` string-match, which only worked for
EfficientNet/MobileNet's specific naming convention and silently protected
nothing on every other architecture.**

It uses Torch-Pruning's own dependency
graph instead of any module name.** `_find_residual_coupled_groups()` in
`compression.py`:

1. Builds a `tp.DependencyGraph` on the model being pruned (reusing the same
   example input `apply_structured_pruning` already derives from the
   dataloader).
2. Walks every coupled group Torch-Pruning discovers via `get_all_groups()`.
3. For each group, checks whether it contains a genuine elementwise
   **addition** node - distinguished from ReLU and other activations (which
   Torch-Pruning also classifies under the same `OPTYPE.ELEMENTWISE`
   category - checking for `ELEMENTWISE` alone would false-positive on every
   ordinary `Conv→BN→ReLU→Conv` chain in the network) by inspecting each
   node's actual PyTorch `grad_fn` class name: a real residual/skip addition
   shows up as an `AddBackward`-class autograd function, an activation shows
   up as e.g. `ReluBackward0`.
4. Every module in a flagged group gets a `pruning_ratio_dict` entry at
   `PRUNING_RESIDUAL_MAX_RATIO` (falling back to `PRUNING_MAX_RATIO`).

**Verified empirically, not just inferred from the library's docs:**

| Architecture | Result |
|---|---|
| ResNet-50 | Finds exactly the 4 stage-boundary groups (`conv3` + `downsample.0` in every Bottleneck, correctly coupled across each stage) |
| EfficientNet-B0 | Finds a **strict superset** of what `'block.3'` covered - the same 14 project-conv module names, plus 14 more (the expand-conv side of the coupling) it was silently missing |
| VGG-16 | Finds **zero** groups - the correct negative control, since VGG has no skip connections at all |

Also confirmed directly (not assumed): Torch-Pruning's own `max_pruning_ratio`
kwarg to `MetaPruner`, even when correctly wired through (see the fix to the
pre-clamp guard bug below), does **not** reliably hold a deliberately-weakened
test group under its requested ceiling - it let the group get pruned almost
exactly as far as the *overall* global target, with no differential
protection. `pruning_ratio_dict`, by contrast, held the same test group to
*exactly* its requested ratio while the rest of the network absorbed the
slack and the model still reached a healthy overall reduction. This is why
the fix here is a graph-based `pruning_ratio_dict` builder, not simply
"pass `max_pruning_ratio` through correctly and stop there."

**A second, independent bug in the same area, now also fixed:** a guard
clause -
```python
if pruning_ratio > max_pruning_ratio:
    pruning_ratio = max_pruning_ratio
```
- used to silently collapse the requested **global** ratio target down to
the cap value whenever the cap was set lower than the target. This meant a
hyperparameter search sweeping `max_pruning_ratio` from 0.2 to 1.0 was never
actually testing "same overall compression, capped per layer" - it was
testing "a smaller overall compression target," since the global ratio
itself got overwritten before Torch-Pruning ever saw it. Combined with a
second bug - the `max_pruning_ratio` value handed to `MetaPruner` itself was
hardcoded to the literal `1.0` regardless of what was requested - neither
half of the intended "global target stays put, one group gets protected"
behavior was ever actually happening. Both are fixed: the global ratio is
now always honored as requested; `pruning_ratio_dict` (built from the
detection above) is what protects specific groups.

---

### Low-Rank Factorization

**What it does:**
Approximates weight matrices as a product of two smaller matrices using truncated
SVD. A `Linear(in=512, out=1024)` layer becomes two layers: `Linear(512, rank,
bias=False)` + `Linear(rank, 1024, bias=True)`. Parameter count drops from
512×1024=524,288 to 512×rank + rank×1024, which for rank=256 is 393,216 - a 25%
reduction on just this layer.

**The math:**
Every weight matrix W can be exactly decomposed as:
```
W = U @ diag(S) @ V^T       # Full SVD
```
where S contains singular values in descending order. The key insight: the
singular values decay rapidly for well-trained layers. If you keep only the top-r
singular values and zero the rest, the approximation error is:
```
||W - W_r||_F² = sum(S[r:]²)
```
The approximation captures `sum(S[:r]²) / sum(S²)` of the total representational
power (energy). For most layers, the top 50% of singular vectors capture 95-99%
of the energy.

Splitting the factored matrices into two layers:
```
U_r = U[:, :rank] @ diag(S[:rank])    # [out_features, rank]
V_r = V[:, :rank].T                    # [rank, in_features]
Layer1 = Linear(in, rank, bias=False): weight = V_r
Layer2 = Linear(rank, out, bias=True):  weight = U_r
```

For Conv2d, the kernel `[out_ch, in_ch, kH, kW]` is reshaped to `[out_ch, in_ch*kH*kW]`,
decomposed, then the factors are folded back into kernel shapes:
```
Conv1 = Conv2d(in_ch, rank, (kH, kW), stride, padding, bias=False)
Conv2 = Conv2d(rank, out_ch, 1×1, bias=True)
```

**Why depthwise convolutions are always skipped:**
Depthwise convolutions have `groups == in_channels`, meaning each input channel
has exactly one output filter - there's no cross-channel mixing. SVD decomposition
assumes a rank-r approximation of the full cross-channel weight matrix. For
depthwise convolutions, the weight matrix is block-diagonal (each block is 1×kH×kW).
Decomposing this and re-folding into non-depthwise convolutions would require
`groups=1` in the output layers, changing the operation completely and breaking
the depthwise structure that makes the architecture efficient. EfficientNet's MBConv
blocks use depthwise convolutions heavily - they are all skipped.

**Why nn.MultiheadAttention is always skipped entirely (whole model):**
See [LRF + nn.MultiheadAttention Safety](#lrf--nnmultiheadattention-safety) above
- this is a hard compatibility issue, not a tuning knob. `apply_low_rank_factorization`
and `apply_adaptive_lrf` both detect it and return the model unchanged; the
pipeline detects it earlier still and disables LRF (and the epsilon search)
before anything runs.

**Why the output head is always skipped:**
See [LRF Output-Head Protection](#lrf-output-head-protection) above.

**The kernel size and latency tradeoff:**
Factorizing `Conv2d(C_in, C_out, k×k)` into `Conv2d(C_in, rank, k×k)` +
`Conv2d(rank, C_out, 1×1)` replaces one GPU kernel launch with two. On a GPU,
each kernel launch has a fixed overhead of ~0.3-0.5ms regardless of how small
the computation is. For small tensors (batch_size=1, 112×112 feature maps), this
overhead dominates the actual computation time. Result: LRF on 3×3 convolutions
makes the model slower even though it has fewer parameters.

For 1×1 (pointwise) convolutions, the factorization is `Linear(C_in, rank)` +
`Linear(rank, C_out)` at each spatial position. Here both operations are genuinely
smaller and GPU batch matrix multiply is efficient even at small sizes. LRF is
beneficial on 1×1 convolutions.

**Rule of thumb for `LRF_SKIP_LARGE_KERNELS`:**
- `False` (default) for EfficientNet, MobileNet - mostly 1×1 convolutions after depthwise skip
- `True` for ResNet, VGG, DenseNet, Inception - heavy 3×3 usage causes latency regression

**Configuration constants in main.py:**
```python
USE_LOW_RANK            = True
LRF_EPSILON             = 0.5
LRF_MIN_LAYER_SIZE      = 64
LRF_MIN_RANK            = 2
LRF_SKIP_LARGE_KERNELS  = True
LRF_FINE_TUNE_EPOCHS    = 3     
LRF_FINE_TUNE_LR        = 0.00001
```

`LRF_EPSILON`: The rank ratio. `rank = int(min(in_dim, out_dim) * epsilon)`.
- epsilon=0.1: Keep 10% of singular values. Very aggressive. High accuracy loss.
- epsilon=0.3: Keep 30%. Moderate compression, noticeable accuracy drop (2-5%).
- epsilon=0.5: Keep 50%. Standard starting point. ~1-3% accuracy drop.
- epsilon=0.7: Keep 70%. Conservative. <1% accuracy drop. Small compression.
- epsilon=0.9: Keep 90%. Minimal compression. Barely affects accuracy.

Use `FIND_OPTIMAL_EPSILON = True` to auto-select the best epsilon via search.
(Automatically skipped - and `USE_LOW_RANK` force-disabled - for any model
containing `nn.MultiheadAttention`; see above.)

`LRF_MIN_LAYER_SIZE`: Skip layers where `in_features ≤ this` or `out_features ≤ this`.
Default 64. Factorizing a `Linear(32, 32)` layer at epsilon=0.5 gives rank=16, which
means two layers of `Linear(32, 16)` + `Linear(16, 32)` - the parameter count barely
changes and the extra kernel launch costs more than it saves.

`LRF_MIN_RANK`: Skip layers where the computed rank < this value. At very low epsilon
on small layers, rank=1 can occur. A rank-1 approximation is a single outer product -
it destroys almost all the representational capacity of the layer. Default 2 prevents
this collapse.

`LRF_SKIP_LARGE_KERNELS`: When True, skips any Conv2d with kernel size > 1×1. This
prevents latency regression on architectures that use 3×3 or 7×7 convolutions heavily.

`LRF_FINE_TUNE_EPOCHS` / `LRF_FINE_TUNE_LR`: Recommended to increase whenever `LRF_EPSILON`
is aggressive enough to cause a noticeable accuracy drop - always knowledge
distillation, reusing `KD_TEMPERATURE` / `KD_ALPHA`.

---

### Adaptive Low-Rank Factorization

**What it does:**
Instead of using one global epsilon for all layers, computes the optimal epsilon
per layer analytically by examining how fast each layer's singular values decay.
Highly compressible layers (fast decay) get a small epsilon (aggressive compression);
layers with flat spectra (slow decay, all singular values roughly equal) get a
large epsilon (light compression).

**The key insight - singular value energy:**
```
energy_fraction(r) = sum(S[:r]²) / sum(S²)
```
This is the fraction of the layer's total representational "power" captured by
the top-r singular values. We find the minimum rank r* such that:
```
energy_fraction(r*) ≥ energy_threshold   (default 0.99 = retain 99%)
```
Then `epsilon* = r* / min(in_dim, out_dim)`.

**Why energy (S²) not magnitude (S):**
The Frobenius norm `||W||_F² = sum(all singular values squared)`. The approximation
error from truncating at rank r is `||W - W_r||_F² = sum(S[r:]²)`. Minimizing this
error means maximizing `sum(S[:r]²) / sum(S²)`. Energy fraction is the exact
measure of approximation quality.

**Example:**
Layer A: singular values [100, 95, 90, 2, 1, 0.5, ...] → top-3 capture 99.99% of energy → epsilon*=0.1
Layer B: singular values [10, 9.8, 9.6, 9.4, 9.2, ...] → top-3 capture only 30% → epsilon*=0.9

Global epsilon=0.5 would over-compress Layer A and under-compress Layer B.
Adaptive epsilon finds the right compression level for each layer individually.

No data or forward passes needed - uses only weight tensors. Runs in O(n) time
per layer. This is strictly better than grid search because it's the exact
analytic solution. The output head is excluded from analysis (same protection as
standard LRF), and the same `nn.MultiheadAttention` whole-model skip and optional
KD recovery fine-tune apply here too.

**Configuration constants in main.py:**
```python
LRF_ADAPTIVE         = True    # Use adaptive instead of global epsilon
LRF_ENERGY_THRESHOLD = 0.99   # Retain this fraction of SVD energy per layer
```

`LRF_ENERGY_THRESHOLD`:
- 0.99 (default): Very conservative. Retains 99% of each layer's information.
  High quality, moderate compression.
- 0.95: Good tradeoff. Some accuracy loss (<1%) with better compression.
- 0.90: More aggressive. 1-3% accuracy drop, better compression ratios.
- Values below 0.85 are rarely worth it - accuracy loss accelerates rapidly.

---

### Weight Clustering

**What it does:**
Runs k-means on each layer's weights and replaces the entire layer with a
`_ClusteredLinear` / `_ClusteredConv2d` module: k trainable centroid values
plus a fixed, compactly-packed per-weight assignment recording which
centroid each original weight position uses. The weight is reconstructed
each forward call as `centroids[assignment]`.

**The k-means process per layer:**
```
1. Flatten all weights to a 1D array: [w1, w2, ..., wN]
2. Run k-means with k centroids
3. Assign each weight to its nearest centroid
4. Store k centroid values (trainable) + N assignment indices (fixed)
```

**Why this compresses - for real, not just in theory:**
- Original: N float32 values = N × 4 bytes
- Clustered (k≤16): 16 centroids (64 bytes, negligible) + N 4-bit packed
  indices ≈ N/2 bytes → ~8x compression
- Clustered (k=256): 256 centroids (1 KB) + N uint8 indices ≈ N bytes → ~4x compression

An earlier version of this toolkit wrote centroid values back as a plain
float32 tensor (same dtype, same element count as the original) - which
meant `get_model_size_mb` could never measure any reduction from clustering
at all, regardless of k. The current `_ClusteredLinear`/`_ClusteredConv2d`
modules genuinely change the storage: their real parameters/buffers (k
centroids + packed indices) are what `get_model_size_mb` sums, so the
compression above is measured, not aspirational.

**Why it doesn't speed up inference by itself:**
Weight clustering is a storage format change. `_ClusteredLinear`/
`_ClusteredConv2d` still fully dequantize to a dense float tensor
(`centroids[assignment]`) before running the same floating point Linear/
Conv2d operation - same FLOPs as before, smaller storage. Actual inference
speedup requires a custom kernel that computes directly on 4-bit or 8-bit
indexed weights (like ARM NN or CoreML provide). Weight clustering prepares
the model for those kernels without implementing them.

**Fine-tuning after clustering - KD whenever a teacher is supplied (always, in
the real pipeline), and now genuinely preserves the k-value structure:**
K-means centroid placement is based on the weight distribution, not on the
loss landscape. Centroid values are often slightly off from the optimal
positions for accuracy. Short fine-tuning lets the centroid values drift
toward better accuracy-preserving positions, using knowledge distillation
against the original model. **The number of centroids doesn't change, and -
unlike an earlier version of this toolkit - fine-tuning genuinely only
adjusts centroid values now:** `centroids` is the only trainable piece of
`_ClusteredLinear`/`_ClusteredConv2d` (the per-weight `assignment` is a
fixed buffer), and backprop through `centroids[assignment]` (PyTorch
advanced indexing) naturally sums the gradient contribution from every
weight position sharing a centroid - the same mechanism that already makes
`nn.Embedding` train correctly when a token ID repeats within a batch. No
custom backward code was needed for this; it falls out of how indexing's
backward pass already works. (A plain cross-entropy fallback exists for
direct/library callers who omit `kd_teacher`, but the real pipeline never
takes that path.)

**Interaction with GPTQ:** the fixed pipeline order runs Clustering before
GPTQ. For Linear layers, `_ClusteredLinear` exposes `.weight` / `.bias` /
`.in_features` / `.out_features` exactly like a real `nn.Linear`, so
`apply_gptq_quantization` reads a real dense weight from it, Hessian-
corrects it, and replaces the whole layer with a fresh `_Int4Linear` -
clustering's role for those layers becomes pre-conditioning (a smoother,
fine-tuned starting point for GPTQ's own correction), not a competing claim
on final storage. Conv2d layers are never touched by GPTQ (a Linear-only,
per-column Hessian algorithm), so for Conv2d, clustering's compact storage
IS the final, lasting compression.

**Configuration constants in main.py:**
```python
USE_CLUSTERING           = True
CLUSTER_NUM_CLUSTERS     = 16
CLUSTER_FINE_TUNE_EPOCHS = 5
CLUSTER_FINE_TUNE_LR     = 0.0001
```

`CLUSTER_NUM_CLUSTERS` (k):
- k=8: 4-bit packed storage (same packing as k=16 - anything ≤16 uses the
  4-bit path). Very aggressive. Noticeable quality loss.
- k=16: 4-bit packed storage. Standard aggressive setting. Default.
- k=32, 64, 128: uint8 storage (any k>16 uses one byte per index - no
  5/6/7-bit packing is implemented, since the byte-alignment complexity
  isn't worth it for values this close to 256 anyway).
- k=256: uint8 storage. Same effective bit-width as INT8 weight quantization
  but with k-means-placed (non-uniform) values instead of a uniform grid.

The k-means clustering happens on CPU with scikit-learn. For a large model like
VGG-16 (138M parameters), this takes several minutes. For EfficientNet-B0 (5.3M),
it's fast.

`CLUSTER_FINE_TUNE_EPOCHS`: 5 is usually enough. 0 = skip entirely (faster, but
you permanently lose 2-5% accuracy). Never use >10 - more epochs risks the model
overfitting to the training set.

`CLUSTER_FINE_TUNE_LR`: Must be much lower than the original training LR. 1e-4 is
standard. If accuracy is still low after fine-tuning, try 5e-5.

---


### Knowledge Distillation Fine-tuning

**What it does:**
Fine-tunes the compressed model using soft probability targets from the original
(uncompressed) model as an additional training signal. Standard fine-tuning trains
only on hard labels (one-hot targets). KD uses the teacher's full probability
distribution as an extra supervision signal.

**This is the SOLE fine-tuning method used everywhere in the toolkit** -
Structured Pruning, Low-Rank Factorization, Weight Clustering, and the
final post-quantization recovery step all use knowledge distillation against
the original model. Plain cross-entropy only survives as a fallback path for
direct/library callers who explicitly omit a teacher.

**The loss function:**
```
loss = α × CrossEntropy(student_logits, hard_labels)      # task loss
     + (1-α) × T² × KLDiv(                                # distillation loss
           softmax(teacher_logits / T),
           softmax(student_logits / T))
```

**Why soft labels are better than hard labels:**
Hard label: "this is a cat" → [0, 0, 1, 0, 0, ...] (all probability on cat)
Teacher soft label: "82% cat, 12% tiger, 6% lion, ..."

The soft labels encode relationships between classes learned over the full training
run. "This cat image slightly resembles a tiger" is genuine information that the
hard label discards. The compressed model learns from this richer signal even on
a small calibration set, recovering 2-5% more accuracy than hard-label fine-tuning
with the same number of epochs.

**Temperature T:**
Dividing logits by T before softmax makes the distribution softer (more spread out).
At high T, the teacher's probabilities become more uniform, revealing more
inter-class relationships. The T² multiplier on the KL loss compensates for this:
softer distributions have smaller KL divergence values, and T² scales them back up
so alpha remains a meaningful mixing coefficient regardless of temperature.

**This step's accuracy gating is two-layered** - see
[KD's Cumulative Recovery Check & Bounded Retry](#kds-cumulative-recovery-check--bounded-retry)
above: the standard marginal gate (revert if KD itself made things worse), plus
a cumulative-vs-original-baseline check with a single bounded retry if recovery
fell short.

**Configuration constants in main.py:**
```python
USE_KD_FINETUNE = True
KD_EPOCHS       = 3
KD_LR           = 0.0001
KD_TEMPERATURE  = 4.0
KD_ALPHA        = 0.7
```
These are also now reused by Pruning's and LRF's own recovery fine-tunes - see
above. No separate per-technique KD hyperparameters were introduced; one
temperature/alpha pair governs every KD-based fine-tune in the toolkit.

`KD_EPOCHS`: 3 is usually sufficient for recovery. The teacher is providing
extra signal per sample, so fewer epochs achieve the same as more standard
fine-tuning epochs. Use 5 for large accuracy drops post-compression.

`KD_LR`: Same reasoning as clustering fine-tune LR - keep low. 1e-4 default.

`KD_TEMPERATURE` (T): Range 2–6. Higher temperature reveals more inter-class
relationships. For classification tasks: 4.0 is a good starting point.
For fine-grained classification (Flowers-102 with similar classes): try 6.0.
For coarse classification: 2.0 is often sufficient.

`KD_ALPHA`: Balance between task loss and distillation loss.
- alpha=1.0: Pure task loss. Equivalent to standard fine-tuning. No KD benefit.
- alpha=0.7 (default): 70% task, 30% distillation. Good for recovery.
- alpha=0.5: Equal weight. Use when you have a well-calibrated teacher and
  small fine-tuning dataset.
- alpha=0.0: Pure distillation. Useful for extreme compression where the
  compressed model has very different capacity than the teacher.

---

### Standard Quantization

**What it does:**
Reduces the numerical precision of model parameters. Float32 (4 bytes/value) →
Float16 (2 bytes) or INT8 (1 byte). Smaller datatypes mean less memory bandwidth,
which is the primary bottleneck for inference on modern hardware.

**Three modes:**

**fp16 (recommended for GPU):**
Casts all parameters to `torch.float16`. Halves memory footprint of every parameter.
RTX GPUs have Tensor Cores that run fp16 matrix multiplications natively at 2-8×
the throughput of float32. No calibration data needed. No accuracy measurement
needed. The model runs normally - PyTorch handles the casting internally.

**dynamic INT8 (for CPU deployment):**
Quantizes only Linear layer weights to INT8. Conv2d layers stay float32 (no CUDA
INT8 kernel for Conv2d in PyTorch). The weight is stored as int8 (1 byte) and
dequantized to float32 just before each matrix multiplication. This gives ~4x
weight memory reduction on Linear layers. Actual inference still runs in float32.

Why the toolkit uses `_manual_dynamic_quantize` instead of PyTorch's
`quantize_dynamic`: After LRF, layers are wrapped in `nn.Sequential` containers.
PyTorch 2.x's dynamic quantization dispatcher cannot handle these containers and
raises "apply_dynamic is not implemented for this packed parameter type". The
manual implementation walks the tree recursively, finds every `nn.Linear` leaf
regardless of nesting, and replaces it with `_Int8Linear` which stores weights
as int8 and dequantizes in `forward()`.

**static INT8 (auto-switched to dynamic):**
Computes activation scales from calibration data, enabling full INT8 computation
for both weights and activations. Requires representative calibration batches.
However, EfficientNet's residual additions use plain Python `+` operator, which
becomes `aten::add` after quantization. This operator has no QuantizedCPU kernel,
causing a crash: "Could not run 'aten::add.out' from 'QuantizedCPU' backend."
The pipeline auto-switches static to dynamic and prints a warning.

**Pipeline-level accuracy gate:** like every other technique, quantization's
result is measured before/after and reverted if its own marginal accuracy drop
exceeds `ACCURACY_DROP_THRESHOLD`. On revert, the model correctly stays on
whatever device it was on before quantization ran (dynamic quant's CPU move
never happened to the model that gets kept) - see
[Global Accuracy-Drop Threshold](#global-accuracy-drop-threshold--per-technique-gating).

**Configuration constants in main.py:**
```python
USE_QUANTIZATION              = True
QUANT_MODE                    = "fp16"
QUANT_NUM_CALIBRATION_BATCHES = 100
```

`QUANT_MODE`:
- `"fp16"`: Best for GPU deployment. Use this.
- `"dynamic"`: Use when you need CPU deployment or have no GPU.
- `"static"`: Auto-switched to dynamic - don't bother setting this.

`QUANT_NUM_CALIBRATION_BATCHES`: Only used for static quantization (which is
auto-switched to dynamic). Has no effect in fp16 or dynamic mode.

---

### GPTQ Quantization

**What it does:**
Applies Hessian-corrected INT4 or INT8 quantization to Linear layers. Standard
quantization rounds each weight independently. GPTQ compensates for the
quantization error of each weight by adjusting the remaining weights to partially
cancel the error.

**The algorithm (simplified):**
```
1. Collect input activations X for this layer from calibration data
2. Estimate Hessian H ≈ 2 * X^T @ X    (correlation between input dimensions)
3. Compute H_inv via Cholesky factorization
4. For each column block of the weight matrix:
   a. Quantize columns to INT4 or INT8
   b. Compute quantization error for each weight
   c. Propagate error to remaining columns: w_j -= error_i * H_inv[i,j] / H_inv[i,i]
```

**Why this is better than standard INT4:**
Standard INT4 rounding: each weight is rounded independently → errors accumulate
additively → 5-15% accuracy drop.
GPTQ: after quantizing weight w_i, the Hessian tells us which other weights have
correlated influence on the layer output. Adjusting them compensates for w_i's
error → errors partially cancel → 0.5-2% accuracy drop at INT4.

**Why GPTQ matters:**
INT4 = 8× compression vs float32, 4× vs INT8. At this compression ratio, standard
quantization is unusable (5-15% accuracy loss). GPTQ makes INT4 viable with
<2% accuracy loss. This is the technique that made 4-bit LLM quantization practical.

**Storage:** weights are packed as real INT4 (`_Int4Linear` - 2 values per byte,
genuine 8× reduction vs float32, measured by `get_model_size_mb` since it sums
both `parameters()` and `buffers()`) plus a per-block float16 scale. True INT4
*compute* speedup (not just storage) still requires a custom inference kernel
(like llama.cpp or TensorRT INT4) that operates directly on packed 4-bit
values - this implementation dequantizes to float before each forward pass,
so it's a genuine storage win today and export-ready for such a kernel later,
not yet a compute-time win by itself.

**Pipeline-level accuracy gate:** like every other technique, GPTQ's result is
measured before/after and reverted if its own marginal accuracy drop exceeds
`ACCURACY_DROP_THRESHOLD`. If GPTQ reverts, every downstream decision (whether
standard quantization runs, what device the final KD step uses) correctly
treats GPTQ as if it had never been requested for this run - see
[Global Accuracy-Drop Threshold](#global-accuracy-drop-threshold--per-technique-gating).

**Configuration constants in main.py:**
```python
USE_GPTQ         = True
GPTQ_BITS        = 4     # 4 = INT4 (8× compression), 8 = INT8 (4× compression)
GPTQ_CAL_BATCHES = 16    # Batches to collect activations for Hessian estimation
GPTQ_BLOCK_SIZE  = 128   # Columns processed per Hessian update block
```

`GPTQ_BITS`: 4 for maximum compression (8×), 8 for better accuracy (4×). INT4 with
GPTQ beats standard INT8 in accuracy while giving 2× better compression.

`GPTQ_CAL_BATCHES`: The Hessian estimation quality improves with more batches but
saturates quickly. 16 batches is sufficient for good rank ordering. More batches
= more accurate Hessian = better weight adjustments, but diminishing returns above 32.

`GPTQ_BLOCK_SIZE`: Standard is 128 (from the original GPTQ paper). Smaller blocks =
more Hessian updates = slightly better quality but slower. Larger blocks = faster
but slightly lower quality. Don't change unless benchmarking.

---

### GPTQ's frozen-weight problem, and why it's now trainable during KD

**The problem:** `_Int4Linear` stores `packed` / `scales` / `bias` ALL as
non-trainable buffers - a deliberate choice favouring a fast, frozen
inference path. But `torch.optim.Adam(student.parameters(), ...)` only ever
sees real `nn.Parameter`s. The pipeline's final post-quantization KD recovery
step exists specifically to "recover accuracy lost from every prior step at
once," explicitly including quantization - but with every GPTQ'd Linear layer
frozen, that step could never actually touch a single one of them. For a
model where GPTQ covers most Linear layers - exactly what this toolkit's own
model registry recommends for every NLP transformer and ViT (`"ViT is 99%
Linear - ideal for GPTQ"`, similarly for BERT/RoBERTa/DistilBERT/ALBERT/
DistilGPT-2) - the recovery step meant to compensate for GPTQ's cost was
silently only ever adjusting LayerNorms, embeddings, and whatever small head
fell below `min_layer_size`.

**The fix - Straight-Through Estimator (STE) quantization-aware training:**
`_Int4LinearQAT` (`compression.py`) keeps a real, full-precision `shadow_weight`
`nn.Parameter` alongside a FIXED per-block scale (taken from GPTQ's own
calibration). Its forward pass computes:
```
w_fake_quant = shadow_weight + (fake_quantize(shadow_weight) - shadow_weight).detach()
```
The forward VALUE equals `fake_quantize(shadow_weight)` exactly (rounded to
the same INT4 grid `_Int4Linear`'s packed storage would represent) - so the
model computes as if it were still genuinely INT4-constrained. But gradients
flow through the `.detach()`'d correction term as zero, landing entirely on
`shadow_weight` - so ordinary backprop and Adam update the real underlying
weight. No custom gradient code beyond this one standard identity.

**When this activates:** `apply_gptq_quantization(..., qat=True)` produces
`_Int4LinearQAT` layers instead of frozen `_Int4Linear` ones. The real
pipeline only requests this when a fine-tune is actually about to use it -
`run_compression_pipeline` passes `qat=USE_KD_FINETUNE`. With
`USE_KD_FINETUNE=False`, GPTQ produces the ordinary frozen `_Int4Linear`
exactly as before, with none of the memory overhead below.

**Memory cost while active:** one full float32 shadow copy per GPTQ'd Linear
layer, alongside its scale - genuinely doubles the weight memory of the
equivalent frozen layer, for as long as `_Int4LinearQAT` objects exist in the
model. This is intentional and temporary, not a permanent cost of using GPTQ:

**Collapse:** `collapse_qat_layers()` runs exactly once, immediately after the
final KD step finishes (`run_compression_pipeline`, regardless of whether
that step's own accuracy gate kept or reverted it) - every `_Int4LinearQAT`
is converted back into a clean, frozen `_Int4Linear` built from the shadow's
final (fine-tuned) values, and the shadow is discarded. From this point on
the model is indistinguishable in structure from one that went through
ordinary (non-QAT) GPTQ - same compact packed storage, just with weights that
have now actually been adjusted by the recovery fine-tune. Safe to call
unconditionally (a no-op scan when QAT was never used).

**A known, accepted precision interaction:** when `QUANT_MODE='fp16'` runs
after GPTQ (the common GPTQ+fp16 combination), it casts the whole model -
including `_Int4LinearQAT`'s `shadow_weight` - to float16 before the KD step
gets to it. `fine_tune_with_distillation` immediately upcasts the whole
student back to float32 at the start of fine-tuning regardless, so this
doesn't break anything, but the float32→float16→float32 round-trip does lose
some of the shadow weight's starting precision before training begins. Given
the shadow weight is fundamentally headed toward an INT4-constrained
representation anyway (and gets adjusted by gradient descent immediately
after), this is treated as an acceptable, minor cost rather than something
worth adding selective-dtype-skip machinery to `apply_quantization` for.

**bits=8 never needed any of this:** that path already leaves a plain, fully
trainable `nn.Linear` behind (it mutates `weight.data` on a real `nn.Linear`
in place, or constructs a fresh one), so QAT/STE is scoped to `bits=4` only,
where the frozen-buffer problem actually exists.

---


## Search Algorithms

### Epsilon Search (LRF)

**When it runs:** `FIND_OPTIMAL_EPSILON = True` and `USE_LOW_RANK = True` (both
are force-disabled automatically if the active model contains
`nn.MultiheadAttention` - see
[LRF + nn.MultiheadAttention Safety](#lrf--nnmultiheadattention-safety)).

**What it does:** Finds the LRF epsilon that maximizes the Compression Quality Index
across accuracy, size, and (optionally) latency.

**Algorithm - anchor sampling + ternary search:**
```
Phase 1: Evaluate 5 fixed anchor epsilons [0.1, 0.25, 0.5, 0.75, 0.9]
Phase 2: Find the interval around the best anchor (binary search refinement)
Phase 3: Pool the ENTIRE cache, filter to candidates within
         accuracy_drop_threshold, select the highest-scoring survivor
```

Why ternary search and not binary search: binary search finds the zero of a function
(where f(x)=0). We want the maximum of a function, not a zero. Binary search
applied to maximization requires a strictly unimodal landscape (one peak). The LRF
score landscape can be approximately unimodal but noisy. Ternary search is the
correct algorithm for finding a maximum in a unimodal landscape: it evaluates two
interior points and eliminates the third that has the lower score.

**Structural-failure abort:** if 2 consecutive trials (anywhere across Phases 1-2)
fail with "No samples were evaluated", the search aborts the remaining trials and
proceeds straight to Phase 3 using whatever succeeded - see
[Structural-Failure Abort Protocol](#structural-failure-abort-protocol-search-loops).

**Configuration constants in main.py:**
```python
FIND_OPTIMAL_EPSILON      = True
EPSILON_SEARCH_NUM_TRIALS = 15
ACCURACY_DROP_THRESHOLD   = 10.0   # global - see dedicated section above
```

`EPSILON_SEARCH_NUM_TRIALS`: Total evaluations allocated to the search.
- Budget = 5 anchors + remaining / 2 ternary iterations.
- At 20 trials: 5 anchors + 7-8 ternary iterations. Usually sufficient.
- At 30 trials: 5 anchors + 12 iterations. More precise narrow range.
- At 10 trials: Only 2-3 ternary iterations. May not converge.

**Why no proxy model for epsilon search:**
The original design included an INT8 proxy model to speed up search. The proxy
always runs on CPU (no CUDA INT8 Conv2d kernel). For EfficientNet-B0 at 1-sample
batch size, CPU inference is ~10× slower than GPU. Using the proxy would make
epsilon search take 10× longer, which defeats the purpose entirely. All epsilon
search evaluations run on the real model on the GPU.

**Results are cached to `epsilon_cache.json`** for crash recovery. If the search
is interrupted and restarted, it resumes from cached evaluations.

---

### Pruning Hyperparameter Search

**When it runs:** `FIND_OPTIMAL_PRUNING = True`

**What is searched:**
- `pruning_ratio` (continuous, 0.05 to PRUNING_MAX_RATIO): the global channel removal fraction
- `pruning_max_ratio` (discrete grid): the per-layer cap
- `iterative_steps` (1 vs 2): single-shot vs two-round pruning

**What is fixed during search:**
- `fine_tune_epochs` per trial (`PRUNING_SEARCH_FT_EPOCHS` - fast approximation;
  the real pipeline uses `PRUNING_FINE_TUNE_EPOCHS`)
- `fine_tune_lr` per trial (`PRUNING_SEARCH_FT_LR`)
- `num_calibration_batches = 10` (reduced from `PRUNING_CAL_BATCHES` for speed)

**Algorithm - every phase now respects the accuracy-drop threshold, not just Phase 1:**
```
Phase 1:  5 anchor ratios evenly spaced in [0, max_ratio]
          → tiered selection: strict (≤threshold) → relaxed (≤threshold+10pp)
            → score-only, in that preference order
Phase 1b: Ternary refinement of ratio in the best anchor's interval
          → same tiered selection across ALL ratio-phase evaluations
Phase 2:  Grid search over max_ratio candidates at the best ratio
          → same tiered selection
Phase 3:  Compare iterative_steps=1 vs iterative_steps=2
          → same tiered selection
Phase 4:  Pool EVERY evaluation from the ENTIRE search, filter to candidates
          within accuracy_drop_threshold, pick the highest score among
          survivors (mirrors the epsilon search's final-selection logic)
```
See [Pruning Search - Phase 4 Rewritten](#pruning-search--phase-4-rewritten-bug-fix)
above for why this changed and exactly what bug it fixes - this is the most
significant correctness change in this version of the search.

**Structural-failure abort:** shares the same 2-consecutive-failure protocol as
the epsilon search, via a single internal choke-point every phase calls through.

**CQI with KL divergence:**
The pruning search includes KL divergence in the scoring:
```
CQI = (acc/baseline_acc)^w_acc × (baseline_size/size_mb)^w_size
    × (1/(1+KL))^w_kl
```
A config with KL=2.0 has its score halved vs KL=0. The search naturally avoids
pruning configurations that catastrophically change model behaviour, not just
those that lose accuracy. **This is now reinforced, not replaced, by the
accuracy-drop tiering above** - KL divergence shapes the score within a tier;
accuracy-drop tiering decides which tier wins.

**Configuration constants in main.py:**
```python
FIND_OPTIMAL_PRUNING      = True
PRUNING_SEARCH_NUM_TRIALS = 15
ACCURACY_DROP_THRESHOLD   = 10.0   # global - see dedicated section above
```

**Results cached to `pruning_search_cache.json`** for crash recovery.

---

## Compression Quality Index (CQI)

The single scoring metric used everywhere: epsilon search, pruning search, and
the compression report.

```
CQI = (acc / baseline_acc)^w_accuracy
    × (baseline_size / size_mb)^w_size
    × (baseline_lat / lat)^w_latency     [when latency available]
    × (1 / (1 + KL))^w_kl               [when KL available, pruning only]
```

Each factor is a ratio raised to its weight exponent. Default weights of 1.0
give the unweighted product. Higher weights amplify that factor's influence
superlinearly:
- w=1: `acc_ratio` - linear penalty for accuracy loss
- w=2: `acc_ratio²` - accuracy loss penalized quadratically
- w=3: `acc_ratio³` - search will very strongly avoid accuracy drops

**Interpretation:**
- CQI = 1.0: No improvement over original model (all ratios = 1.0)
- CQI = 2.0: Twice as good on the combined weighted tradeoff
- CQI = 0.5: Half as good - net loss from compression (viability threshold)
- CQI < 0.5: Compression is causing more harm than good; search returns None

**These weights are yours to tune - the toolkit will never adjust them
automatically.** If a high `CQI_W_SIZE` causes raw scores to favor a
heavily-compressed-but-accuracy-poor region of the search space (exactly the
failure mode that originally motivated the Phase 4 rewrite above), the
correct lever is `ACCURACY_DROP_THRESHOLD` - a hard gate that search and the
pipeline both respect regardless of how the weights are set - not a change to
the weights themselves.

**Configuration constants in main.py:**
```python
CQI_W_ACCURACY = 1.0
CQI_W_SIZE     = 2.0
CQI_W_LATENCY  = 1.0
CQI_W_KL       = 1.0
```

**Examples of tuning CQI weights for different deployment goals:**
- Prioritize speed: `CQI_W_LATENCY = 3.0` → search strongly prefers lower latency
- Protect accuracy: `CQI_W_ACCURACY = 2.0` → accuracy loss penalized quadratically
- Prevent behavioral drift: `CQI_W_KL = 3.0` → search avoids pruning configs with high KL
- Only care about size: `CQI_W_LATENCY = 0.0, CQI_W_ACCURACY = 0.5, CQI_W_SIZE = 2.0`

---

## Hyperparameter Reference - main.py Constants

**main.py is the sole source of truth for every hyperparameter.**
All constants are applied unconditionally in `main()` - changing a value here
always takes effect, no matter what argparse defaults exist. You never need to
touch any other file to adjust behaviour.

CLI flags (`--no-clustering`, `--device cpu`, etc.) only control boolean
enable/disable switches. Every numeric and string value always comes from main.py.

```python
# ── Active Model ──────────────────────────────────────────────────────────────
ACTIVE_MODEL = 'resnet50'
# Which model to compress. See model_registry.py for all 20 options.
# Changing this automatically loads the right dataset, input shape, and
# recommended technique config.
# NOTE: any architecture containing nn.MultiheadAttention (e.g. vit_b_16) gets
# USE_LOW_RANK and FIND_OPTIMAL_EPSILON force-disabled at RUNTIME regardless
# of the values set below - see "LRF + nn.MultiheadAttention Safety" above.

# ── Paths ─────────────────────────────────────────────────────────────────────
PRETRAIN_MODEL_PATH    = f'models/{ACTIVE_MODEL}.pth'
REPORT_SAVE_PATH       = 'compression_report.png'
EPSILON_LANDSCAPE_PATH = 'epsilon_landscape.png'
ONNX_SAVE_PATH         = 'compressed_model.onnx'

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_SAMPLE_SIZE = 2000
# 0 = full training set. Positive int = first N samples.
# Affects: clustering fine-tune, pruning calibration, LRF fine-tune, KD fine-tune.

TEST_SAMPLE_SIZE = 500
# 0 = full test set. 500 is fast for iteration. Use 0 for final results.
# Also used by every per-technique accuracy gate now - a smaller test set
# means faster gating checks but noisier accuracy estimates for the gate's
# revert decisions.

BATCH_SIZE = 32

# ── Model ─────────────────────────────────────────────────────────────────────
NUM_CLASSES = 102   # Fallback only - registry sets this automatically per model.

# ── Pre-training ──────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS = 10
PRETRAIN_LR     = 0.001
# Used only when no checkpoint exists. Registry may override per-model
# (e.g. Transformers use lr=0.00002, 5 epochs).

FORCE_RETRAIN = False   # True = retrain even if checkpoint exists.

# ── Technique enable/disable flags ────────────────────────────────────────────
USE_BN_FUSION    = True    # Always True. Zero accuracy cost. Never gated.
USE_PRUNING      = True    # Structured filter pruning. Gated.
USE_LOW_RANK     = True    # SVD factorization. Gated. Auto-disabled for
                           # nn.MultiheadAttention architectures.
USE_CLUSTERING   = True    # K-means weight quantization. Gated.
USE_KD_FINETUNE  = True    # Knowledge distillation fine-tuning. Gated +
                           # cumulative-recovery retry check.
USE_QUANTIZATION = True    # fp16 / dynamic INT8. Gated.
USE_GPTQ         = True    # Hessian-corrected INT4/INT8 on Linear layers. Gated.
LRF_ADAPTIVE     = True    # Per-layer epsilon from SVD energy (better than search).

# ── Structured Pruning ────────────────────────────────────────────────────────
PRUNING_RATIO             = 0.3
# Target fraction of channels to remove globally. 0.3 = 30% of channels removed.
# The search (FIND_OPTIMAL_PRUNING=True) will override this with the best value found.

PRUNING_MAX_RATIO         = 0.5
# Maximum fraction any single layer/coupled-group can be pruned, enforced via
# pruning_ratio_dict (see "Architecture-Agnostic Residual Group Detection").
# Previously hardcoded to a no-op 1.0 by two interacting bugs - both fixed;
# this constant now actually takes effect. Set lower (e.g. 0.2) to protect
# individual layers more aggressively; this is the ONLY place to change this
# value for the whole toolkit. Also the fallback for PRUNING_RESIDUAL_MAX_RATIO
# below.

PRUNING_RESIDUAL_MAX_RATIO = None
# Ceiling specifically for auto-detected residual/skip-connection-coupled
# groups. None (default) = fall back to PRUNING_MAX_RATIO above. Set an
# explicit number to give residual-critical layers a DIFFERENT ceiling than
# everything else. Applies to both the pruning search and the final pipeline
# run - no CLI flag for this one, main.py constant only.

PRUNING_MODEL_TYPE        = 'classifier'
# Used for the behavioral probe after pruning. Options: 'classifier', 'embedding', 'generative'.
# Controls whether KL divergence or cosine similarity is used in the probe.

PRUNING_FINE_TUNE_EPOCHS  = 3      # KD recovery epochs after pruning (always KD now).
PRUNING_FINE_TUNE_LR      = 0.0001
PRUNING_CAL_BATCHES       = 50     # Activation-statistics calibration batches.
PRUNING_ITERATIVE_STEPS   = 1      # 1 = single-shot, 2 = two-step (more stable).
PRUNING_ROUND_TO          = None   # Set 8 or 16 for Tensor Core channel alignment.
PRUNING_ISOMORPHIC        = False  # True = identical structure across coupled groups.
PRUNING_REPORT_PATH       = 'pruning_report.png'

# ── Low-Rank Factorization ────────────────────────────────────────────────────
LRF_EPSILON              = 0.5
# Rank ratio for standard LRF. Lower = more compression, more accuracy loss.
# Only used when LRF_ADAPTIVE=False. Otherwise adaptive per-layer epsilons are used.

LRF_MIN_LAYER_SIZE       = 64     # Skip layers with dim <= this (too small to benefit).
LRF_MIN_RANK             = 2      # Minimum rank (prevents rank-1 collapse).
LRF_SKIP_LARGE_KERNELS   = True   # Skip 3x3+ convs on CNN models (latency regression risk).
LRF_FINE_TUNE_EPOCHS     = 3      
LRF_FINE_TUNE_LR         = 0.00001

# Adaptive LRF
LRF_ENERGY_THRESHOLD = 0.99
# Fraction of SVD energy to retain per layer. 0.99 = keep 99% of weight information.
# Lower = more compression. Range: 0.90 (aggressive) to 0.999 (conservative).

# ── Weight Clustering ─────────────────────────────────────────────────────────
CLUSTER_NUM_CLUSTERS     = 16
# k for k-means. Each weight becomes one of k centroids.
# k=8 aggressive, k=16 default, k=32 conservative, k=256 = uint8 (max compression).

CLUSTER_FINE_TUNE_EPOCHS = 5      # Fine-tune epochs after clustering (0 = skip).
CLUSTER_FINE_TUNE_LR     = 0.0001

# ── Knowledge Distillation Fine-tuning ───────────────────────────────────────
KD_EPOCHS      = 3      # Fine-tune epochs. 3-5 is usually enough.
KD_LR          = 0.0001 # Keep small - recovering, not retraining.
KD_TEMPERATURE = 4.0    # Softmax temperature (2-6). Higher = softer teacher output.
KD_ALPHA       = 0.7    # 0.7 = 70% task loss + 30% distillation loss.
# Also reused by Pruning's and LRF's own recovery fine-tunes - one
# temperature/alpha pair governs every KD-based fine-tune in the toolkit.

# ── Quantization ──────────────────────────────────────────────────────────────
QUANT_MODE                    = "fp16"
# "fp16" = half-precision (GPU, ~2x memory reduction, Tensor Core speedup).
# "dynamic" = INT8 on Linear weights only (CPU, no calibration needed).
# "static" = auto-switched to dynamic (residual additions crash on QuantizedCPU).

QUANT_NUM_CALIBRATION_BATCHES = 100

# ── GPTQ ──────────────────────────────────────────────────────────────────────
GPTQ_BITS        = 4    # 4 = INT4 (8x compression), 8 = INT8 (4x compression).
GPTQ_CAL_BATCHES = 16   # Batches to estimate Hessian from activations.
GPTQ_BLOCK_SIZE  = 128  # Standard GPTQ block size.

# ── Hyperparameter Search ─────────────────────────────────────────────────────
FIND_OPTIMAL_EPSILON      = True   # Run LRF epsilon search before compressing.
                                    # Auto-skipped for nn.MultiheadAttention models.
FIND_OPTIMAL_PRUNING      = True   # Run pruning ratio search before compressing.

ACCURACY_DROP_THRESHOLD   = 10.0
# GLOBAL - applies to every search AND every pipeline technique's per-step
# gate (see "Global Accuracy-Drop Threshold & Per-Technique Gating" above).
# Searches: anchors for refinement use threshold+10pp (relaxed); final
#   selection uses this strict threshold; falls back to best-available only
#   if nothing meets even the relaxed tier.
# Pipeline: each technique (Pruning, LRF, Clustering, GPTQ, Quantization, KD)
#   is independently reverted if ITS OWN marginal accuracy drop exceeds this.

EPSILON_SEARCH_NUM_TRIALS  = 20   # Total LRF epsilon evaluations.
PRUNING_SEARCH_NUM_TRIALS  = 20   # Total pruning evaluations.
PRUNING_SEARCH_FT_EPOCHS   = 3    # Fine-tune epochs PER TRIAL during pruning search
                                   # (fast approximation; real pipeline uses
                                   # PRUNING_FINE_TUNE_EPOCHS).
PRUNING_SEARCH_FT_LR       = 1e-4 # Fine-tune lr per trial during pruning search.

FINE_TUNE_ABORT_THRESHOLD  = 40.0
# Direct pp value (not a multiplier) - see "Early-Abort Threshold for
# Recovery Fine-Tunes" above. Applies to Pruning's and LRF's recovery
# fine-tunes, in both the real pipeline and their searches. Has little
# effect on the epsilon search specifically since EPSILON_SEARCH_FT_EPOCHS=1
# leaves nothing to abort out of. Raised from 30.0 after a real run showed a
# recoverable-looking config (34.7pp epoch-1 drop) get cut off too early.

# ── CQI Weights ───────────────────────────────────────────────────────────────
CQI_W_ACCURACY = 1.0   # Weight for accuracy factor in CQI scoring.
CQI_W_SIZE     = 1.0   # Weight for size / compression ratio factor.
CQI_W_LATENCY  = 1.0   # Weight for latency / speedup factor.
CQI_W_KL       = 1.0   # Weight for KL divergence penalty (pruning only).
# Increase a weight to make that factor dominate search decisions.
# Example: CQI_W_ACCURACY=3, CQI_W_SIZE=1 → search prioritises accuracy.
# These are yours to tune; ACCURACY_DROP_THRESHOLD (not these weights) is the
# hard gate that prevents a high CQI_W_SIZE from selecting an
# accuracy-destroying configuration - see the CQI section above.

# ── ONNX ──────────────────────────────────────────────────────────────────────
EXPORT_ONNX = False
RUN_ONNX    = False
ONNX_OPSET  = 17
```
---

## Model Registry - 20 Models

All models in `model_registry.py`. Use `python main.py --list-models` to print them.

| ACTIVE_MODEL | Architecture | Dataset | Params | Notes |
|---|---|---|---|---|
| `efficientnet_b0` | NAS/MBConv | flowers102 | 5.3M | Default. Depthwise convs skipped by LRF |
| `resnet18` | ResNet | cifar100 | 11.2M | Smallest ResNet. Skip 3×3 for LRF |
| `resnet50` | ResNet | cifar100 | 25.6M | Bottleneck blocks. Skip 3×3 for LRF |
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
| `vit_b_16` | ViT | cifar100 | 86.6M | **nn.MultiheadAttention - LRF and epsilon search auto-disabled at runtime, regardless of USE_LOW_RANK/FIND_OPTIMAL_EPSILON** |
| `swin_t` | Swin Transformer | cifar100 | 28.3M | Mostly Linear (NOT nn.MultiheadAttention). LRF very effective |
| `bert_base` | BERT | sst2 | 110.0M | Requires transformers+datasets |
| `distilbert` | DistilBERT | sst2 | 66.4M | Already distilled - does more help? |
| `roberta_base` | RoBERTa | sst2 | 125.0M | Uniform weights → clustering effective |
| `albert_base` | ALBERT | sst2 | 11.7M | Shared weights across layers |
| `distilgpt2` | GPT-2 | sst2 | 81.9M | Causal attention, classification |

Each model entry in the registry contains `recommended` settings - when you change
`ACTIVE_MODEL`, `main.py` reads these and auto-applies the right technique configuration
for `pretrain_epochs` and `pretrain_lr` only (every other setting is controlled
exclusively by main.py's constants, per its own docstring). The
`nn.MultiheadAttention` safety check is a separate, runtime correctness check
(`model_has_multihead_attention()` in `helper_functions.py`) - it is not part of
the registry's `recommended` dict and cannot be overridden by it.

---

## Functions - compression.py

### `apply_bn_fusion(model)`
Fuses all Conv→BN and Linear→BN pairs. Zero accuracy cost. Returns new model.
Never gated.

### `apply_structured_pruning(model, dataloader, device, pruning_ratio, model_type, ..., max_pruning_ratio, residual_max_ratio=None, kd_temperature, kd_alpha, capture_pre_finetune_accuracy=False)`
Full structured pruning pipeline. Recovery fine-tune always uses
`_kd_recovery_fine_tune()` (knowledge distillation against `model`, this
function's own pre-pruning parameter). Returns pruned model with
`._pruning_report` (now also containing `pre_finetune_accuracy`) and
`._pre_finetune_accuracy`. `capture_pre_finetune_accuracy=True` (only ever
set by `apply_compression_pipeline`, never by the search functions in
`optimization.py`) measures accuracy once right after pruning but before
the recovery fine-tune, for per-technique impact reporting - see
"Per-Technique Impact Reporting" above. `residual_max_ratio` (`None` by
default, falling back to `max_pruning_ratio`) sets the ceiling for
auto-detected residual/skip-connection-coupled groups specifically - see
[Architecture-Agnostic Residual Group Detection](#architecture-agnostic-residual-group-detection).

### `_find_residual_coupled_groups(model, example_input, ignored_layers)`
Architecture-agnostic detection of residual/skip-connection-coupled
dependency groups via Torch-Pruning's own dependency graph - no module-name
matching. Distinguishes genuine addition nodes from ReLU/other activations
(both classified under Torch-Pruning's single `OPTYPE.ELEMENTWISE` category)
by checking each node's actual `grad_fn` class name for `'Add'`. Returns a
list of groups, each a list of the real `nn.Conv2d`/`nn.Linear` modules
coupled through that addition. Called from inside `apply_structured_pruning`
- see [Architecture-Agnostic Residual Group Detection](#architecture-agnostic-residual-group-detection)
for the full mechanism and empirical verification.

### `apply_low_rank_factorization(model, epsilon, min_layer_size, min_rank, skip_large_kernels, dataloader=None, device=None, num_classes=None, fine_tune_epochs=0, fine_tune_lr=0.0001, kd_teacher=None, kd_temperature=4.0, kd_alpha=0.7, capture_pre_finetune_accuracy=False)`
Standard global-epsilon LRF. Returns the model UNCHANGED (no factorization
applied anywhere) if `nn.MultiheadAttention` is detected. Always protects the
output head. Optional KD recovery fine-tune when `fine_tune_epochs > 0` and
`dataloader`/`device`/`num_classes` are all supplied. Now always returns
`._lrf_report` (`{factorized_count, skipped_count, mha_skip,
pre_finetune_accuracy}` - previously this data was print-only) and
`._pre_finetune_accuracy`. `capture_pre_finetune_accuracy=True` behaves as
described for pruning above - same opt-in contract, same "only the real
pipeline sets this" boundary.

### `compute_adaptive_epsilons(model, energy_threshold, min_layer_size, skip_large_kernels)`
Analytically computes per-layer optimal epsilon from SVD energy decay. Returns
`{layer_name: epsilon}` dict. No data needed. Excludes the output head from
analysis.

### `apply_adaptive_lrf(model, adaptive_epsilons, global_epsilon, min_layer_size, min_rank, skip_large_kernels, dataloader=None, device=None, num_classes=None, fine_tune_epochs=0, fine_tune_lr=0.0001, kd_teacher=None, kd_temperature=4.0, kd_alpha=0.7, capture_pre_finetune_accuracy=False)`
Applies LRF using per-layer epsilons from `compute_adaptive_epsilons()`. Falls
back to `global_epsilon` for layers not in the dict. Same `nn.MultiheadAttention`
safety, output-head protection, optional KD recovery fine-tune, and
`._lrf_report`/`._pre_finetune_accuracy`/`capture_pre_finetune_accuracy`
contract as `apply_low_rank_factorization`.

### `apply_weight_clustering(model, dataloader, num_clusters, fine_tune_epochs, fine_tune_lr, device, num_classes, layers_to_cluster, kd_teacher=None, kd_temperature=4.0, kd_alpha=0.7, test_loader=None, capture_pre_finetune_accuracy=False)`
K-means clustering + optional fine-tune. KD when `kd_teacher` is supplied
(always, in the real pipeline); plain cross-entropy fallback otherwise.
Returns new model with `._clustering_report`
(`{num_clusters, layers_clustered, layer_details, pre_finetune_accuracy}`)
and `._pre_finetune_accuracy` attached. `test_loader` is new and only used
when `capture_pre_finetune_accuracy=True` - this function has no
early-abort mechanism of its own to share `test_loader` with, unlike
pruning/LRF.

### `fine_tune_with_distillation(student, teacher, dataloader, num_classes, epochs, lr, device, temperature, alpha)`
KD fine-tune. Returns fine-tuned copy of student (student not modified), with
`._kd_history = [{epoch, loss, train_acc, test_acc}, ...]` attached. Includes
the epoch-1-exactly-0.0% safety check.

### `apply_gptq_quantization(model, dataloader, device, bits, num_calibration_batches, block_size, min_layer_size, target_layers)`
GPTQ INT4/INT8 quantization on Linear layers. Returns new model with weights
adjusted to INT4/INT8-quantized positions, and `._gptq_report` now also
containing `total_eligible_layers` (the denominator for `layers_quantized` -
was computed internally already but never returned, so a `0/0` result was
only ever visible in console output).

### `apply_quantization(model, dataloader, mode, num_calibration_batches)`
Standard quantization. mode = "fp16" | "dynamic" | "static".

### `apply_compression_pipeline(model, dataloader, device, ..., pruning_max_ratio=0.5, pruning_residual_max_ratio=None, test_loader=None, accuracy_drop_threshold=10.0, original_size_mb=None, original_latency_ms=None, input_shape=(1,3,224,224), input_dtype=None)`
Runs all enabled techniques in fixed order. This is what `run_compression_pipeline()`
in `helper_functions.py` calls internally. When `test_loader` is supplied, every
mutating technique (Pruning, LRF, Clustering, the dormant in-pipeline KD/GPTQ/
Quantization steps) is individually gated on its own marginal accuracy drop -
see "Global Accuracy-Drop Threshold & Per-Technique Gating" above - and a
detailed `[Impact]` block is printed for each via `report_technique_impact()`
- see "Per-Technique Impact Reporting" above. Returns the compressed model
with `._gating_report` and `._gating_accuracy` attached when gating was
enabled. `original_size_mb`/`original_latency_ms` let the caller (normally
`run_compression_pipeline`, which measures these once) avoid a redundant
re-measurement; `input_shape`/`input_dtype` feed every impact report's
latency measurement.

---

## Functions - helper_functions.py

### `setup_data(train_sample, test_sample, batch_size, device)`
Returns (train_loader, test_loader) for Oxford Flowers-102.

### `setup_data_for_model(model_name, train_sample, test_sample, batch_size, device)`
Returns (train_loader, test_loader) for any model in the registry (dispatches
to the right dataset automatically).

### `load_or_train_model(train_loader, test_loader, num_classes, model_path, epochs, lr, device, force_retrain)`
Returns EfficientNet-B0 fine-tuned for num_classes. Legacy function - use
`load_or_train_from_registry()` for all other models.

### `load_or_train_from_registry(model_name, train_loader, test_loader, model_path, epochs, lr, device, force_retrain)`
Returns any model from the registry, fine-tuned for its target task.

### `measure_accuracy(model, dataloader, device)`
Top-1 classification accuracy. Auto-casts input to model dtype (fp16 fix).
**Raises** `RuntimeError` (message contains `"No samples were evaluated"`) if
every batch failed structurally. **Prints a major warning** (but still
returns `0.0`) if at least one batch succeeded but literally nothing was
predicted correctly. Returns float percentage `[0.0, 100.0]` otherwise.

### `gate_technique_accuracy(technique_label, pre_model, post_model, test_loader, device, accuracy_drop_threshold, pre_accuracy=None)`
The shared per-technique accuracy gate. Measures (or accepts a known)
pre-accuracy, measures post-accuracy, compares the marginal drop against the
threshold, and either keeps `post_model` or reverts to `pre_model` (deleting
the discarded model and clearing CUDA cache). Returns
`(kept_model, kept_model_accuracy, was_kept)`.

### `report_technique_impact(technique_label, pre_model, post_model, pre_accuracy, post_accuracy, was_kept, pre_device, post_device, original_accuracy, original_size_mb, original_latency_ms, input_shape=(1,3,224,224), input_dtype=None, pre_size_mb=None, pre_latency_ms=None, pre_finetune_accuracy=None, structural_summary=None, structural_zero=False, latency_iterations=10, latency_warmup=3)`
`gate_technique_accuracy`'s companion - see "Per-Technique Impact Reporting"
above for the full explanation. Prints (and returns as a dict) marginal and
cumulative-vs-original accuracy/size/latency for one gated technique, plus
an algorithm-vs-fine-tune accuracy split when `pre_finetune_accuracy` is
supplied, plus an explicit "consider disabling this technique" warning when
`structural_zero=True`. Never re-measures accuracy (`pre_accuracy`/
`post_accuracy` are required - the gate has already measured both by the
time this is called); always measures size (cheap) and latency (reduced
fidelity, defensive) fresh for whichever of `pre_size_mb`/`pre_latency_ms`
isn't already supplied. Called only from `apply_compression_pipeline()` and
`run_compression_pipeline()` - never from `optimization.py`'s search
functions.

### `model_has_multihead_attention(model)`
Returns the count of `nn.MultiheadAttention` modules anywhere in `model`.
Used to detect LRF-incompatible architectures (ViT-B/16, etc.) before the
pipeline starts.

### `measure_latency(model, input_shape, device, num_iterations, warmup, input_dtype)`
Inference latency with GPU synchronization. Returns dict: mean_ms, median_ms,
p95_ms, p99_ms. `input_dtype=torch.long` for NLP models.

### `get_model_size_mb(model)`
Total parameter + buffer size in MB. Correctly handles fp16 (2 bytes) and
INT8 (1 byte) by using `element_size()`.

### `count_parameters(model)`
Count of `requires_grad=True` parameters only. Buffers excluded.

### `train_model(model, train_loader, test_loader, num_classes, epochs, lr, device, save_path)`
Generic training loop. Adam + ReduceLROnPlateau. Saves best checkpoint.

### `export_to_onnx(model, save_path, input_shape, device, opset_version, dynamic_batch)`
Export float32 model to ONNX. Verifies graph with onnx.checker. Raises
RuntimeError if INT8 quantized ops are present (not ONNX-traceable).

### `run_onnx_inference(onnx_path, dataloader, input_shape, num_latency_iterations, warmup)`
Load ONNX model, measure accuracy + latency via ONNX Runtime. CPU only.

### `run_compression_pipeline(args)`
Full pipeline orchestration. Called by `main()`. Can also be called
programmatically with a manually constructed `argparse.Namespace`. Performs
the `nn.MultiheadAttention` safety check, measures baseline accuracy, size,
and latency once each for reuse throughout (size/latency are new - measured
alongside the pre-existing once-only baseline accuracy, specifically so
every per-technique impact report's "cumulative vs. original" numbers share
one true-original reference), runs the (optional) pruning/epsilon searches,
runs Phase A (`apply_compression_pipeline`, gated, with impact reporting)
and Phase B (GPTQ → Quantization → final KD, each gated and impact-reported
here directly), then evaluates and reports.

---

## Functions - optimization.py

### `compression_quality_index(accuracy, size_mb, baseline_accuracy, baseline_size, latency_ms, baseline_latency_ms, kl_divergence, w_accuracy, w_size, w_latency, w_kl)`
The single shared metric for all search and reporting. Returns float CQI ≥ 0.

### `find_optimal_epsilon_smart(model, test_loader, device, baseline_accuracy, baseline_size, baseline_latency_ms, tolerance, num_trials, cache_path, min_layer_size, input_shape, w_accuracy, w_size, w_latency, accuracy_drop_threshold)`
Anchor + ternary search for best LRF epsilon. Returns (optimal_epsilon, all_results, history).
`optimal_epsilon` is `None` if best CQI < 0.5, or if nothing survived the
2-consecutive-structural-failure abort protocol.

### `find_optimal_pruning_params(model, dataloader, test_loader, device, baseline_accuracy, baseline_size, baseline_latency_ms, num_trials, cache_path, model_type, num_classes, num_calibration_batches, max_pruning_ratio, residual_max_ratio=None, round_to, isomorphic, input_shape, w_accuracy, w_size, w_latency, w_kl, accuracy_drop_threshold, search_ft_epochs, search_ft_lr)`
Four-phase search for pruning_ratio, max_pruning_ratio, iterative_steps -
every phase now accuracy-drop-tiered, Phase 4 pools the entire cache (see
"Pruning Search - Phase 4 Rewritten" above). Returns
(optimal_params_dict, all_results, history). `optimal_params` is `None` if
no config is viable.

### `_select_best_by_tier(entries, baseline_accuracy, accuracy_drop_threshold)`
Shared helper used by every phase of `find_optimal_pruning_params`: prefers
the highest-scoring entry within `accuracy_drop_threshold`, falls back to
`+10pp` relaxed, falls back to pure score-max only as a last resort.

---

## Functions - visualization.py

### `generate_compression_report(original_model, compressed_model, metrics_dict, save_path, dataloader, device, input_shape, latency_iterations, latency_warmup)`
Generates the 4-panel compression summary PNG. Called automatically at pipeline end.
Already wraps its `measure_accuracy` calls in broad exception handling, so the
new structural-failure `RuntimeError` degrades gracefully into a "N/A" panel
rather than crashing report generation.

### `plot_epsilon_landscape(results, search_history, optimal_epsilon, baseline_accuracy, baseline_size, save_path)`
4-panel epsilon search result visualization. Called automatically when FIND_OPTIMAL_EPSILON=True.

### `plot_pruning_report(pruning_report, save_path)`
4-panel pruning result visualization. Called automatically when USE_PRUNING=True
AND pruning survived its accuracy gate.

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
"Global Accuracy-Drop Threshold & Per-Technique Gating" above. Raise
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
Two consecutive structural failures - see "Structural-Failure Abort Protocol"
above. The search still completes (using whatever succeeded before the
abort), but if this happens on every run for a given model, that model likely
has a fundamental incompatibility with the technique being searched (check for
patterns like the `nn.MultiheadAttention` case, or verify `num_classes` /
input shape are actually correct for this model).

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

## Contributing
This project isn't taking external pull requests yet
