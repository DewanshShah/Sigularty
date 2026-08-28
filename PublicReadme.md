# Using sigularty

This is a quick reference for calling sigularty as a library: how to
import it, the two ways to get a model in, and what every `compress()`
argument does. For the theory behind each technique, why pruning runs
before LRF, what CQI means, why quantization is always last, see
[README.md].

---

## Install

```bash
pip install sigularty
```

## Quickstart

**Fastest path: a registry model + its paired dataset in one call:**

```python
from sigularty import load_from_registry, compress

r = load_from_registry('resnet18', device='cuda')

result = compress(
    r.model,
    r.train_loader,
    num_classes=r.num_classes,
    use_pruning=True,
    use_lrf=True,
)

print(result)
# CompressionResult(ratio=3.42x, accuracy_retention=96.1%, size=42.7->12.5 MB, speedup=1.8x, cqi=2.113)

compressed_model = result.model
```

**Bring your own model:**

```python
from sigularty import compress

result = compress(model, train_loader, num_classes=num_classes)
compressed_model = result.model
```

`model` can be any `nn.Module`. `dataloader` (positional, second argument)
is used for calibration, fine-tuning, and evaluation throughout the
pipeline: it's never optional. The original model is never modified;
`compress()` always works on a deep copy.

---

## Other entry points

```python
from sigularty import (
    compress,               # run the full compression pipeline
    analyze,                # inspect a model, get technique recommendations
    load_from_registry,     # pull a registry model + its dataset together
    finetune,                # fine-tune a model before compressing it
    find_best_lr,            # LR range test: run before finetune()/pretrain_*
    plot_compression_report,
    plot_pruning_report,
    plot_epsilon_landscape,
)
```

| Function | Signature | Returns |
|---|---|---|
| `analyze(model, dataloader=None, *, device=None)` | Inspects a model without modifying it. | `AnalysisResult` |
| `load_from_registry(model_name, *, device=None, train_sample=None, test_sample=500, batch_size=32, model_path=None, force_retrain=False, pretrain_epochs=10, pretrain_lr=1e-4)` | See list of registry model names below. Sole supported registry entry point (see v1.2.0 note above). | `RegistryResult` |
| `finetune(model, train_loader, *, test_loader=None, num_classes=10, epochs=10, lr=1e-3, max_batches=0, device=None, save_path=None)` | Fine-tunes in place and returns the same object. | `nn.Module` |
| `find_best_lr(model, dataloader, *, device=None, num_classes=10, start_lr=1e-7, end_lr=10.0, num_steps=100)` | LR range test; run before `finetune()` or `pretrain_epochs > 0`. | `float` |

Registry model names (pass as `model_name`): `custom_cnn`, `resnet18`,
`resnet50`, `resnext50_32x4d`, `wide_resnet50_2`, `vgg16`, `densenet121`,
`convnext_tiny`, `regnet_y_400mf`, `shufflenet_v2_x1_0`, `squeezenet1_1`,
`mobilenet_v3_large`, `efficientnet_b0`, `inception_v3`, `vit_b_16`,
`swin_t`, `bert_base`, `distilbert`, `roberta_base`, `albert_base`,
`distilgpt2`.

---

## What `compress()` returns

```python
result.model                   # nn.Module: the compressed model, ready to use
result.compression_ratio       # size_original / size_compressed, e.g. 3.8 = 3.8x smaller
result.accuracy_retention      # (compressed_acc / original_acc) * 100
result.size_original_mb
result.size_compressed_mb
result.original_accuracy
result.compressed_accuracy
result.original_latency_ms
result.compressed_latency_ms
result.latency_speedup         # original_lat / compressed_lat, >1.0 = faster
result.cqi                     # Compression Quality Index: see README.md
result.techniques_applied      # list[str], only techniques that actually ran/survived their accuracy gate — see "Accuracy-drop gating" below
result.report_path             # path to the saved PNG report, or None
result.pruning_report          # dict of per-layer pruning detail, or None (also None if pruning ran but was reverted by the gate)
```

`analyze()` returns an `AnalysisResult`: `size_mb`, `num_parameters`,
`architecture_type` (`'cnn'` / `'transformer'` / `'hybrid'` / `'unknown'`),
`recommended_techniques`, `per_layer_signals` (a `LayerSignal` per eligible
layer: `name`, `layer_type`, `lrf_epsilon`, `prunable`, `size_kb`), and
`accuracy` if you passed a dataloader.

`load_from_registry()` returns a `RegistryResult`: `model`, `train_loader`,
`test_loader`, `num_classes`, `model_name`, `dataset_name`.

---

## `compress()` hyperparameters

Every argument after `model` and `dataloader` is keyword-only. Grouped
the same way the function itself groups them.

### Device

| Arg | Default | What it does |
|---|---|---|
| `device` | `None` | `'cuda'` or `'cpu'`. Auto-detected when not set. |

### Model metadata

| Arg | Default | What it does |
|---|---|---|
| `model_type` | `'unknown'` | Architecture class for pruning's safety clamp: `'classifier'`, `'embedding'`, `'generative'`, or `'unknown'`. |
| `num_classes` | `10` | Output class count, used by every fine-tune step's accuracy metric. |

### Pre-training

Use these when the model hasn't been trained on your target dataset yet
(e.g. fresh ImageNet weights with an untrained head, straight out of
`load_from_registry()` with no existing checkpoint).

| Arg | Default | What it does |
|---|---|---|
| `pretrain_epochs` | `0` | Epochs to fine-tune before compressing. `0` skips this entirely. |
| `pretrain_lr` | `1e-3` | Learning rate for that pre-training. |
| `pretrain_test_loader` | `None` | Validation loader for pre-training. Falls back to `dataloader`. |

### Hyperparameter search (optional, adds evaluations before the pipeline runs)

| Arg | Default | What it does |
|---|---|---|
| `find_optimal_epsilon` | `False` | Auto-search for the best LRF epsilon instead of using `lrf_epsilon` as-is. |
| `find_optimal_pruning` | `False` | Auto-search for the best pruning ratio instead of using `pruning_ratio` as-is. |
| `epsilon_search_trials` | `15` | Evaluation budget for the epsilon search. |
| `pruning_search_trials` | `16` | Evaluation budget for the pruning search. |
| `pruning_search_ft_epochs` | `1` | Fine-tune epochs per trial during the pruning search (kept low for speed: the real run uses `pruning_fine_tune_epochs`). |
| `pruning_search_ft_lr` | `1e-4` | Fine-tune learning rate per trial during the pruning search. |
| `accuracy_drop_threshold` | `5.0` | Max acceptable accuracy drop in percentage points: used by both searches' final selection AND the pipeline's per-technique revert gate (Phase A only — see "Accuracy-drop gating" below). |
| `early_abort_threshold` | `None` | **New.** Direct pp value (not a multiplier on `accuracy_drop_threshold`). After epoch 1 of Pruning's or LRF's own recovery fine-tune — in both the real pipeline and their searches — abort the remaining epochs if the drop vs. the original baseline already exceeds this. `None` (default) = disabled; every fine-tune always runs to completion. |
| `epsilon_cache_path` | `None` | **New.** Override the auto-derived per-model cache file for the epsilon search. `None` = `.sigularty_cache/epsilon_<model>.json`. |
| `pruning_cache_path` | `None` | **New.** Override the auto-derived per-model cache file for the pruning search. `None` = `.sigularty_cache/pruning_<model>.json`. |

**Accuracy-drop gating is always active in Phase A (new behaviour).**
Structured Pruning, Low-Rank Factorization, Weight Clustering, and the
in-pipeline KD step are each measured before/after and reverted to their
pre-technique state if that ONE technique's own marginal drop exceeds
`accuracy_drop_threshold`. Each technique gets an independent budget — an
earlier costly technique does not eat into a later technique's allowance.
A reverted technique will not appear in `result.techniques_applied` —
check the console output for a `❌ [TechniqueName] SKIPPED — accuracy
dropped ...` line if a technique you enabled seems to be missing.
**Phase B (standard Quantization, GPTQ) is not gated** — those two run
after Phase A completes and always apply once enabled; see README.md for
why they're handled differently.

**Search result caching (fixed — was previously a silent correctness
bug).** Each search caches trial results to disk purely by hyperparameter
value, with no reference to which model produced them. Earlier versions of
`compress()` always used the same two hardcoded filenames
(`epsilon_cache.json`, `pruning_search_cache.json`) for every call — so
compressing two different models from the same working directory could
silently reuse one model's cached trial results for the other, producing
search output that looked plausible but was measuring the wrong model
entirely. `epsilon_cache_path`/`pruning_cache_path` now default to a
filename derived from the model's class name + parameter count +
`num_classes`, so different models get separate cache files automatically.
Pass an explicit path yourself for a stronger guarantee (e.g. distinct
caches per dataset too, not just per model/class-count). If you have
`epsilon_cache.json` / `pruning_search_cache.json` left over from an older
version in your working directory, they're now orphaned — safe to delete.

### Technique enable flags

| Arg | Default |
|---|---|
| `use_pruning` | `False` |
| `use_lrf` | `True` |
| `use_clustering` | `True` |
| `use_quantization` | `True` |
| `use_kd_finetune` | `True` |
| `use_gptq` | `False` |

### Pruning hyperparameters

| Arg | Default | What it does |
|---|---|---|
| `pruning_ratio` | `0.3` | Target fraction of channels removed globally. |
| `pruning_max_ratio` | `0.95` | Hard cap on how much any single layer/dependency-group can be pruned. |
| `pruning_residual_max_ratio` | `None` | **New.** Ceiling specifically for auto-detected residual/skip-connection-coupled groups — the layers whose channel count IS the residual stream for an entire network stage, so collapsing them damages every downstream block in that stage, not just one layer's worth of capacity. `None` (default) falls back to `pruning_max_ratio` above. See README.md's "Architecture-Agnostic Residual Group Detection" section for the detection mechanism. |
| `pruning_model_type` | `'classifier'` | Same role as `model_type`, specific to pruning's clamp. |
| `pruning_fine_tune_epochs` | `3` | KD recovery epochs after pruning. |
| `pruning_fine_tune_lr` | `1e-4` | Learning rate for that recovery fine-tune. |
| `pruning_cal_batches` | `50` | Calibration batches for activation-statistics importance scoring. |
| `pruning_iterative_steps` | `1` | Prune in one shot (`1`) or multiple incremental rounds (more stable, slower). |
| `pruning_isomorphic` | `False` | Force identical pruning structure across coupled dependency groups. |
| `pruning_round_to` | `None` | Round pruned channel counts to a multiple of this (e.g. `8`/`16` for Tensor Core alignment). |

**Note:** `find_optimal_pruning=True` now correctly forwards
`pruning_max_ratio` (and `pruning_residual_max_ratio`) into the search
itself, and correctly reads the search's winning max-ratio back out
afterward. In earlier versions, the pruning search always ran uncapped
regardless of `pruning_max_ratio`, and even when the search picked a safe
config, its chosen max-ratio was silently discarded on the way back into
the real pipeline — the pipeline always fell back to whatever
`pruning_max_ratio` default was in effect (`0.95` unless you changed it).
Combined with gating being off (see above), this could let pruning
collapse the residual-critical layers at every stage boundary with nothing
to catch or revert it. Both are fixed now; if you were relying on
`find_optimal_pruning` before this fix, re-run — the pruning config it
actually applies may now be meaningfully different (and safer).

### Low-Rank Factorization (LRF) hyperparameters

| Arg | Default | What it does |
|---|---|---|
| `lrf_epsilon` | `0.5` | Rank ratio kept. Lower = more compression, more accuracy risk. |
| `lrf_adaptive` | `False` | Compute epsilon per layer from SVD energy decay instead of one global value. |
| `lrf_energy_threshold` | `0.99` | Fraction of SVD energy retained per layer, when adaptive. |
| `lrf_min_layer_size` | `64` | Skip layers with a dimension below this. |
| `lrf_min_rank` | `2` | Skip a layer if its computed rank would fall below this. |
| `lrf_skip_large_kernels` | `False` | Skip Conv2d layers with kernel > 1x1: prevents latency regression on 3x3-heavy CNNs (ResNet/VGG-style). |

### Weight clustering hyperparameters

| Arg | Default | What it does |
|---|---|---|
| `num_clusters` | `16` | k for k-means weight clustering. |
| `cluster_fine_tune_epochs` | `5` | Recovery fine-tune epochs after clustering. |
| `cluster_fine_tune_lr` | `1e-5` | Learning rate for that recovery fine-tune. |

### Knowledge distillation fine-tuning hyperparameters

The final recovery step, run after quantization/GPTQ so it can recover
accuracy lost from every prior step at once.

| Arg | Default | What it does |
|---|---|---|
| `kd_epochs` | `3` | Fine-tune epochs. |
| `kd_lr` | `1e-5` | Learning rate. |
| `kd_temperature` | `4.0` | Softmax temperature for the teacher's soft labels. |
| `kd_alpha` | `0.7` | Weight on hard-label loss; `1 - kd_alpha` goes to the distillation loss. |

### Quantization hyperparameters

| Arg | Default | What it does |
|---|---|---|
| `quant_mode` | `'fp16'` | `'fp16'`, `'dynamic'` (INT8, CPU-friendly), or `'static'` (auto-switched to `'dynamic'`). |
| `quant_cal_batches` | `100` | Calibration batches: only used by `'static'`. |

### GPTQ hyperparameters

| Arg | Default | What it does |
|---|---|---|
| `gptq_bits` | `4` | `4` for INT4 (8x compression) or `8` for INT8 (4x). |
| `gptq_cal_batches` | `16` | Batches used to estimate the Hessian from activations. |
| `gptq_block_size` | `128` | Columns processed per Hessian update block. |

### CQI scoring weights

Each is an exponent applied to that factor's ratio in the Compression
Quality Index: raise one to make the search/report weight that factor
more heavily.

| Arg | Default |
|---|---|
| `cqi_w_accuracy` | `1.0` |
| `cqi_w_size` | `1.0` |
| `cqi_w_latency` | `1.0` |
| `cqi_w_kl` | `1.0` (only meaningful when pruning ran) |

### Report

| Arg | Default | What it does |
|---|---|---|
| `save_report` | `True` | Generate the PNG compression report. |
| `report_path` | `'compression_report.png'` | Where to save it. |