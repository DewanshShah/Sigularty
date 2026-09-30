# Using sigularty

This is a quick reference for calling sigularty as a library: how to
import it, the two ways to get a model in, and what every `compress()`
argument does. For the theory behind each technique, why pruning runs
before LRF, what CQI means, why quantization runs before the final
recovery step, see [README.md].

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
    test_loader=r.test_loader,
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

result = compress(model, train_loader, test_loader=test_loader, num_classes=num_classes)
compressed_model = result.model
```

`model` can be any `nn.Module`. `dataloader` (positional, second argument)
is used for calibration and fine-tuning throughout the pipeline: it's
never optional. `test_loader` is a genuine held-out set used for every
accuracy measurement — baseline, both hyperparameter searches, every
technique's gate, the final report. If you omit it, `compress()`
auto-splits `dataloader`'s dataset 70/30 and prints a warning when it
does; passing your own is always preferable, since the auto-split both
reduces how much data your model actually gets fine-tuned on and can't
guarantee class balance in the held-out portion. The original model is
never modified; `compress()` always works on a deep copy.

---

## Other entry points

```python
from sigularty import (
    compress,               # run the full compression pipeline
    analyze,                # inspect a model, get technique recommendations
    load_from_registry,     # pull a registry model + its dataset together
    finetune,                # fine-tune a model before compressing it
    find_best_lr,            # LR range test: run before finetune()/pretrain_*
    export_to_onnx,          # export the compressed model to ONNX
    run_onnx_inference,      # run + benchmark that .onnx file on ONNX Runtime
    plot_compression_report,
    plot_pruning_report,
    plot_epsilon_landscape,
)
```

| Function | Signature | Returns |
|---|---|---|
| `analyze(model, dataloader=None, *, device=None)` | Inspects a model without modifying it. Pass a held-out `dataloader` if you want the reported accuracy to reflect generalisation rather than whatever the model has already been trained on. | `AnalysisResult` |
| `load_from_registry(model_name, *, device=None, train_sample=None, test_sample=500, batch_size=32, model_path=None, force_retrain=False, pretrain_epochs=10, pretrain_lr=1e-4)` | See list of registry model names below. Sole supported registry entry point. | `RegistryResult` |
| `finetune(model, train_loader, *, test_loader=None, num_classes=10, epochs=10, lr=1e-3, max_batches=0, device=None, save_path=None)` | Fine-tunes in place and returns the same object. `test_loader` defaults to `train_loader` if omitted — pass a genuine held-out loader if you want the per-epoch `test_acc` it prints to mean anything. | `nn.Module` |
| `find_best_lr(model, dataloader, *, device=None, num_classes=10, start_lr=1e-7, end_lr=10.0, num_steps=100)` | LR range test; run before `finetune()` or `pretrain_epochs > 0`. | `float` |
| `export_to_onnx(model, save_path, input_shape=(1,3,224,224), device='cpu', opset_version=17, dynamic_batch=True)` | Exports `model` (typically `result.model`) to a verified `.onnx` file. See [Exporting to ONNX](#exporting-to-onnx) below. | `str` (absolute path to the `.onnx` file) |
| `run_onnx_inference(onnx_path, dataloader, input_shape=(1,3,224,224), num_latency_iterations=100, warmup=10)` | Loads that `.onnx` file into an ONNX Runtime session and measures real accuracy + latency on it. | `dict` |

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
result.cqi                     # Compression Quality Index, >1.0 = better tradeoff than the original; can be negative — see "CQI scoring" below
result.techniques_applied      # list[str], only techniques that actually ran/survived their accuracy gate — see "Accuracy-drop gating" below
result.report_path             # path to the saved PNG report, or None
result.pruning_report          # dict of per-layer pruning detail, or None (also None if pruning ran but was reverted by the gate)
```

Every accuracy value on this object — `original_accuracy`,
`compressed_accuracy`, and everything derived from them
(`accuracy_retention`, `cqi`'s accuracy factor, every gate's keep/revert
decision along the way) — is measured against `test_loader`, never
against `dataloader`.

`analyze()` returns an `AnalysisResult`: `size_mb`, `num_parameters`,
`architecture_type` (`'cnn'` / `'transformer'` / `'hybrid'` / `'unknown'`),
`recommended_techniques`, `per_layer_signals` (a `LayerSignal` per eligible
layer: `name`, `layer_type`, `lrf_epsilon`, `prunable`, `size_kb`), and
`accuracy` if you passed a dataloader.

`load_from_registry()` returns a `RegistryResult`: `model`, `train_loader`,
`test_loader`, `num_classes`, `model_name`, `dataset_name`.

---

## Exporting to ONNX

Once you have a compressed model, export it to ONNX to check it actually
runs cleanly outside PyTorch and to benchmark it on the runtime most
deployment targets (mobile, edge, serving) actually use.

```python
from sigularty import compress, export_to_onnx, run_onnx_inference

result = compress(model, train_loader, test_loader=test_loader, num_classes=num_classes)

onnx_path = export_to_onnx(
    result.model,
    save_path='compressed_model.onnx',
    input_shape=(1, 3, 224, 224),   # use the registry's meta['input_shape'] for a registry model
)

# Run it on ONNX Runtime and measure real accuracy + latency on that engine
stats = run_onnx_inference(onnx_path, test_loader)
print(stats)
# {'accuracy_pct': 88.4, 'mean_latency_ms': 3.21, 'median_latency_ms': 3.05,
#  'p95_latency_ms': 3.9, 'p99_latency_ms': 4.6, 'onnx_path': '/abs/path/compressed_model.onnx'}
```

`export_to_onnx()` moves the model to `device` (`'cpu'` is recommended —
CUDA exports need a CUDA-capable export machine, and most ONNX Runtime
deployment targets are CPU anyway), traces it with a single dummy input of
`input_shape`, and runs `onnx.checker.check_model()` on the result before
returning the absolute path. Requires `pip install onnx`.

**Dynamic INT8 (`quant_mode='dynamic'`) does not export.** Its `qint8`
custom ops aren't traceable by `torch.onnx.export`, and this is a real,
raised `RuntimeError` with a remediation message, not a silently-broken
file. If you need an ONNX export, use `quant_mode='fp16'` or
`quant_mode='static'` instead (both export cleanly), or export the model
*before* dynamic quantization would have run.

**`input_shape` must match what the model actually expects.** For a
registry NLP model this is a token-ID shape like `(1, 128)`, not the
vision default `(1, 3, 224, 224)` — pull it from
`meta['input_shape']`/`meta['input_dtype']` (see `get_model_meta()` in
`model_registry.py`) rather than guessing.

`run_onnx_inference()` loads the exported file into an
`onnxruntime.InferenceSession` (CPU execution provider), measures top-1
accuracy over the full `dataloader` you pass it — use your held-out
`test_loader` for a real, generalization-reflecting number — then times
`num_latency_iterations` dummy forward passes (after `warmup` untimed
passes). Requires `pip install onnxruntime`. Raises `FileNotFoundError` if
`onnx_path` doesn't exist, and `ImportError` if `onnxruntime` isn't
installed.

---

## `compress()` hyperparameters

Every argument after `model` and `dataloader` is keyword-only. Grouped
the same way the function itself groups them.

### Evaluation data

| Arg | Default | What it does |
|---|---|---|
| `test_loader` | `None` | Genuine held-out set. Every accuracy measurement in `compress()` — baseline, both hyperparameter searches, every technique's gate, the final report — reads from `test_loader`, never from `dataloader`. If omitted, `compress()` auto-splits `dataloader`'s dataset 70/30 (fixed seed, plain random) and reuses that one split for the entire call, printing a warning each time it happens. An extra warning prints if the resulting held-out split comes out under 100 samples. Passing your own is strongly recommended over relying on the auto-split — see the Quickstart note above. |

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
`load_from_registry()` with no existing checkpoint). This step always
runs before `test_loader` is resolved via auto-split — pretrain trains on
`dataloader`'s 70% share, never on the held-out portion.

| Arg | Default | What it does |
|---|---|---|
| `pretrain_epochs` | `0` | Epochs to fine-tune before compressing. `0` skips this entirely. |
| `pretrain_lr` | `1e-3` | Learning rate for that pre-training. |
| `pretrain_test_loader` | `None` | Validation loader for pre-training. Falls back to `test_loader`. |

### Hyperparameter search (optional, adds evaluations before the pipeline runs)

| Arg | Default | What it does |
|---|---|---|
| `find_optimal_epsilon` | `False` | Auto-search for the best LRF epsilon instead of using `lrf_epsilon` as-is. If the best candidate scores below `1.0` CQI (worse than not compressing), or every candidate exceeds `accuracy_drop_threshold`, LRF is disabled for the run rather than falling back to `lrf_epsilon`. |
| `find_optimal_pruning` | `False` | Auto-search for the best pruning ratio instead of using `pruning_ratio` as-is. Same rule: no viable configuration disables pruning for the run. |
| `epsilon_search_trials` | `15` | Evaluation budget for the epsilon search. |
| `pruning_search_trials` | `16` | Evaluation budget for the pruning search. |
| `pruning_search_ft_epochs` | `1` | Fine-tune epochs per trial during the pruning search (kept low for speed: the real run uses `pruning_fine_tune_epochs`). |
| `pruning_search_ft_lr` | `1e-4` | Fine-tune learning rate per trial during the pruning search. |
| `accuracy_drop_threshold` | `5.0` | Max acceptable accuracy drop in percentage points. Used three ways: both searches' final selection, the pipeline's per-technique revert gate, and the budget the CQI accuracy factor is normalized against (see "CQI scoring" below). |
| `early_abort_threshold` | `None` | Direct pp value (not a multiplier on `accuracy_drop_threshold`). After epoch 1 of Pruning's or LRF's own recovery fine-tune — in both the real pipeline and their searches — abort the remaining epochs if the drop vs. the original baseline already exceeds this. `None` (default) = disabled; every fine-tune always runs to completion. |
| `epsilon_cache_path` | `None` | Override the auto-derived per-model cache file for the epsilon search. `None` = `.sigularty_cache/epsilon_<model>.json`. |
| `pruning_cache_path` | `None` | Override the auto-derived per-model cache file for the pruning search. `None` = `.sigularty_cache/pruning_<model>.json`. |

**Accuracy-drop gating is always active.** Structured Pruning, Low-Rank
Factorization, Weight Clustering, GPTQ, standard Quantization, and the
final KD recovery fine-tune are each measured before/after and reverted
to their pre-technique state if that ONE technique's own marginal drop
exceeds `accuracy_drop_threshold`. Each technique gets an independent
budget — an earlier costly technique does not eat into a later
technique's allowance. A reverted technique will not appear in
`result.techniques_applied` — check the console output for a `❌
[TechniqueName] SKIPPED — accuracy dropped ...` line if a technique you
enabled seems to be missing.

**GPTQ uses a QAT-aware (straight-through estimator) path whenever
`use_kd_finetune=True`**, so the final KD fine-tune can actually update
GPTQ-quantized weights instead of finding them frozen — collapsed back to
compact storage automatically once KD finishes. If the final KD step's
cumulative recovery still falls short of `accuracy_drop_threshold` after
its own gate passes, one bounded, LR-adjusted retry runs automatically
before the result is accepted as-is. KD always runs after GPTQ and
standard quantization, never before, so it recovers accuracy lost to
every prior step at once, including quantization/GPTQ's own damage. fp16
is not skipped when GPTQ is absent or reverted — fp16-only stays a fully
independent, always-available configuration.

**Search result caching.** Each search caches trial results to disk
purely by hyperparameter value, with no reference to which model produced
them. `epsilon_cache_path`/`pruning_cache_path` default to a filename
derived from the model's class name + parameter count + `num_classes`, so
different models get separate cache files automatically. Pass an
explicit path yourself for a stronger guarantee (e.g. distinct caches per
dataset too, not just per model/class-count).

Cached scores don't record which CQI weights, CQI accuracy formula, or
latency settings produced them. Delete the cache files
(`.sigularty_cache/` by default) after changing any `cqi_w_*` argument,
after changing `accuracy_drop_threshold` (it now shapes every cached
score, not just the final selection), or after upgrading from a release
whose CQI used the plain accuracy ratio — those cached scores sit on a
different scale and would be mixed with current ones under the same keys.

If you have `.sigularty_cache/` files from before `test_loader` existed,
delete them. Those cached accuracy numbers were measured against
`dataloader` (effectively training data, since `compress()` had no other
option at the time) rather than a genuine held-out set — reusing them now
would silently mix pre-fix and post-fix numbers under the same cache
keys, and the cached values would understate real accuracy drop.

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
| `pruning_residual_max_ratio` | `None` | Ceiling specifically for auto-detected residual/skip-connection-coupled groups — the layers whose channel count IS the residual stream for an entire network stage, so collapsing them damages every downstream block in that stage, not just one layer's worth of capacity. `None` (default) falls back to `pruning_max_ratio` above. See README.md's "Architecture-Agnostic Residual Group Detection" section for the detection mechanism. |
| `pruning_model_type` | `'classifier'` | Same role as `model_type`, specific to pruning's clamp. |
| `pruning_fine_tune_epochs` | `3` | KD recovery epochs after pruning. |
| `pruning_fine_tune_lr` | `1e-4` | Learning rate for that recovery fine-tune. |
| `pruning_cal_batches` | `50` | Calibration batches for activation-statistics importance scoring. |
| `pruning_iterative_steps` | `1` | Prune in one shot (`1`) or multiple incremental rounds (more stable, slower). |
| `pruning_isomorphic` | `False` | Force identical pruning structure across coupled dependency groups. |
| `pruning_round_to` | `None` | Round pruned channel counts to a multiple of this (e.g. `8`/`16` for Tensor Core alignment). |

**Note:** `find_optimal_pruning=True` forwards `pruning_max_ratio` (and
`pruning_residual_max_ratio`) into the search itself, and reads the
search's winning max-ratio back out afterward — the config the search
recommends is the config the real pipeline actually applies.

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

The final recovery step. KD always runs after quantization/GPTQ, never
before, so it recovers accuracy lost from every prior step at once —
including quantization/GPTQ's own damage, since those are gated too (see
"Accuracy-drop gating" above). If GPTQ is enabled, it uses a QAT-aware
path specifically so this step can update GPTQ-quantized weights rather
than finding them frozen.

| Arg | Default | What it does |
|---|---|---|
| `kd_epochs` | `3` | Fine-tune epochs. |
| `kd_lr` | `1e-5` | Learning rate. |
| `kd_temperature` | `4.0` | Softmax temperature for the teacher's soft labels. |
| `kd_alpha` | `0.7` | Weight on hard-label loss; `1 - kd_alpha` goes to the distillation loss. |
| `kd_max_batches` | `50` | Max batches per KD epoch. Use `0` for the full dataloader each epoch. |

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

### CQI scoring

The Compression Quality Index combines accuracy, size, latency, and
(for pruning) output-distribution drift into one number. Both
hyperparameter searches rank every trial by it, and it's the `cqi`
value on `CompressionResult` and in the report.

```
CQI = accuracy_factor
    × (baseline_size / size)^cqi_w_size
    × (baseline_latency / latency)^cqi_w_latency
    × (1 / (1 + KL))^cqi_w_kl                       [pruning only]
```

**Reading it:** `1.0` means no better than the original model; `2.5`
means the combined accuracy/size/speed tradeoff is 2.5x better.

**The accuracy factor** is a function of the accuracy drop measured
against `accuracy_drop_threshold`. Let `x = drop / accuracy_drop_threshold`,
where `drop = original_accuracy - compressed_accuracy` in percentage
points. `x = 1` means the drop exactly equals the budget, the same point
at which the pipeline's own gate would revert a technique.

| Region | Factor | Behavior |
|---|---|---|
| `x < 0` (accuracy improved) | `1 + (2 - 1) · t / (1 + t)`, `t = -x` | Saturating reward, capped at `2.0`. Improving accuracy earns a bounded bonus, never an unlimited one. |
| `0 ≤ x < 1` (within budget) | `1 - 0.1 · x` | Gentle straight-line decline. Using the entire budget costs at most `0.1`, so a small drop barely changes the ranking. |
| `x ≥ 1` (at or past budget) | `1.9 - exp(25 · (x - 1))` | Continuous with the region above at `x = 1`, then falls away unboundedly and steeply. |

The three pieces join continuously, so there are no gaps or jumps in the
score. Worked example at `accuracy_drop_threshold=10`:

| Accuracy drop | Factor |
|---|---|
| -11 pp (improved) | ≈ 1.524 |
| -1 pp | ≈ 1.091 |
| 0 pp | 1.000 |
| 1 pp | 0.990 |
| 5 pp | 0.950 |
| 10 pp | 0.900 |
| 11 pp | ≈ -10.28 |

The point of the last piece is that a size or latency ratio can't buy
back an accuracy drop at or past your budget: the factor falls faster
than any compression ratio can grow. Because of that, **CQI can be
negative**, and a negative value is meaningful: the accuracy loss alone
already makes that configuration worse than doing nothing, regardless of
how small or fast the model got. The searches treat any winner scoring
below `1.0` as non-viable and disable that technique for the run.

`accuracy_drop_threshold` is what drives all of this. `compress()`
always passes it through, so every trial score inside both searches and
the final `result.cqi` use this accuracy factor. Its shape constants
(the `2.0` reward ceiling, `0.1` in-budget slope, and `25` post-budget
steepness) are fixed defaults and not exposed as `compress()` arguments;
the one knob you control is the threshold itself.

The size, latency, and KL factors are plain ratios raised to their
weight. Both searches measure latency on every trial, so `cqi_w_latency`
affects which epsilon or pruning config wins, not just the final report.
A technique that shrinks the model but makes it slower is scored
accordingly; raise `cqi_w_latency` to make the searches stricter about
speed. Latency is a ratio like size, so a large size reduction can still
outweigh a moderate slowdown. The per-technique accuracy gate is
unaffected by these weights: it only checks accuracy, and no technique is
reverted for being slower.

| Arg | Default | What it does |
|---|---|---|
| `cqi_w_accuracy` | `1.0` | No effect inside `compress()`. Since `accuracy_drop_threshold` is always passed, CQI uses the accuracy factor above instead of the plain `(accuracy / baseline_accuracy)` ratio this weight used to scale. Kept so existing calls don't break. |
| `cqi_w_size` | `1.0` | Exponent on the size ratio. Raise to favor smaller models. |
| `cqi_w_latency` | `1.0` | Exponent on the latency ratio. Raise to favor faster models. |
| `cqi_w_kl` | `1.0` | Exponent on the KL-divergence penalty. Only meaningful when pruning ran. |

### Report

| Arg | Default | What it does |
|---|---|---|
| `save_report` | `True` | Generate the PNG compression report. |
| `report_path` | `'compression_report.png'` | Where to save it. |