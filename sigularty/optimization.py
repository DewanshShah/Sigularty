"""
optimization.py
===============
Hyperparameter search algorithms for model compression techniques.

Public API
----------
  find_optimal_epsilon_smart(model, test_loader, device, ..., num_trials)
      -> (float | None, dict, list)
  find_optimal_pruning_params(model, dataloader, test_loader, device, ..., num_trials)
      -> (dict | None, dict, list)
  compression_quality_index(accuracy, size_mb, baseline_accuracy, baseline_size,
                             latency_ms=None, baseline_latency_ms=None,
                             kl_divergence=None,
                             w_accuracy=1.0, w_size=1.0, w_latency=1.0, w_kl=1.0)
      -> float   [public; used everywhere: search, visualization, compression report]

Compression Quality Index (CQI)
---------------------------------
  CQI is the single scoring metric used by all search algorithms and the report.

  CQI = (acc / baseline_acc)^w_acc
      × (baseline_size / size_mb)^w_size
      × (baseline_lat / lat)^w_lat          [when latency supplied]
      × (1 / (1 + kl_divergence))^w_kl      [when KL supplied; pruning only]

  Each factor is raised to its weight (exponent form).  Default w=1 recovers
  the original unweighted formula.  Increasing w_latency=3 makes the search
  strongly prefer models with lower latency over models with better accuracy
  or size, because (lat_ratio)^3 penalises latency regressions cubically.

    CQI = 1.0  → no improvement over original model (all factors = 1.0).
    CQI = 3.5  → 3.5× better on the combined weighted accuracy×compression×speedup.

  When latency or KL values are None/zero, those factors are omitted (weight has
  no effect on missing factors) so callers without latency still work unchanged.

  This metric is used everywhere: epsilon search, pruning search, and the
  compression report visualization.

Proxy Model Infrastructure
---------------------------
  Both search functions can optionally run on a dynamic INT8 proxy model.
  The proxy is a deep copy of the original model with all nn.Linear weights
  quantized to INT8, always running on CPU.  The goal is faster per-trial
  inference, not a structurally different model.

  Proxy correlation is validated before the search begins:
    1. Two anchor configurations are evaluated on BOTH the proxy and the
       real float32 model.
    2. If their efficiency score rank orders agree → search proceeds on proxy.
    3. If rank orders differ → warning is printed and search falls back to
       the real float32 model.  No results from a mis-correlated proxy are used.

  After any proxy search, the winning configuration is re-evaluated once on
  the real float32 model.  Both proxy_score and real_score are stored in every
  returned results dict.

  Limitations of the proxy:
    - Proxy latency is CPU latency.  The ratio baseline/compressed should
      still correlate with GPU latency ratios but absolute values differ.
    - Very small models or unusual activation distributions may show poor
      proxy correlation.  The fallback ensures correctness regardless.
    - Pruning requires modifying model structure (Torch-Pruning), which must
      run on float32 models.  For pruning search, the proxy is used for the
      correlation check only; search evaluations always run on the real model.
      The speedup benefit from the proxy is therefore limited to inference
      time within each evaluation (not the pruning step itself).

num_trials Budget Allocation
-----------------------------
  Both search functions accept num_trials which controls total evaluations.
  No phase size is hardcoded — all budgets are derived proportionally:

  Epsilon search:
    proxy_budget    = 2  (deducted when proxy available)
    search_budget   = num_trials - proxy_budget
    n_anchors       = min(5, search_budget)
    n_ternary_iters = max(0, search_budget - n_anchors) // 2
    NOTE: num_trials controls the NUMBER of trials only. Each trial's OWN
    cost now also depends on search_ft_epochs (see "Per-Trial Fine-Tuning"
    below) — 0 (default off) reproduces the original near-instant-per-trial
    behaviour; >0 multiplies every trial's cost by roughly that many real
    training epochs.

  Pruning search:
    proxy_budget    = 2  (deducted when proxy available)
    search_budget   = num_trials - proxy_budget
    ratio_budget    = max(5, int(0.60 * search_budget))
    grid_budget     = max(2, int(0.35 * search_budget))
    iter_budget     = 1
    leftover added to grid_budget

Per-Trial Fine-Tuning & Early-Abort Multiplier
-------------------------------------------------
  Pruning search has always fine-tuned every trial (via
  apply_structured_pruning's fine_tune_epochs/fine_tune_lr). Epsilon search
  did NOT — _evaluate_epsilon applied LRF and measured accuracy immediately,
  by design, to keep trials fast. search_ft_epochs (find_optimal_epsilon_
  smart) makes epsilon-search fine-tuning OPT-IN: 0 (default) preserves the
  original fast behaviour; >0 routes each trial through
  apply_low_rank_factorization's own existing fine-tune mechanism, the same
  one the main pipeline's LRF step already uses. This is a genuine,
  multiplicative cost increase for epsilon search specifically — see
  find_optimal_epsilon_smart's docstring for the exact tradeoff.

  Both search functions accept early_abort_threshold: a direct percentage-
  point value, not a multiplier. After epoch 1 of ANY trial's fine-tune, if
  that epoch's accuracy drop versus baseline_accuracy already exceeds
  early_abort_threshold (e.g. early_abort_threshold=30.0 means "exceeds
  30pp"), the remaining epochs of THAT trial are skipped — recovering that
  much ground in the epochs left over is judged implausible, so finishing
  the fine-tune would only spend compute on a result the trial's own
  scoring would rank poorly anyway. None (default) disables this — every
  trial's fine-tune always runs to completion, exactly as before. This is
  deliberately independent of accuracy_drop_threshold — changing one does
  not silently move the other.

  This is intentionally a DIFFERENT mechanism from the Structural-Failure
  Abort Protocol below — early-abort fires on a degraded-but-VALID accuracy
  number (the model still produces real, if bad, predictions), is purely a
  compute-saving heuristic the caller opts into, and does NOT count toward
  or interact with the consecutive-structural-failure counter. A trial that
  early-aborts is still a normal, cacheable, fully-eligible result for that
  trial's own ranking — it just got there having run fewer epochs.
  early_abort_threshold is forwarded through the same shared choke points
  every phase already calls through (_run_eval for pruning, both the anchor
  loop and _eps_score_fn for epsilon), so it applies identically everywhere
  fine-tuning happens during either search.

Structural-Failure Abort Protocol
-----------------------------------
  measure_accuracy() (helper_functions.py) raises a RuntimeError containing
  the substring "No samples were evaluated" when EVERY batch in an evaluation
  fails (e.g. a Sequential-wrapped layer breaking a parent module's direct
  .weight/.bias access — see compression.py's nn.MultiheadAttention guard).
  This is a hard structural break, not an ordinary bad-but-valid result.

  Both search functions track CONSECUTIVE structural failures across their
  ENTIRE run (all phases share one counter — it does not reset between
  anchor sampling, ternary refinement, grid search, etc.).  Any successful
  trial resets the counter to zero.  On the SECOND consecutive structural
  failure, the search aborts immediately: no further trials are attempted,
  and whatever was cached before the abort is fed straight into the
  function's final-selection logic (Phase 3 for epsilon, Phase 4 for
  pruning), which always operates by pooling the entire cache rather than
  assuming any particular phase completed.  A single isolated failure does
  NOT abort anything — it is skipped and the search moves to the next trial.

Design rules
------------
  * Search functions NEVER plot — they return data; visualization.py renders it.
  * Input model is NEVER mutated.  All operations deepcopy first.
  * Results are cached to disk after every evaluation (crash recovery).
  * Every public function returns (optimal, all_results_dict, history_list).
  * num_trials controls total evaluations in each search.  No hardcoded sizes.
  * No global state.
"""

import copy
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from sigularty.compression import _manual_dynamic_quantize, apply_low_rank_factorization, apply_structured_pruning

from sigularty.helper_functions import (
    get_model_size_mb,
    measure_accuracy,
    measure_latency,
)


# ============================================================================
# ── CACHE UTILITIES ──────────────────────────────────────────────────────────
# ============================================================================

def _load_cache(cache_path: str) -> dict:
    """
    Load evaluation cache from disk.
    Returns empty dict on any error so search always starts without crashing.
    """
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict, cache_path: str) -> None:
    """
    Persist evaluation cache to disk.
    Silently swallows I/O errors — in-memory cache is always the source of truth.
    """
    if not cache_path:
        return
    try:
        parent = os.path.dirname(cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


# ============================================================================
# ── SELECTION HELPER — accuracy-tiered "best by score" ──────────────────────
# Shared by every phase of find_optimal_pruning_params() so refinement,
# grid search, and the iterative-steps comparison all respect the accuracy
# drop threshold the same way Phase 1's anchor selection already did,
# instead of only gating at the very end.
# ============================================================================

def _select_best_by_tier(
    entries: List[dict],
    baseline_accuracy: float,
    accuracy_drop_threshold: float,
) -> Optional[dict]:
    """
    Pick the highest-scoring entry, preferring ones within the accuracy-drop
    threshold over ones outside it.

    Tier 1 (strict):  accuracy drop <= accuracy_drop_threshold  → max score.
    Tier 2 (relaxed):  accuracy drop <= accuracy_drop_threshold + 10.0 → max score.
    Tier 3 (score-only): no accuracy gating at all → max score.

    Falls through tiers only when the stricter tier is empty.  Returns None
    only when `entries` itself is empty.
    """
    if not entries:
        return None
    strict = [e for e in entries if (baseline_accuracy - e.get('accuracy', 0)) <= accuracy_drop_threshold]
    if strict:
        return max(strict, key=lambda e: e['score'])
    relaxed = [e for e in entries if (baseline_accuracy - e.get('accuracy', 0)) <= accuracy_drop_threshold + 10.0]
    if relaxed:
        return max(relaxed, key=lambda e: e['score'])
    return max(entries, key=lambda e: e['score'])


# ============================================================================
# ── COMPRESSION QUALITY INDEX (CQI) ──────────────────────────────────────────
# Single shared metric for all compression search algorithms and the
# compression report visualization.  Exposed as public API.
#
# Weighted exponent form:
#   CQI = acc_ratio^w_acc × size_ratio^w_size × lat_ratio^w_lat × kl_factor^w_kl
#
# w=1 (default) reproduces the original formula.  Increasing a weight raises
# that factor to a higher power, making its influence superlinear:
#   acc_ratio=0.9, w=1 → 0.9     (mild penalty for accuracy loss)
#   acc_ratio=0.9, w=3 → 0.729   (strong penalty — search avoids accuracy drops)
#   acc_ratio=0.9, w=5 → 0.590   (dominant — search treats accuracy as primary)
# ============================================================================

def compression_quality_index(
    accuracy: float,
    size_mb: float,
    baseline_accuracy: float,
    baseline_size: float,
    latency_ms: Optional[float] = None,
    baseline_latency_ms: Optional[float] = None,
    kl_divergence: Optional[float] = None,
    w_accuracy: float = 1.0,
    w_size: float = 1.0,
    w_latency: float = 1.0,
    w_kl: float = 1.0,
) -> float:
    """
    Compression Quality Index (CQI) — weighted multi-factor compression score.

    CQI = (acc / baseline_acc)^w_accuracy
        × (baseline_size / size_mb)^w_size
        × (baseline_lat / lat)^w_latency    [when both latency values supplied]
        × (1 / (1 + KL))^w_kl              [when kl_divergence supplied]

    Each factor is a ratio ≥ 0 raised to its weight (exponent form).
    Default weights of 1.0 reproduce the original unweighted formula.
    Higher weights amplify that factor's influence on ranking:
      w_latency=3 → latency ratio cubed → search strongly favours faster models.
      w_accuracy=2, w_size=1 → accuracy twice as important as compression ratio.

    Missing factors (None inputs) are omitted entirely — their weight has no
    effect and the CQI falls back to the remaining factors only.

    Args:
        accuracy:             Compressed model top-1 accuracy (%).
        size_mb:              Compressed model size in MB.
        baseline_accuracy:    Original model accuracy (%).
        baseline_size:        Original model size in MB.
        latency_ms:           Compressed model mean latency ms, or None.
        baseline_latency_ms:  Original model mean latency ms, or None.
        kl_divergence:        KL divergence from behavioral probe, or None.
                              Lower is better (0 = identical outputs).
        w_accuracy:           Accuracy factor weight (default 1.0).
        w_size:               Size factor weight (default 1.0).
        w_latency:            Latency factor weight (default 1.0).
        w_kl:                 KL divergence penalty weight (default 1.0).

    Returns:
        CQI ≥ 0.0.  CQI > 1.0 means the compressed model beats the baseline
        on the combined weighted score.  Returns 0.0 on degenerate inputs.
    """
    if baseline_accuracy <= 0.0 or size_mb <= 0.0:
        return 0.0

    acc_ratio  = accuracy      / baseline_accuracy
    size_ratio = baseline_size / size_mb

    # Apply weights as exponents (w=1 → identity, w>1 → superlinear influence)
    cqi = (acc_ratio ** w_accuracy) * (size_ratio ** w_size)

    if (
        latency_ms is not None and baseline_latency_ms is not None
        and latency_ms > 0.0 and baseline_latency_ms > 0.0
    ):
        lat_ratio = baseline_latency_ms / latency_ms
        cqi *= lat_ratio ** w_latency

    if kl_divergence is not None and kl_divergence >= 0.0:
        kl_factor = 1.0 / (1.0 + kl_divergence)
        cqi *= kl_factor ** w_kl

    return cqi


# Backward-compatible alias — internal callers use this name.
# New external code should call compression_quality_index() directly.
_efficiency_score = compression_quality_index


# ============================================================================
# ── PROXY MODEL INFRASTRUCTURE ───────────────────────────────────────────────
# ============================================================================

def _build_proxy_model(model: nn.Module) -> Tuple[nn.Module, str]:
    """
    Build an INT8 proxy of `model` for faster search evaluations.

    Deep-copies the model, applies dynamic INT8 quantization to all nn.Linear
    layers, and returns the result together with its required device ('cpu').
    The proxy always runs on CPU — INT8 quantized ops have no CUDA kernel.
    Conv2d layers remain float32 (dynamic quantization only targets Linear).

    Args:
        model: Original float32 model.  Never modified.

    Returns:
        (proxy_model, 'cpu')
    """

    proxy = _manual_dynamic_quantize(copy.deepcopy(model))
    proxy.eval()
    try:
        proxy.cpu()
    except Exception:
        pass
    return proxy, 'cpu'


def _validate_proxy_correlation(
    real_model: nn.Module,
    proxy_model: nn.Module,
    proxy_device: str,
    real_device: str,
    correlation_eval_fns: List[Callable],
) -> bool:
    """
    Validate that proxy efficiency score rank order matches the real model.

    With 2 anchor configurations, rank order (which config scores higher) is
    a clear binary check.  Pearson correlation requires more points and is
    unstable at this sample size.

    Each callable in `correlation_eval_fns` accepts (model, device) and
    returns a dict with at least a 'score' key.  They are called once on
    the proxy and once on the real model.

    Returns True if rank orders match (proxy is trustworthy).
    Returns False if ranks differ or any evaluation raises — falls back
    to real model.
    """
    if len(correlation_eval_fns) < 2:
        return True   # cannot check with fewer than 2 points — assume OK

    proxy_scores: List[float] = []
    real_scores:  List[float] = []

    for eval_fn in correlation_eval_fns:
        try:
            p = eval_fn(proxy_model, proxy_device)
            r = eval_fn(real_model,  real_device)
            proxy_scores.append(p.get('score', 0.0))
            real_scores.append(r.get('score',  0.0))
        except Exception as exc:
            print(f"  [Proxy] Correlation eval failed: {exc} — assuming no correlation.")
            return False

    # Compare rank orders: which index is highest?
    proxy_winner = proxy_scores.index(max(proxy_scores))
    real_winner  = real_scores.index( max(real_scores))

    if proxy_winner == real_winner:
        print(f"  [Proxy] Correlation PASSED — rank orders match.")
        print(f"    proxy scores: {[f'{s:.3f}' for s in proxy_scores]}")
        print(f"    real  scores: {[f'{s:.3f}' for s in real_scores]}")
        return True
    else:
        print(f"  [Proxy] Correlation FAILED — rank orders differ.  Falling back to real model.")
        print(f"    proxy scores: {[f'{s:.3f}' for s in proxy_scores]}")
        print(f"    real  scores: {[f'{s:.3f}' for s in real_scores]}")
        return False


# ============================================================================
# ── TERNARY SEARCH (shared kernel) ───────────────────────────────────────────
# Used by epsilon search and pruning ratio search.
# ============================================================================

def _ternary_search(
    score_fn: Callable[[float], Tuple[Optional[float], Optional[dict]]],
    low: float,
    high: float,
    tolerance: float,
    max_iterations: int,
    phase_label: str = 'ternary',
) -> Tuple[float, List[dict]]:
    """
    Ternary search for the scalar parameter that maximises score in [low, high].

    Assumes approximately unimodal landscape.  Each iteration evaluates the
    1/3 and 2/3 interior points and eliminates one third of the interval.
    Convergence: O(log₃(1/tolerance)) iterations.

    Abort sentinel: if score_fn returns (None, None) for either evaluation
    point, the search stops immediately (caller's score_fn has decided this
    run can no longer continue — e.g. two consecutive structural failures).
    Whatever history was collected before the sentinel is still returned.

    Args:
        score_fn:       Callable(value) -> (score, result_dict), or
                        (None, None) to signal "stop searching now".
        low, high:      Search interval.
        tolerance:      Stop when (high - low) < tolerance.
        max_iterations: Hard cap on evaluation pairs.
        phase_label:    Written into history entries.

    Returns:
        (refined_midpoint, history_list)
    """
    history: List[dict] = []

    for iteration in range(max_iterations):
        if (high - low) < tolerance:
            break

        m1 = low  + (high - low) / 3
        m2 = high - (high - low) / 3

        s1, r1 = score_fn(m1)
        if s1 is None:
            break
        s2, r2 = score_fn(m2)
        if s2 is None:
            break

        for val, score_val, r in [(m1, s1, r1), (m2, s2, r2)]:
            entry = {'phase': phase_label, 'iteration': iteration, 'score': score_val}
            entry.update(r)
            history.append(entry)
            # Compact print: show whatever the key parameter is
            key_name = 'epsilon' if 'epsilon' in r else 'pruning_ratio'
            print(f"    iter {iteration + 1}  {key_name}={r.get(key_name, val):.4f}  "
                  f"acc={r.get('accuracy', 0):.1f}%  score={score_val:.3f}")


        if s1 < s2:
            low = m1
        else:
            high = m2

    return (low + high) / 2, history


# ============================================================================
# ── EPSILON EVALUATION (LRF) ─────────────────────────────────────────────────
# ============================================================================

def _evaluate_epsilon(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    epsilon: float,
    baseline_accuracy: float,
    baseline_size: float,
    baseline_latency_ms: float,
    min_layer_size: int,
    cache: dict,
    input_shape: tuple = (1, 3, 224, 224),
    latency_iters: int = 10,
    w_accuracy: float = 1.0,
    w_size: float = 1.0,
    w_latency: float = 1.0,
    min_rank: int = 1,
    skip_large_kernels: bool = False,
    # ── Per-trial fine-tuning (opt-in — 0 epochs reproduces the old,
    #    fine-tune-free behaviour exactly) ────────────────────────────────────
    dataloader: Optional[DataLoader] = None,
    num_classes: Optional[int] = None,
    fine_tune_epochs: int = 0,
    fine_tune_lr: float = 0.0001,
    accuracy_drop_threshold: Optional[float] = None,
    early_abort_threshold: Optional[float] = None,
) -> dict:
    """
    Evaluate a single LRF epsilon value.

    Checks in-memory cache first (str(round(epsilon, 4)) key).
    On cache miss: applies LRF, OPTIONALLY fine-tunes (see below), measures
    accuracy + size + latency, computes three-factor efficiency score,
    stores in cache.

    PER-TRIAL FINE-TUNING (opt-in, fine_tune_epochs=0 by default):
      Originally this function never fine-tuned — it applied LRF and
      measured accuracy immediately, by design, to keep epsilon search fast
      (~13 forward passes saved per trial by also skipping latency
      measurement; see below). fine_tune_epochs > 0 routes through
      apply_low_rank_factorization's OWN existing fine_tune_epochs/
      fine_tune_lr mechanism (the same one the main pipeline's LRF step
      already uses) — this function does not implement a separate fine-tune
      loop, it just stops leaving those arguments at their defaults.

      Cost warning: this is a real, multiplicative cost increase, not a
      free improvement. EPSILON_SEARCH_FT_EPOCHS × num_trials extra training
      epochs are now possible where zero existed before — on a large model
      this can turn a near-instant search into one that takes as long as a
      pruning search with similar settings. The early_abort_threshold
      below blunts the WORST case (clearly-bad epsilons abort after epoch
      1) but does not eliminate the added cost for epsilons that look
      viable through a full fine-tune.

      `dataloader`/`num_classes` are required (non-None) for fine-tuning to
      actually run — exactly mirroring apply_low_rank_factorization's own
      internal check.  If fine_tune_epochs > 0 but either is None, that
      function prints a warning and skips its own fine-tune; this function
      does not duplicate that check.

      `test_loader`/`baseline_accuracy`/`accuracy_drop_threshold`/
      `early_abort_threshold` are passed straight through to the fine-tune
      so its early-abort check (see _kd_recovery_fine_tune's docstring) is
      active under the exact same opt-in rule as everywhere else it's used:
      all four must be non-None, or the check is silently skipped.

    Args:
        model:                Uncompressed base model (never modified).  Also
                              used as the fine-tune's KD teacher (via
                              apply_low_rank_factorization's own default —
                              not passed explicitly here).
        test_loader:          Evaluation DataLoader.  ALSO the early-abort
                              check's test-set source when fine-tuning runs.
        device:               Compute device.
        epsilon:              LRF rank ratio in (0.0, 1.0].
        baseline_accuracy:    Original accuracy % (for score, AND for the
                              early-abort check's drop calculation — same
                              number, since this search's "baseline" already
                              IS the absolute original model's accuracy).
        baseline_size:        Original size MB (for score).
        baseline_latency_ms:  Original latency ms (for score).
        min_layer_size:       Passed to apply_low_rank_factorization.
        cache:                In-memory cache (mutated on miss).
        input_shape:          Dummy input for latency measurement.
        latency_iters:        Timed forward passes (keep small during search).
        dataloader:           Training/calibration DataLoader for the optional
                              per-trial fine-tune.  None = fine-tuning skipped
                              regardless of fine_tune_epochs (matches
                              apply_low_rank_factorization's own contract).
        num_classes:          Output class count for the optional fine-tune.
        fine_tune_epochs:     KD recovery epochs per trial.  0 (default) =
                              old behaviour, no fine-tune, fastest.
        fine_tune_lr:         Learning rate for the optional fine-tune.
        accuracy_drop_threshold: pp threshold used for scoring/selection.
                              Independent of early_abort_threshold below —
                              only consulted if fine-tuning runs.
        early_abort_threshold: Direct pp value — "abort if epoch-1 drop
                              exceeds this many percentage points". Not a
                              multiplier on accuracy_drop_threshold. None =
                              early-abort disabled even if fine-tuning runs
                              (fine-tune still completes all epochs).

    Returns:
        Dict: epsilon, accuracy, size_mb, latency_ms, score, phase,
              proxy_score, real_score.

    Raises:
        RuntimeError: propagated from measure_accuracy() when EVERY batch in
                      this evaluation failed structurally (message contains
                      "No samples were evaluated").  This is intentionally
                      NOT swallowed here — the caller (find_optimal_epsilon_smart)
                      counts these toward the 2-consecutive-failure abort.
                      This can now also originate from the early-abort check's
                      OWN measure_accuracy() call when fine-tuning runs, since
                      that call uses the exact same function and propagation
                      rule — a mid-fine-tune structural failure is treated
                      identically to a post-fine-tune one.
    """

    cache_key = str(round(epsilon, 4))
    if cache_key in cache:
        return cache[cache_key]

    try:
        compressed = apply_low_rank_factorization(
            model,
            epsilon=epsilon,
            min_layer_size=min_layer_size,
            min_rank=min_rank,
            skip_large_kernels=skip_large_kernels,
            dataloader=dataloader,
            device=device,
            num_classes=num_classes,
            fine_tune_epochs=fine_tune_epochs,
            fine_tune_lr=fine_tune_lr,
            test_loader=test_loader,
            baseline_accuracy=baseline_accuracy,
            accuracy_drop_threshold=accuracy_drop_threshold,
            early_abort_threshold=early_abort_threshold,
        )
        size_mb  = get_model_size_mb(compressed)
        accuracy = measure_accuracy(compressed, test_loader, device)

        # ⚠️  LATENCY IS INTENTIONALLY NOT MEASURED PER-EPSILON
        # Reason: LRF (low-rank factorization) increases kernel count regardless of epsilon.
        #   Original:  W (one kernel) 
        #   LRF:       U @ V (two kernels, even for high rank)
        # Therefore, latency barely varies with epsilon (all epsilons ≈ same latency).
        # The two-factor score (accuracy × size) is sufficient for epsilon selection.
        # Actual per-epsilon latency IS measured once in the final pipeline on the 
        # winning config, where real-world speedup or regression is visible.
        # This design saves ~13 forward passes per trial (10 warmup + 10 iterations × 2 kernels).
        # NOTE: this rationale is independent of (and unaffected by) the optional
        # per-trial fine-tuning above — skipping latency saves a fixed ~13 forward
        # passes regardless of whether fine_tune_epochs is 0 or N.
        latency_ms = baseline_latency_ms   # neutral: latency factor = 1.0 in CQI
        
        score      = compression_quality_index(
            accuracy, size_mb, baseline_accuracy, baseline_size,
            # omit latency to avoid measuring 13 forward passes per trial
            w_accuracy=w_accuracy, w_size=w_size,
        )
    except RuntimeError as exc:
        if 'No samples were evaluated' in str(exc):
            raise
        print(f"  ⚠️  Failed to evaluate ε={epsilon:.4f}: {exc}")
        accuracy   = 0.0
        size_mb    = baseline_size
        latency_ms = baseline_latency_ms
        score      = 0.0
    except (TypeError, AttributeError, NameError, ImportError):
        # These are almost always programmer errors (wrong/missing kwargs,
        # typos, a missing import) rather than a legitimate "this epsilon
        # doesn't work" result. Silently scoring them as accuracy=0.0 would
        # hide a real bug behind a fake "LRF makes this model worse" search
        # conclusion — exactly what happened when apply_low_rank_factorization
        # didn't yet accept test_loader/baseline_accuracy/accuracy_drop_
        # threshold/early_abort_threshold and every trial TypeError'd here.
        # Let it propagate and crash loudly instead of laundering it into
        # experimental data.
        raise
    except Exception as exc:
        print(f"  ⚠️  Failed to evaluate ε={epsilon:.4f}: {exc}")
        accuracy   = 0.0
        size_mb    = baseline_size
        latency_ms = baseline_latency_ms
        score      = 0.0

    result = {
        'epsilon':    round(epsilon, 4),
        'accuracy':   accuracy,
        'size_mb':    size_mb,
        'latency_ms': latency_ms,
        'score':      score,
        'phase':      'uncached',
        'proxy_score': None,
        'real_score':  None,
    }
    cache[cache_key] = result
    return result


# ============================================================================
# ── PRUNING CONFIG EVALUATION ────────────────────────────────────────────────
# ============================================================================

def _evaluate_pruning_config(
    model: nn.Module,
    dataloader: DataLoader,
    test_loader: DataLoader,
    device: str,
    pruning_ratio: float,
    fine_tune_epochs: int,
    fine_tune_lr: float,
    iterative_steps: int,
    baseline_accuracy: float,
    baseline_size: float,
    baseline_latency_ms: float,
    cache: dict,
    cache_path: str,
    model_type: str = 'classifier',
    num_classes: int = 10,
    num_calibration_batches: int = 50,
    max_pruning_ratio: float = 1.0,
    residual_max_ratio: Optional[float] = None,
    round_to: Optional[int] = None,
    isomorphic: bool = False,
    input_shape: tuple = (1, 3, 224, 224),
    input_dtype: Optional[torch.dtype] = None,
    latency_iters: int = 10,
    w_accuracy: float = 1.0,
    w_size: float = 1.0,
    w_latency: float = 1.0,
    w_kl: float = 1.0,
    accuracy_drop_threshold: Optional[float] = None,
    early_abort_threshold: Optional[float] = None,
) -> dict:
    """
    Evaluate a single pruning configuration.

    Cache key = "{ratio}|{max_ratio}|{epochs}|{lr}|{iter_steps}" — encodes all
    five searched/fixed parameters so any combination is independently
    cacheable.  (max_pruning_ratio is included because Phase 2 searches it
    too; omitting it would let same-ratio/different-cap trials collide.)

    Args:
        model:                  Base model (never modified — deepcopy inside pruning).
        dataloader:             DataLoader for pruning calibration and fine-tuning.
        test_loader:            DataLoader for accuracy evaluation.  ALSO the
                                early-abort check's test-set source — see
                                accuracy_drop_threshold/early_abort_threshold.
        device:                 Compute device.
        pruning_ratio:          Global channel removal fraction.
        fine_tune_epochs:       Fine-tune epochs after pruning (0 = skip).
        fine_tune_lr:           Fine-tune learning rate.
        iterative_steps:        Torch-Pruning iterative steps.
        baseline_accuracy:      Original accuracy % (for score). ALSO the
                                early-abort check's drop reference — this
                                search's "baseline" already IS the absolute
                                original model, so no separate value needed.
        baseline_size:          Original size MB (for score).
        baseline_latency_ms:    Original latency ms (for score).
        cache:                  In-memory cache (mutated on miss).
        cache_path:             Disk path for crash-recovery writes.
        model_type:             Safety clamping type.
        num_classes:            Output classes for fine-tune accuracy metric.
        num_calibration_batches: Calibration batches for importance collection.
        max_pruning_ratio:      Hard cap per group (passed to MetaPruner).
        round_to:               Round pruned channels to this multiple.
        isomorphic:             Force same structure on all coupled groups.
        input_shape:            Dummy input for latency measurement.
        latency_iters:          Timed forward passes (keep small).
        accuracy_drop_threshold: pp threshold used for scoring/selection,
                                independent of early_abort_threshold below.
                                Passed straight through to apply_structured_
                                pruning's KD recovery fine-tune.
        early_abort_threshold: Direct pp value — "abort this trial's
                                fine-tune after epoch 1 if its drop exceeds
                                this many percentage points." Not a
                                multiplier. None (default) = disabled.

    Returns:
        Dict: pruning_ratio, max_pruning_ratio, fine_tune_epochs, fine_tune_lr,
              iterative_steps, accuracy, kl_divergence, size_mb, latency_ms,
              score, phase, proxy_score, real_score, layers_pruned_count.

    Raises:
        RuntimeError: propagated from measure_accuracy() when EVERY batch in
                      this evaluation failed structurally (message contains
                      "No samples were evaluated").  Intentionally NOT
                      swallowed — the caller (find_optimal_pruning_params)
                      counts these toward the 2-consecutive-failure abort.
    """

    # max_pruning_ratio is now a searched variable, so it must be in the cache key
    # to avoid collisions between evaluations with the same ratio but different cap.
    cache_key = (
        f"{round(pruning_ratio,     4)}"
        f"|{round(max_pruning_ratio, 2)}"
        f"|{fine_tune_epochs}"
        f"|{round(fine_tune_lr,     8)}"
        f"|{iterative_steps}"
    )
    if cache_key in cache:
        return cache[cache_key]

    kl_divergence: Optional[float] = None
    layers_pruned_count: int = -1   # -1 = unknown (eval failed before pruning ran)
    torch_pruning_incompatible: bool = False

    try:
        pruned = apply_structured_pruning(
            model,
            dataloader=dataloader,
            device=device,
            pruning_ratio=pruning_ratio,
            model_type=model_type,
            num_classes=num_classes,
            fine_tune_epochs=fine_tune_epochs,
            fine_tune_lr=fine_tune_lr,
            num_calibration_batches=num_calibration_batches,
            iterative_steps=iterative_steps,
            round_to=round_to,
            isomorphic=isomorphic,
            max_pruning_ratio=max_pruning_ratio,
            residual_max_ratio=residual_max_ratio,
            test_loader=test_loader,
            baseline_accuracy=baseline_accuracy,
            accuracy_drop_threshold=accuracy_drop_threshold,
            early_abort_threshold=early_abort_threshold,
        )
        _report = getattr(pruned, '_pruning_report', {}) or {}
        layers_pruned_count = len(_report.get('layers_pruned', []))
        torch_pruning_incompatible = _report.get('torch_pruning_incompatible', False)
        size_mb    = get_model_size_mb(pruned)
        accuracy   = measure_accuracy(pruned, test_loader, device)
        # latency_iters=0 means skip latency measurement during search (use baseline).
        # This saves 13 forward passes per trial.  Final pipeline measures latency properly.
        if latency_iters > 0:
            lat        = measure_latency(
                pruned, input_shape, device,
                num_iterations=latency_iters, warmup=3, input_dtype=input_dtype,
            )
            latency_ms = lat['mean_ms']
        else:
            latency_ms = baseline_latency_ms   # neutral: latency factor = 1.0

        # Extract KL divergence from the behavioral probe attached by apply_structured_pruning.
        # The probe measures how much the model's output distribution changed after pruning.
        # We multiply (1/(1+KL)) into the score so configurations that catastrophically
        # change model behaviour are ranked lower even if they achieve good size/latency.
        probe = getattr(pruned, '_pruning_report', {}).get('behavioral_probe', {})
        kl_val = probe.get('value', None)
        if probe.get('metric') == 'kl_divergence' and kl_val is not None:
            kl_divergence = float(kl_val)

        score = compression_quality_index(
            accuracy, size_mb, baseline_accuracy, baseline_size,
            latency_ms, baseline_latency_ms,
            kl_divergence=kl_divergence,
            w_accuracy=w_accuracy, w_size=w_size,
            w_latency=w_latency, w_kl=w_kl,
        )
    except RuntimeError as exc:
        if 'No samples were evaluated' in str(exc):
            raise
        print(f"  ⚠️  Pruning eval failed "
              f"(ratio={pruning_ratio:.3f}, max_ratio={max_pruning_ratio:.2f}, "
              f"steps={iterative_steps}): {exc}")
        accuracy   = 0.0
        size_mb    = baseline_size
        latency_ms = baseline_latency_ms
        score      = 0.0
    except (TypeError, AttributeError, NameError, ImportError):
        # Same reasoning as _evaluate_epsilon: these are almost always a
        # wrong/missing kwarg or a typo in the calling code, not a genuine
        # "this pruning config doesn't work" result. Propagate instead of
        # silently scoring accuracy=0.0 and letting the search conclude
        # "pruning makes this model worse" based on a crash, not real data.
        raise
    except Exception as exc:
        print(f"  ⚠️  Pruning eval failed "
              f"(ratio={pruning_ratio:.3f}, max_ratio={max_pruning_ratio:.2f}, "
              f"steps={iterative_steps}): {exc}")
        accuracy   = 0.0
        size_mb    = baseline_size
        latency_ms = baseline_latency_ms
        score      = 0.0

    result = {
        'pruning_ratio':       round(pruning_ratio,     4),
        'max_pruning_ratio':   round(max_pruning_ratio, 2),
        'fine_tune_epochs':    fine_tune_epochs,
        'fine_tune_lr':        fine_tune_lr,
        'iterative_steps':     iterative_steps,
        'accuracy':            accuracy,
        'kl_divergence':       kl_divergence,
        'size_mb':             size_mb,
        'latency_ms':          latency_ms,
        'score':               score,
        'phase':               'uncached',
        'proxy_score':         None,
        'real_score':          score,
        'layers_pruned_count': layers_pruned_count,
        'torch_pruning_incompatible': torch_pruning_incompatible,
    }
    cache[cache_key] = result
    _save_cache(cache, cache_path)
    return result


# ============================================================================
# ── PUBLIC API: LRF EPSILON SEARCH ───────────────────────────────────────────
# ============================================================================

def find_optimal_epsilon_smart(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    baseline_accuracy: float,
    baseline_size: float,
    baseline_latency_ms: Optional[float] = None,
    tolerance: float = 0.05,
    num_trials: int = 15,
    cache_path: str = "epsilon_cache.json",
    min_layer_size: int = 64,
    min_rank: int = 1,
    skip_large_kernels: bool = False,
    input_shape: tuple = (1, 3, 224, 224),
    input_dtype: Optional[torch.dtype] = None,
    w_accuracy: float = 1.0,
    w_size: float = 1.0,
    w_latency: float = 1.0,
    accuracy_drop_threshold: float = 5.0,
    # ── Per-trial fine-tuning (opt-in — see _evaluate_epsilon's docstring
    #    for the cost tradeoff this introduces) ───────────────────────────────
    dataloader: Optional[DataLoader] = None,
    num_classes: Optional[int] = None,
    search_ft_epochs: int = 0,
    search_ft_lr: float = 0.0001,
    early_abort_threshold: Optional[float] = None,
) -> Tuple[Optional[float], dict, list]:
    """
    Two-phase LRF epsilon search: anchor sampling + binary search refinement.

    accuracy_drop_threshold: max allowed accuracy drop (pp). Anchor for refinement
    uses threshold+10pp; final selection uses strict threshold.

    Uses the Compression Quality Index (CQI) to rank epsilon candidates.
    CQI weights control how much each factor (accuracy, size) influences ranking.
    Latency is not measured during search (scores are neutral on that axis)
    so w_latency has no effect here — it is accepted for API consistency.

    PER-TRIAL FINE-TUNING (opt-in via search_ft_epochs, default 0):
      Historically this search never fine-tuned — every trial applied LRF
      and measured accuracy immediately. search_ft_epochs > 0 changes that:
      every anchor AND every ternary-refinement trial now also runs a real
      KD recovery fine-tune (via apply_low_rank_factorization's own existing
      mechanism — see _evaluate_epsilon) before its accuracy is measured.

      THIS IS A REAL COST INCREASE, NOT A FREE IMPROVEMENT. With
      search_ft_epochs=0 (default), num_trials trials cost ~num_trials
      forward-pass-only evaluations (seconds each). With search_ft_epochs=N,
      the cost becomes ~num_trials × N real training epochs — on a large
      model this can take as long as a pruning search with comparable
      settings, where it previously took a small fraction of that time.
      early_abort_threshold (below) caps the WORST case — a clearly-bad
      epsilon aborts its fine-tune after epoch 1 — but does not eliminate
      the cost for epsilons that look viable through every epoch.

      dataloader/num_classes must both be supplied (non-None) for any
      fine-tuning to actually happen, mirroring
      apply_low_rank_factorization's own contract exactly — passing
      search_ft_epochs > 0 without them is a silent no-op (a warning prints
      from inside apply_low_rank_factorization, search proceeds as if
      search_ft_epochs were 0).

    Budget allocation (controlled by num_trials):
      search_budget   = num_trials
      n_anchors       = min(5, search_budget)
      n_ternary_iters = max(0, search_budget - n_anchors) // 2
      (Unaffected by search_ft_epochs — that controls cost PER trial, not
      the NUMBER of trials.)

    Structural-failure abort: if 2 CONSECUTIVE trials (across anchors AND
    ternary refinement combined — one shared counter, reset on any success)
    raise the "No samples were evaluated" structural failure, the search
    stops immediately and proceeds straight to Phase 3's final selection
    using whatever was cached before the abort.  If literally zero trials
    ever succeeded, this returns (None, {'warning': ...}, history) instead
    of crashing.  (When search_ft_epochs > 0, this can now also be raised
    from inside a trial's fine-tune — e.g. its own early-abort check's
    measure_accuracy() call — not just from the post-fine-tune measurement;
    both are treated identically by this protocol.)

    Args:
        model:                Uncompressed base model.  Never modified.  Also
                              the fine-tune's KD teacher when fine-tuning runs
                              (via apply_low_rank_factorization's default).
        test_loader:          Evaluation DataLoader.  Also the early-abort
                              check's test-set source when fine-tuning runs.
        device:               'cuda' or 'cpu'.
        baseline_accuracy:    Original model accuracy %.  Also the early-abort
                              check's drop reference — this search's notion of
                              "baseline" already IS the absolute original
                              model, so no separate value is needed for that.
        baseline_size:        Original model size in MB.
        baseline_latency_ms:  Original model latency ms.  Measured internally
                              (10 iterations) if None.
        tolerance:            Ternary search stops when interval width < this.
        num_trials:           Total evaluation budget.
        cache_path:           JSON file for crash-recovery cache.
        min_layer_size:       Skip LRF layers with dim <= this.
        min_rank:             Skip LRF layers when the computed rank is below this.
        skip_large_kernels:   If True, skip Conv2d layers with kernel size > 1×1.
        input_shape:          Dummy input shape for latency measurement.
        input_dtype:          Dummy input dtype for latency measurement.
                              None = float32 (vision default). Pass torch.long
                              for NLP models (token-ID inputs) — get this from
                              the registry's meta['input_dtype'], not a
                              class-name guess (every NLPClassifierWrapper-
                              wrapped model has the same class name regardless
                              of which HF model it wraps, so guessing from
                              type(model).__name__ can never work here).
        w_accuracy:           CQI accuracy factor weight (default 1.0).
        w_size:               CQI size factor weight (default 1.0).
        w_latency:            CQI latency factor weight (default 1.0, no effect
                              during search since latency is not measured).
        accuracy_drop_threshold: Max allowed accuracy drop (pp) for final
                              selection.  ALSO the pp value early_abort_threshold
                              scales, when fine-tuning is active.
        dataloader:           Training/calibration DataLoader for the optional
                              per-trial fine-tune.  None (default) = no
                              fine-tuning regardless of search_ft_epochs.
        num_classes:          Output class count for the optional fine-tune.
        search_ft_epochs:     KD recovery epochs PER TRIAL.  0 (default) =
                              old behaviour — no fine-tune, fastest search.
        search_ft_lr:         Learning rate for the optional per-trial fine-tune.
        early_abort_threshold: Direct pp value — "abort a trial's fine-tune
                              after epoch 1 if its drop exceeds this many
                              percentage points." Not a multiplier on
                              accuracy_drop_threshold. Only relevant when
                              search_ft_epochs > 0. None (default) = disabled
                              even if fine-tuning runs. Note: with the
                              default EPSILON_SEARCH_FT_EPOCHS=1 there is no
                              "remaining epoch" to abort out of, so this has
                              little effect here unless you raise that value.

    Returns:
        (optimal_epsilon, all_results_dict, search_history)
          optimal_epsilon   : Best ε (float) or None if CQI < 0.5 or nothing succeeded.
          all_results_dict  : {str(eps): {epsilon, accuracy, size_mb, latency_ms,
                               score, proxy_score, real_score, phase}}.
                               May contain a 'warning' string key.
          search_history    : Ordered list of all evaluation dicts.
    """
    LATENCY_ITERS = 10
    # Dynamic anchors: n_anchors evenly spaced across (0, 1.0].
    # Formula: i/(n_anchors+1) for i in 1..n_anchors+1 (last = 1.0 excluded).
    # Avoids hardcoding 0.1 as the floor — search can now explore < 0.1.
    _n_eps_anchors = 5
    ANCHORS = [round(i / (_n_eps_anchors + 1), 4) for i in range(1, _n_eps_anchors + 1)]
    # e.g. n=5 → [0.1667, 0.333, 0.5, 0.667, 0.833]

    cache          = _load_cache(cache_path)
    history: List[dict] = []

    # ── Measure baseline latency ──────────────────────────────────────────────
    if baseline_latency_ms is None:
        print("  [ε-search] Measuring baseline latency (10 iterations)...")
        try:
            # BUGFIX: this used to guess NLP-ness from the model's own class
            # name ('bert' in str(type(model).__name__).lower()...), but
            # every registry NLP model is wrapped in NLPClassifierWrapper —
            # type(model).__name__ is ALWAYS 'NLPClassifierWrapper', never
            # containing 'bert'/'roberta' — so that check never once fired.
            # Combined with input_shape still defaulting to the vision shape
            # (1,3,224,224), a 4D float tensor got forced through the NLP
            # model's forward pass, crashing deep inside it (typically a
            # `batch_size, seq_length = input_ids.size()`-style unpack
            # against a 4-tuple). Silently caught below and reported as a
            # fake 1.0ms fallback. Now uses the real, caller-supplied
            # input_shape/input_dtype (the registry's actual values, when
            # the caller passes them — see run_compression_pipeline).
            _lat = measure_latency(model, input_shape, device,
                                   num_iterations=LATENCY_ITERS, warmup=3, input_dtype=input_dtype)
            baseline_latency_ms = _lat['mean_ms']
            print(f"  [ε-search] Baseline latency measured: {baseline_latency_ms:.2f} ms")
        except Exception as e:
            print(f"  [ε-search] ⚠️  Latency measurement failed: {e}")
            print(f"  [ε-search] Falling back to 1.0 ms (this disables latency-based ranking)")
            baseline_latency_ms = 1.0
    print(f"  [ε-search] Baseline: {baseline_accuracy:.2f}%  "
          f"{baseline_size:.2f} MB  {baseline_latency_ms:.2f} ms")

    # ── No proxy — always run on the caller's device (CUDA if available) ─────────
    # INT8 proxy runs on CPU only (no CUDA INT8 kernel for conv layers).
    # For EfficientNet-style models, CPU INT8 is ~10× SLOWER than GPU float32.
    # Using the proxy would make epsilon search take 10× longer, not shorter.
    # Running the real model on CUDA is always the right choice.
    use_proxy   = False
    eval_model  = model
    eval_device = device
    eval_bl_lat = baseline_latency_ms
    remaining   = num_trials
    print(f"  [ε-search] Budget: {num_trials} trials on {device.upper()} (no proxy — GPU float32 faster than CPU INT8).")

    # ── Budget allocation ─────────────────────────────────────────────────────
    n_anchors      = min(len(ANCHORS), remaining)
    ternary_budget = max(0, remaining - n_anchors)
    max_iter       = ternary_budget // 2   # each iteration = 2 evals
    run_anchors    = ANCHORS[:n_anchors]

    # ── Structural-failure tracking — shared across anchors AND ternary ──────
    _failure_state = {'count': 0, 'aborted': False}

    def _on_structural_failure(label: str) -> None:
        _failure_state['count'] += 1
        print(f"     Consecutive structural failures: {_failure_state['count']}/2")
        if _failure_state['count'] >= 2:
            print(f"\n  🛑 SEARCH ABORTED ({label}) — 2 consecutive structural "
                  f"failures. Proceeding to final selection with whatever "
                  f"succeeded before the abort.\n")
            _failure_state['aborted'] = True

    # ── Phase 1: Anchor Sampling ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EPSILON SEARCH — PHASE 1: ANCHOR SAMPLING")
    print(f"{'=' * 70}")
    print(f"  Baseline : {baseline_accuracy:.2f}%  |  {baseline_size:.2f} MB  "
          f"|  {baseline_latency_ms:.2f} ms")
    print(f"  Anchors  : {run_anchors}  "
          f"(ternary budget: {ternary_budget} evals → {max_iter} iterations)\n")

    for anchor in run_anchors:
        try:
            result = _evaluate_epsilon(
                eval_model, test_loader, eval_device, anchor,
                baseline_accuracy, baseline_size, eval_bl_lat,
                min_layer_size, cache, input_shape, LATENCY_ITERS,
                w_accuracy=w_accuracy, w_size=w_size, w_latency=w_latency,
                min_rank=min_rank, skip_large_kernels=skip_large_kernels,
                dataloader=dataloader, num_classes=num_classes,
                fine_tune_epochs=search_ft_epochs, fine_tune_lr=search_ft_lr,
                accuracy_drop_threshold=accuracy_drop_threshold,
                early_abort_threshold=early_abort_threshold,
            )
        except RuntimeError as exc:
            if 'No samples were evaluated' in str(exc):
                print(f"  ❌ Trial failed structurally (ε={anchor:.4f}): {exc}")
                _on_structural_failure("anchor sampling")
                if _failure_state['aborted']:
                    break
                continue
            raise

        _failure_state['count'] = 0   # any success resets the counter

        if result.get('phase') in ('uncached', None):
            result['phase']       = 'anchor'
        result['proxy_score'] = result['score'] if use_proxy else None
        cache[str(round(anchor, 4))] = result
        _save_cache(cache, cache_path)

        print(f"  ε={anchor:.2f}  acc={result['accuracy']:.1f}%  "
              f"size={result['size_mb']:.2f} MB  "
              f"lat={result['latency_ms']:.1f} ms  "
              f"score={result['score']:.3f}")
        history.append({
            'phase': 'anchor', 'epsilon': round(anchor, 4),
            'score': result['score'], 'accuracy': result['accuracy'],
            'size_mb': result['size_mb'], 'latency_ms': result['latency_ms'],
            'iteration': 0,
        })

    _succeeded_anchors = [a for a in run_anchors if str(round(a, 4)) in cache]

    if not _succeeded_anchors:
        print("\n  ⚠️  Every epsilon anchor failed — no viable epsilon found.")
        return None, {'warning': 'All epsilon trials failed (structurally or otherwise).'}, history

    # Landscape summary
    print(f"\n  {'─' * 72}")
    print(f"  {'ε':>6}  {'accuracy':>9}  {'size':>8}  {'latency':>9}  {'score':>8}")
    print(f"  {'─' * 72}")
    anchor_scores = {a: cache[str(round(a, 4))]['score'] for a in _succeeded_anchors}
    best_anchor   = max(anchor_scores, key=anchor_scores.get)
    for anchor in _succeeded_anchors:
        r      = cache[str(round(anchor, 4))]
        marker = '  ← best' if anchor == best_anchor else ''
        print(f"  ε={anchor:.2f}  {r['accuracy']:>7.1f}%  "
              f"{r['size_mb']:>6.2f} MB  {r['latency_ms']:>7.1f} ms  "
              f"{r['score']:>8.3f}{marker}")
    print(f"  {'─' * 72}\n")

    _relax = accuracy_drop_threshold + 10.0
    _viable = [a for a in _succeeded_anchors
               if (baseline_accuracy - cache[str(round(a,4))].get('accuracy',0)) <= _relax]
    refine_anchor = (max(_viable, key=lambda a: cache[str(round(a,4))]['score'])
                     if _viable else best_anchor)
    if refine_anchor != best_anchor:
        _d = baseline_accuracy - cache[str(round(best_anchor,4))].get('accuracy',0)
        print(f"  ℹ️  CQI-best ε={best_anchor:.2f} has {_d:.1f}pp drop "
              f"(>{_relax:.0f}pp relaxed limit). Refining ε={refine_anchor:.2f} instead.\n")
    if refine_anchor in (_succeeded_anchors[0], _succeeded_anchors[-1]):
        print(f"  ⚠  Refine anchor at boundary ε={refine_anchor:.2f}.\n")

    # ── Phase 2: Binary Search Refinement (skipped if Phase 1 aborted) ───────
    if not _failure_state['aborted'] and max_iter > 0:
        print(f"{'=' * 70}")
        print("EPSILON SEARCH — PHASE 2: BINARY SEARCH REFINEMENT")
        print(f"{'=' * 70}")

        best_idx = _succeeded_anchors.index(refine_anchor)
        _step = _succeeded_anchors[1] - _succeeded_anchors[0] if len(_succeeded_anchors) > 1 else 0.15
        if best_idx == 0:
            # Best anchor is at lower boundary — extend below it.
            # Min epsilon: LRF rank of 1 corresponds to ε=1/min_dim ≈ 0.001.
            # Use half the anchor step below, floor at 0.01.
            low  = max(0.01, refine_anchor - _step)
            high = _succeeded_anchors[min(1, len(_succeeded_anchors) - 1)]
        elif best_idx == len(_succeeded_anchors) - 1:
            # Best anchor at upper boundary — extend above.
            low  = _succeeded_anchors[best_idx - 1]
            high = min(1.0, refine_anchor + _step)
        else:
            low  = _succeeded_anchors[best_idx - 1]
            high = _succeeded_anchors[best_idx + 1]

        print(f"  Refining ε={refine_anchor:.2f} → [{low:.4f}, {high:.4f}]  "
              f"budget={max_iter} evals\n")

        def _eps_score_fn(eps: float) -> Tuple[Optional[float], Optional[dict]]:
            if _failure_state['aborted']:
                return None, None
            ck         = str(round(eps, 4))
            was_cached = ck in cache
            try:
                result = _evaluate_epsilon(
                    eval_model, test_loader, eval_device, eps,
                    baseline_accuracy, baseline_size, eval_bl_lat,
                    min_layer_size, cache, input_shape, LATENCY_ITERS,
                    w_accuracy=w_accuracy, w_size=w_size, w_latency=w_latency,
                    min_rank=min_rank, skip_large_kernels=skip_large_kernels,
                    dataloader=dataloader, num_classes=num_classes,
                    fine_tune_epochs=search_ft_epochs, fine_tune_lr=search_ft_lr,
                    accuracy_drop_threshold=accuracy_drop_threshold,
                    early_abort_threshold=early_abort_threshold,
                )
            except RuntimeError as exc:
                if 'No samples were evaluated' in str(exc):
                    print(f"  ❌ Trial failed structurally (ε={eps:.4f}): {exc}")
                    _on_structural_failure("ternary refinement")
                    return None, None
                raise
            _failure_state['count'] = 0
            if not was_cached:
                result['phase']       = 'ternary'
                result['proxy_score'] = result['score'] if use_proxy else None
                cache[ck]             = result
                _save_cache(cache, cache_path)
            return result['score'], result

        _, ternary_hist = _ternary_search(
            score_fn       = _eps_score_fn,
            low            = low,
            high           = high,
            tolerance      = 0.001,  # very tight: use full budget, not early-exit
            max_iterations = max_iter,
            phase_label    = 'ternary',
        )
        history.extend(ternary_hist)
        print(f"\n  Ternary search done.  {len(ternary_hist)} evaluation(s).\n")
    elif _failure_state['aborted']:
        print("  [ε-search] Phase 2 skipped — search already aborted in Phase 1.\n")

    # ── Phase 3: Final Selection ──────────────────────────────────────────────
    # Pools the ENTIRE cache regardless of how far the search got before any
    # abort — this is what makes the abort protocol safe: whatever succeeded
    # is still eligible to win, nothing collected before an abort is wasted.
    print(f"{'=' * 70}")
    print("EPSILON SEARCH — PHASE 3: FINAL SELECTION")
    print(f"{'=' * 70}\n")

    valid: List[Tuple[float, dict]] = [
        (float(k), v)
        for k, v in cache.items()
        if isinstance(v, dict) and 'score' in v
        and (baseline_accuracy - v.get('accuracy', 0)) <= accuracy_drop_threshold
    ]
    if not valid:
        print(f"  ⚠️  No epsilon within {accuracy_drop_threshold:.1f}pp — "
              f"returning best available.")
        valid = [(float(k), v) for k, v in cache.items()
                 if isinstance(v, dict) and 'score' in v]

    if not valid:
        print("  ⚠️  No epsilon trial succeeded at all — no viable epsilon.")
        return None, {'warning': 'No epsilon trial produced a usable result.'}, history

    valid.sort(key=lambda x: x[1]['score'], reverse=True)

    # No tie-break — highest score wins unconditionally.
    # A tie-break that prefers lower ε can override a genuinely higher-scoring
    # ε=0.9 entry, which defeats the purpose of score-based optimisation.

    print(f"  {'Rank':>4}  {'ε':>8}  {'Accuracy':>9}  {'Size':>8}  "
          f"{'Latency':>9}  {'Score':>8}  Phase")
    print(f"  {'─' * 72}")
    for rank, (eps, r) in enumerate(valid[:3], start=1):
        print(f"  #{rank:>3}   ε={eps:.4f}  {r['accuracy']:>7.2f}%  "
              f"{r['size_mb']:>6.2f} MB  {r['latency_ms']:>7.1f} ms  "
              f"{r['score']:>8.3f}  {r.get('phase','?')}")
    print()

    optimal_epsilon_value: float         = valid[0][0]
    winner: dict                         = valid[0][1]
    optimal_epsilon: Optional[float]     = optimal_epsilon_value
    warning_msg:     Optional[str]       = None

    # Viability gate: score < 1.0 means compression makes combined tradeoff worse.
    if winner['score'] < 1.0:
        accuracy_drop = baseline_accuracy - winner['accuracy']
        warning_msg = (
            f"Best ε={optimal_epsilon_value:.4f} scores {winner['score']:.3f} < 1.0. "
            f"Accuracy drop: {accuracy_drop:.2f}%. "
            f"LRF makes this model worse overall — skipping LRF."
        )
        optimal_epsilon = None
        print(f"  ❌  {warning_msg}\n")

    # ── Phase 4: Real-model validation (when proxy was used) ─────────────────
    if use_proxy and optimal_epsilon is not None:
        print(f"  [ε-search] Validating ε={optimal_epsilon:.4f} on real model...")
        real_result = _evaluate_epsilon(
            model, test_loader, device, optimal_epsilon,
            baseline_accuracy, baseline_size, baseline_latency_ms,
            min_layer_size, {}, input_shape, LATENCY_ITERS,
            w_accuracy=w_accuracy, w_size=w_size, w_latency=w_latency,
            min_rank=min_rank, skip_large_kernels=skip_large_kernels,
            dataloader=dataloader, num_classes=num_classes,
            fine_tune_epochs=search_ft_epochs, fine_tune_lr=search_ft_lr,
            accuracy_drop_threshold=accuracy_drop_threshold,
            early_abort_threshold=early_abort_threshold,
        )
        key = str(round(optimal_epsilon, 4))
        if key in cache:
            cache[key]['real_score']  = real_result['score']
            cache[key]['proxy_score'] = winner['score']
        _save_cache(cache, cache_path)
        print(f"    Proxy score: {winner['score']:.3f}  "
              f"Real score: {real_result['score']:.3f}  "
              f"Accuracy: {real_result['accuracy']:.2f}%")

    if not warning_msg:
        acc_drop_display = baseline_accuracy - winner['accuracy']
        print(
            f"\n  ✅ Optimal epsilon : {optimal_epsilon:.4f}\n"
            f"     Accuracy       : {winner['accuracy']:.2f}%  "
            f"(drop: {acc_drop_display:.2f}%)\n"
            f"     Size           : {winner['size_mb']:.2f} MB\n"
            f"     Latency        : {winner['latency_ms']:.2f} ms\n"
            f"     Score          : {winner['score']:.3f}\n"
        )

    # Build output results dict
    all_results: dict = {}
    for k, v in cache.items():
        if isinstance(v, dict) and 'score' in v:
            entry = dict(v)
            entry.setdefault('proxy_score', None)
            entry.setdefault('real_score',  None)
            all_results[k] = entry
    if warning_msg:
        all_results['warning'] = warning_msg

    return optimal_epsilon, all_results, history


# ============================================================================
# ── PUBLIC API: PRUNING HYPERPARAMETER SEARCH ────────────────────────────────
# ============================================================================

def find_optimal_pruning_params(
    model: nn.Module,
    dataloader: DataLoader,
    test_loader: DataLoader,
    device: str,
    baseline_accuracy: float,
    baseline_size: float,
    baseline_latency_ms: Optional[float] = None,
    num_trials: int = 16,
    cache_path: str = "pruning_search_cache.json",
    # Fixed params — not searched
    model_type: str = 'classifier',
    num_classes: int = 10,
    num_calibration_batches: int = 50,
    max_pruning_ratio: float = 1.0,
    residual_max_ratio: Optional[float] = None,
    round_to: Optional[int] = None,
    isomorphic: bool = False,
    input_shape: tuple = (1, 3, 224, 224),
    input_dtype: Optional[torch.dtype] = None,
    w_accuracy: float = 1.0,
    w_size: float = 1.0,
    w_latency: float = 1.0,
    w_kl: float = 1.0,
    accuracy_drop_threshold: float = 5.0,
    search_ft_epochs: int = 1,
    search_ft_lr: float = 1e-4,
    early_abort_threshold: Optional[float] = None,
) -> Tuple[Optional[dict], dict, list]:
    """
    Hyperparameter search over pruning_ratio and pruning_max_ratio.

    Searched:
      pruning_ratio (continuous)    — ~60% of budget via anchor + ternary
      max_pruning_ratio (discrete)  — grid search at best ratio (~35%)
      iterative_steps (1 vs 2)      — one comparison at best config

    Not searched (configurable but fixed during each individual trial):
      fine_tune_epochs / fine_tune_lr (search_ft_epochs, search_ft_lr in main.py)
      model_type, num_classes, num_calibration_batches, round_to, isomorphic.

    CQI weights adjust which factor most influences the ranking:
      w_accuracy=2, w_size=1 → accuracy twice as important as compression
      w_kl=3 → search strongly avoids configs with high KL (behavioural drift)
      w_latency is accepted for API consistency but latency is not measured
      during search (scores are neutral on that axis).

    Accuracy-drop gating now applies to EVERY phase, not just Phase 1's
    anchor selection.  Phase 1b (ternary ratio refinement), Phase 2
    (max_pruning_ratio grid), and Phase 3 (iterative_steps comparison) all
    use the same strict → relaxed(+10pp) → score-only tier fallback via
    _select_best_by_tier().  Phase 4 no longer just validates a single
    funnel winner — it pools EVERY evaluation from the entire search,
    filters to those within accuracy_drop_threshold, and picks the
    highest-scoring survivor (mirroring find_optimal_epsilon_smart's
    Phase 3).  This means a low-ratio anchor with a small accuracy drop can
    win outright over a high-ratio anchor with a higher raw CQI score but a
    catastrophic accuracy drop — raw score alone no longer overrides the
    accuracy gate at the final step.

    Structural-failure abort: shares the same 2-consecutive-failure
    protocol as find_optimal_epsilon_smart.  The counter is shared across
    ALL phases (anchors, ternary, grid, iter-steps) via the single internal
    _run_eval() choke-point every phase calls through.  On abort, remaining
    trials in the current and all subsequent phases are skipped; Phase 4
    still runs against whatever is cached.

    Early-abort threshold (a DIFFERENT, broader check — degraded-but-valid
    accuracy, not a structural break): when early_abort_threshold is set,
    EVERY trial's fine-tune (across all phases, also via the shared
    _run_eval() choke-point) can stop after epoch 1 if that epoch's drop vs
    baseline_accuracy already exceeds early_abort_threshold directly (a pp
    value, not a multiplier on accuracy_drop_threshold) — see
    _kd_recovery_fine_tune's docstring for the full mechanism. This is
    independent of (and does not interact with) the
    structural-failure protocol above: a trial that early-aborts here still
    counts as a SUCCESS for the consecutive-failure counter (it produced a
    valid, if poor, result — nothing raised), and is still fully eligible to
    win at Phase 4 if its score happens to be competitive.

    Budget allocation:
      ratio_budget  = max(5, int(0.60 * num_trials))
      grid_budget   = max(2, int(0.35 * num_trials))
      iter_budget   = 1

    Args:
        model:                  Base model.  Never modified.
        dataloader:             DataLoader for pruning calibration + fine-tuning.
        test_loader:            DataLoader for accuracy evaluation.
        device:                 'cuda' or 'cpu'.
        baseline_accuracy:      Original model accuracy %.
        baseline_size:          Original model size in MB.
        baseline_latency_ms:    Original model latency ms (measured if None).
        num_trials:             Total evaluation budget.
        cache_path:             JSON crash-recovery cache file.
        model_type:             Pruning safety clamping type (not searched).
        num_classes:            Output class count for fine-tune metric.
        num_calibration_batches: Activation collection batches (full pipeline).
        max_pruning_ratio:      Upper bound of the max_ratio grid search.
        residual_max_ratio:     Ceiling for auto-detected residual/skip-
                                connection-coupled groups specifically, held
                                fixed across every trial in this search (not
                                itself a searched dimension the way max_ratio
                                is) — None (default) falls back to whatever
                                max_pruning_ratio each individual trial is
                                testing. See apply_structured_pruning's
                                docstring for the detection mechanism.
        round_to:               Round channel counts to this multiple.
        isomorphic:              Force isomorphic pruning across groups.
        input_shape:            Dummy input for latency measurement.
        w_accuracy:             CQI accuracy factor weight (default 1.0).
        w_size:                 CQI size factor weight (default 1.0).
        w_latency:              CQI latency factor weight (not active during search).
        w_kl:                   CQI KL divergence penalty weight (default 1.0).
        accuracy_drop_threshold: Max allowed accuracy drop (pp), applied at
                                 every phase and the final selection.  ALSO the
                                 pp value early_abort_threshold scales.
        search_ft_epochs:       Fine-tune epochs per trial during search
                                 (controllable from main.py).
        search_ft_lr:           Fine-tune learning rate per trial during search
                                 (controllable from main.py).
        early_abort_threshold: Direct pp value — "abort a trial's fine-tune
                                 after epoch 1 if its drop exceeds this many
                                 percentage points." Not a multiplier on
                                 accuracy_drop_threshold. None (default) =
                                 disabled — every trial's fine-tune runs all
                                 search_ft_epochs regardless of how bad
                                 epoch 1 looks (old behaviour).

    Returns:
        (optimal_params, all_results_dict, search_history)
          optimal_params   : {'pruning_ratio', 'pruning_max_ratio',
                              'iterative_steps'} or None (no viable config).
          all_results_dict : {cache_key: {all eval fields}} + optional 'warning'.
          search_history   : Ordered list of all evaluation dicts.
    """
    LATENCY_ITERS       = 10
    SEARCH_FT_EPOCHS    = search_ft_epochs   # controllable from main.py
    SEARCH_FT_LR        = search_ft_lr       # controllable from main.py
    SEARCH_CAL_BATCHES  = 10     # reduced from caller's value for search speed
    # 10 cal batches is enough to rank configs — activation stats stabilise quickly.
    # The final pipeline run uses the full num_calibration_batches (e.g. 50).
    # 10 vs 50 batches = 5× faster calibration × 20 trials = major speedup.
    # Candidates for max_pruning_ratio grid — evenly spaced within [0.2, param]
    # The upper bound is the caller-supplied max_pruning_ratio; smaller values
    # prevent global budget from collapsing individual layers.
    def _build_max_ratio_candidates(upper: float, n: int) -> list:
        lo = 0.2
        hi = max(lo + 0.05, upper)
        if n == 1:
            return [round(upper, 2)]
        step = (hi - lo) / (n - 1)
        return [round(lo + i * step, 2) for i in range(n)]

    cache          = _load_cache(cache_path)
    history: List[dict] = []

    # ── Measure baseline latency ──────────────────────────────────────────────
    if baseline_latency_ms is None:
        print("  [prune-search] Measuring baseline latency (10 iterations)...")
        try:
            # Same bug as the epsilon search had: this used to always use the
            # vision-default input_shape with no dtype override at all, so an
            # NLP model would silently crash and fall back to a fake 1.0ms
            # with NO error printed (unlike epsilon search's version, which
            # at least logged the failure). Now uses the real caller-supplied
            # input_shape/input_dtype and reports failures out loud.
            _lat = measure_latency(model, input_shape, device,
                                   num_iterations=LATENCY_ITERS, warmup=3, input_dtype=input_dtype)
            baseline_latency_ms = _lat['mean_ms']
            print(f"  [prune-search] Baseline latency measured: {baseline_latency_ms:.2f} ms")
        except Exception as e:
            print(f"  [prune-search] ⚠️  Latency measurement failed: {e}")
            print(f"  [prune-search] Falling back to 1.0 ms (this disables latency-based ranking)")
            baseline_latency_ms = 1.0
    print(f"  [prune-search] Baseline: {baseline_accuracy:.2f}%  "
          f"{baseline_size:.2f} MB  {baseline_latency_ms:.2f} ms")

    # ── No proxy — all pruning search evaluations run on the real model on device ─
    # Torch-Pruning requires float32 model tracing, so an INT8 proxy cannot be
    # used for the pruning step itself.  Running a proxy correlation check would
    # call apply_structured_pruning on CPU, which is catastrophically slow:
    # 50 cal batches + 1 epoch fine-tuning on CPU ≈ 5–10 minutes per check.
    # All evaluations run on the caller's device (CUDA when available).
    remaining = num_trials
    print(f"  [prune-search] Budget: {num_trials} trials on {device.upper()} (real model, no proxy).")

    # ── Budget allocation ─────────────────────────────────────────────────────
    ratio_budget = max(5, int(0.60 * remaining))
    grid_budget  = max(2, int(0.35 * remaining))
    iter_budget  = 1
    leftover     = max(0, remaining - ratio_budget - grid_budget - iter_budget)
    grid_budget += leftover

    n_ratio_anchors  = min(5, ratio_budget)
    ratio_ternary_b  = max(0, ratio_budget - n_ratio_anchors)
    ratio_max_iter   = ratio_ternary_b // 2

    print(f"  Budget allocation → ratio: {ratio_budget}  grid: {grid_budget}  "
          f"iter_steps: {iter_budget}")

    # Ratio anchors: divide [0, max_pruning_ratio] into n equal parts.
    # first = max/n, last = max. Scales with max_pruning_ratio, no hardcoded 0.05.
    ratio_anchors = [
        round(max_pruning_ratio * i / n_ratio_anchors, 4)
        for i in range(1, n_ratio_anchors + 1)
    ]

    # ── Structural-failure tracking — single choke point every phase calls
    #    through (_run_eval), so the 2-consecutive-failure abort protocol
    #    covers Phase 1, 1b, 2, and 3 uniformly without per-phase plumbing.
    _failure_state = {'count': 0, 'aborted': False}

    def _run_eval(
        ratio: float, it_steps: int,
        max_ratio: float = max_pruning_ratio,
    ) -> Optional[dict]:
        """
        Shared eval helper — always runs on real float32 model on device (CUDA).
        max_ratio is a variable parameter so Phase 2 can explore different caps.
        Uses SEARCH_CAL_BATCHES (10) not num_calibration_batches (50) — 5× faster.
        Skips latency measurement (latency_iters=0 means use baseline_latency_ms).

        Returns None when the search has already aborted, or when this trial
        itself is the 2nd consecutive structural failure (which also sets the
        abort flag).  Callers must treat a None return as "skip this trial".
        """
        if _failure_state['aborted']:
            return None
        try:
            result = _evaluate_pruning_config(
                model, dataloader, test_loader, device,
                pruning_ratio=ratio,
                fine_tune_epochs=SEARCH_FT_EPOCHS,
                fine_tune_lr=SEARCH_FT_LR,
                iterative_steps=it_steps,
                baseline_accuracy=baseline_accuracy,
                baseline_size=baseline_size,
                baseline_latency_ms=baseline_latency_ms,
                cache=cache, cache_path=cache_path,
                model_type=model_type, num_classes=num_classes,
                num_calibration_batches=SEARCH_CAL_BATCHES,   # reduced for search speed
                max_pruning_ratio=max_ratio,
                residual_max_ratio=residual_max_ratio,
                round_to=round_to, isomorphic=isomorphic,
                input_shape=input_shape,
                input_dtype=input_dtype,
                latency_iters=0,   # skip latency during search — saves 13 fwd passes/trial
                w_accuracy=w_accuracy, w_size=w_size,
                w_latency=w_latency, w_kl=w_kl,
                accuracy_drop_threshold=accuracy_drop_threshold,
                early_abort_threshold=early_abort_threshold,
            )
        except RuntimeError as exc:
            if 'No samples were evaluated' in str(exc):
                print(f"  ❌ Trial failed structurally (ratio={ratio:.3f}): {exc}")
                _failure_state['count'] += 1
                print(f"     Consecutive structural failures: {_failure_state['count']}/2")
                if _failure_state['count'] >= 2:
                    print(f"\n  🛑 SEARCH ABORTED — 2 consecutive structural "
                          f"failures. Skipping remaining trials; proceeding to "
                          f"Phase 4 with whatever succeeded before the abort.\n")
                    _failure_state['aborted'] = True
                return None
            raise
        # Torch-Pruning couldn't build a dependency graph for this
        # architecture AT ALL (see the explanation apply_structured_pruning
        # already printed) -- this is deterministic given the architecture,
        # not data- or ratio-dependent, so unlike the generic structural-
        # failure counter above (which tolerates one isolated failure
        # before aborting, since that kind of failure genuinely could be a
        # fluke), this aborts the entire search immediately on the very
        # first occurrence. Every remaining trial in every remaining phase
        # would fail identically; there is nothing to learn by trying again.
        if result.get('torch_pruning_incompatible', False):
            print(
                f"\n  🛑 PRUNING SEARCH ABORTED — Torch-Pruning cannot build "
                f"a dependency graph for this architecture at all (see the "
                f"explanation above). This is architectural, not ratio-"
                f"dependent, so every remaining trial would fail "
                f"identically — stopping immediately rather than spending "
                f"the rest of the search budget confirming that.\n"
            )
            _failure_state['aborted'] = True
            return None
        _failure_state['count'] = 0
        return result

    # ── Phase 1: Ratio Anchor Sampling ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PRUNING SEARCH — PHASE 1: RATIO ANCHORS")
    print(f"{'=' * 70}")
    print(f"  Anchors  : {ratio_anchors}  "
          f"(ft_epochs={SEARCH_FT_EPOCHS}, lr={SEARCH_FT_LR:.0e})\n")

    anchor_results: List[Tuple[float, dict]] = []
    for ratio in ratio_anchors:
        result = _run_eval(ratio, 1)
        if result is None:
            if _failure_state['aborted']:
                break
            continue
        if result.get('phase') in ('uncached', None):
            result['phase'] = 'ratio_anchor'
        kl_str = f"  KL={result['kl_divergence']:.3f}" if result.get('kl_divergence') is not None else ""
        print(f"  ratio={ratio:.3f}  acc={result['accuracy']:.1f}%  "
              f"size={result['size_mb']:.2f} MB  "
              f"lat={result['latency_ms']:.1f} ms  score={result['score']:.3f}{kl_str}")
        anchor_results.append((ratio, result))
        history.append({
            'phase': 'ratio_anchor',
            'pruning_ratio': ratio, 'max_pruning_ratio': max_pruning_ratio,
            'iterative_steps': 1,
            'accuracy': result['accuracy'], 'score': result['score'],
            'kl_divergence': result.get('kl_divergence'),
            'size_mb': result['size_mb'], 'latency_ms': result['latency_ms'],
            'iteration': 0,
        })

    if not anchor_results:
        print("\n  ⚠️  Every ratio anchor failed — no viable pruning configuration.")
        all_results = {k: dict(v) for k, v in cache.items() if isinstance(v, dict)}
        return None, all_results, history

    # Print anchor summary table (mirrors the epsilon search table format)
    anchor_scores_map = {r: result['score']    for r, result in anchor_results}
    anchor_acc_map    = {r: result['accuracy'] for r, result in anchor_results}

    def _anchor_acc_drop(a: float) -> float:
        # BUGFIX: previously looked up `cache.get(str(round(a,4)), {})`, which
        # mis-keyed against _evaluate_pruning_config's actual compound cache
        # key ("ratio|max_ratio|epochs|lr|steps") and ALWAYS missed, silently
        # falling back to "0.0pp drop" for every anchor regardless of its real
        # accuracy.  Now reads directly from this search's own in-memory
        # anchor_acc_map, built from the exact results just computed above.
        return baseline_accuracy - anchor_acc_map.get(a, baseline_accuracy)

    print("  " + "─" * 72)
    print(f"  {'Ratio':>8}   {'Accuracy':>9}   {'Size':>8}   {'Latency':>9}   {'Score':>7}   {'Acc Drop':>9}")
    print("  " + "─" * 72)
    _best_within_threshold = max(
        (r for r, _ in anchor_results if _anchor_acc_drop(r) <= accuracy_drop_threshold),
        key=lambda a: anchor_scores_map.get(a, 0),
        default=None,
    )
    for _ar, _res in anchor_results:
        _drop = _anchor_acc_drop(_ar)
        _within = _drop <= accuracy_drop_threshold
        _marker = "  ← best (within threshold)" if _ar == _best_within_threshold else ""
        _drop_str = f"+{abs(_drop):.1f}pp" if _drop < 0 else f"-{_drop:.1f}pp"
        _ok = "✅" if _within else "❌"
        print(f"  r={_ar:.3f}   {_res['accuracy']:>7.1f}%   {_res['size_mb']:>6.2f} MB"
              f"   {_res['latency_ms']:>7.1f} ms   {_res['score']:>6.3f}   {_ok} {_drop_str}{_marker}")
    print("  " + "─" * 72)

    # Find best ratio anchor.
    # Selection priority:
    #   1. Prefer anchors within accuracy_drop_threshold (default 5pp) — ranked by score.
    #   2. If none pass the strict threshold, widen to threshold+10pp — ranked by score.
    #   3. Last resort: pure score maximization (no accuracy gate).
    # This prevents the score formula from selecting a heavily-pruned model with
    # near-zero size (huge size_ratio) but catastrophic accuracy loss.
    _succeeded_ratios = [r for r, _ in anchor_results]
    _pv_strict = [a for a in _succeeded_ratios if _anchor_acc_drop(a) <= accuracy_drop_threshold]
    _pv_relax  = [a for a in _succeeded_ratios if _anchor_acc_drop(a) <= accuracy_drop_threshold + 10.0]

    if _pv_strict:
        best_ratio_anchor = max(_pv_strict, key=lambda a: anchor_scores_map.get(a, 0))
        _tier = "strict"
    elif _pv_relax:
        best_ratio_anchor = max(_pv_relax,  key=lambda a: anchor_scores_map.get(a, 0))
        _tier = "relaxed"
    elif anchor_scores_map:
        best_ratio_anchor = max(anchor_scores_map, key=anchor_scores_map.get)
        _tier = "score-only"
    else:
        best_ratio_anchor = _succeeded_ratios[len(_succeeded_ratios) // 2]
        _tier = "fallback"

    print(f"\n  Best anchor ratio = {best_ratio_anchor:.4f}  "
          f"(score={anchor_scores_map.get(best_ratio_anchor, 0):.3f}  "
          f"acc_drop={_anchor_acc_drop(best_ratio_anchor):.1f}pp  tier={_tier})\n")

    # ── Phase 1b: Ratio Ternary Refinement ───────────────────────────────────
    # Early exit: if every anchor pruned 0 layers, the architecture has no
    # prunable Conv2d groups (e.g. transformers). Further search is noise.
    # (Reads layers_pruned_count, which _evaluate_pruning_config now actually
    # populates — previously this checked a key that was never written, so
    # the comparison was always False for any populated entry.)
    _all_pruned_zero = all(
        result.get('layers_pruned_count', -1) == 0
        for _, result in anchor_results
    )
    if _all_pruned_zero:
        print(f"  ⚠️  All anchors pruned 0 layers — Torch-Pruning found no prunable "
              f"Conv2d groups in this architecture (e.g. transformer). "
              f"Skipping remaining search phases.")
        all_results = {k: dict(v) for k, v in cache.items() if isinstance(v, dict)}
        return None, all_results, history

    if not _failure_state['aborted'] and ratio_max_iter > 0:
        print(f"{'=' * 70}")
        print("PRUNING SEARCH — PHASE 1b: RATIO REFINEMENT")
        print(f"{'=' * 70}\n")

        best_idx = _succeeded_ratios.index(best_ratio_anchor)
        _p_step = _succeeded_ratios[1] - _succeeded_ratios[0] if len(_succeeded_ratios) > 1 else 0.1
        if best_idx == 0:
            r_low  = max(0.01, best_ratio_anchor - _p_step)
            r_high = _succeeded_ratios[min(1, len(_succeeded_ratios) - 1)]
        elif best_idx == len(_succeeded_ratios) - 1:
            r_low  = _succeeded_ratios[best_idx - 1]
            r_high = min(max_pruning_ratio, best_ratio_anchor + _p_step)
        else:
            r_low  = _succeeded_ratios[best_idx - 1]
            r_high = _succeeded_ratios[best_idx + 1]

        print(f"  Binary search in [{r_low:.3f}, {r_high:.3f}]  budget={ratio_max_iter} evals\n")

        def _ratio_score_fn(ratio: float) -> Tuple[Optional[float], Optional[dict]]:
            result = _run_eval(ratio, 1)
            if result is None:
                return None, None
            if result.get('phase') in ('uncached', None):
                result['phase'] = 'ratio_ternary'
            return result['score'], result

        best_ratio_refined, ternary_hist = _ternary_search(
            score_fn       = _ratio_score_fn,
            low            = r_low,
            high           = r_high,
            tolerance      = 0.001,  # use full budget
            max_iterations = ratio_max_iter,
            phase_label    = 'ratio_ternary',
        )
        history.extend(ternary_hist)

        # Find best ratio across all ratio-phase evaluations — now gated by
        # accuracy tier, not pure score-max (bug #5: ternary refinement used
        # to climb purely toward the highest-CQI region even when that region
        # was the most accuracy-destructive one).
        ratio_phase_entries = [
            v for v in cache.values()
            if isinstance(v, dict)
            and v.get('iterative_steps') == 1
            and 'pruning_ratio' in v
            and v.get('max_pruning_ratio') == round(max_pruning_ratio, 2)
        ]
        _best_ratio_entry = _select_best_by_tier(ratio_phase_entries, baseline_accuracy, accuracy_drop_threshold)
        best_ratio = _best_ratio_entry['pruning_ratio'] if _best_ratio_entry else best_ratio_anchor
        print(f"\n  Best ratio after refinement: {best_ratio:.4f}\n")
    else:
        if _failure_state['aborted']:
            print("  [prune-search] Phase 1b skipped — search already aborted.\n")
        best_ratio = best_ratio_anchor

    # ── Phase 2: max_pruning_ratio Grid ─────────────────────────────────────────
    # The pruning_ratio sets the global channel removal target.
    # max_pruning_ratio caps how much any single layer can be pruned, preventing
    # the global budget from collapsing individual layers.  These two interact:
    # a lower max_ratio protects critical layers but reduces effective compression.
    # We grid-search max_ratio at the best pruning_ratio found in Phase 1.
    max_ratio_results: dict = {}
    if not _failure_state['aborted']:
        print(f"{'=' * 70}")
        print("PRUNING SEARCH — PHASE 2: MAX-RATIO GRID")
        print(f"{'=' * 70}")
        max_ratio_candidates = _build_max_ratio_candidates(max_pruning_ratio, grid_budget)
        print(f"  Fixed ratio={best_ratio:.4f}  "
              f"max_ratio candidates: {max_ratio_candidates}\n")

        for mr in max_ratio_candidates:
            result = _run_eval(best_ratio, 1, max_ratio=mr)
            if result is None:
                if _failure_state['aborted']:
                    break
                continue
            if result.get('phase') in ('uncached', None):
                result['phase'] = 'max_ratio_grid'
            max_ratio_results[mr] = result
            kl_str = f"  KL={result['kl_divergence']:.3f}" if result.get('kl_divergence') is not None else ""
            print(f"  max_ratio={mr:.2f}  acc={result['accuracy']:.1f}%  "
                  f"score={result['score']:.3f}{kl_str}")
            history.append({
                'phase': 'max_ratio_grid',
                'pruning_ratio': best_ratio, 'max_pruning_ratio': mr,
                'iterative_steps': 1,
                'accuracy': result['accuracy'], 'score': result['score'],
                'kl_divergence': result.get('kl_divergence'),
                'size_mb': result['size_mb'], 'latency_ms': result['latency_ms'],
                'iteration': 0,
            })
    else:
        print("  [prune-search] Phase 2 skipped — search already aborted.\n")

    if max_ratio_results:
        # Select by accuracy-gated tier, then score (bug #5 fix — previously
        # pure score-max regardless of accuracy).
        _mr_best = _select_best_by_tier(list(max_ratio_results.values()), baseline_accuracy, accuracy_drop_threshold)
        best_max_ratio = next(mr for mr, r in max_ratio_results.items() if r is _mr_best) \
                         if _mr_best is not None else max_pruning_ratio
    else:
        best_max_ratio = max_pruning_ratio
    print(f"\n  Best max_ratio = {best_max_ratio:.2f}  "
          f"(score={max_ratio_results.get(best_max_ratio, {}).get('score', 0):.3f})\n")

    # ── Phase 3: iterative_steps 1 vs 2 ─────────────────────────────────────
    res_it1 = res_it2 = None
    if not _failure_state['aborted']:
        print(f"{'=' * 70}")
        print("PRUNING SEARCH — PHASE 3: ITERATIVE STEPS COMPARISON")
        print(f"{'=' * 70}\n")

        res_it1 = _run_eval(best_ratio, 1, max_ratio=best_max_ratio)
        res_it2 = _run_eval(best_ratio, 2, max_ratio=best_max_ratio)
        if res_it2 is not None and res_it2.get('phase') in ('uncached', None):
            res_it2['phase'] = 'iter_steps'

        if res_it1 is not None:
            print(f"  steps=1  acc={res_it1['accuracy']:.1f}%  score={res_it1['score']:.3f}")
        if res_it2 is not None:
            print(f"  steps=2  acc={res_it2['accuracy']:.1f}%  score={res_it2['score']:.3f}")
    else:
        print("  [prune-search] Phase 3 skipped — search already aborted.\n")

    _iter_candidates = [r for r in (res_it1, res_it2) if r is not None]
    if _iter_candidates:
        _iter_best = _select_best_by_tier(_iter_candidates, baseline_accuracy, accuracy_drop_threshold)
        best_iter_steps   = 2 if (res_it2 is not None and _iter_best is res_it2) else 1
        best_iter_result  = _iter_best if _iter_best is not None else (res_it1 or res_it2)
        print(f"  → Best: iterative_steps = {best_iter_steps}\n")
        history.append({
            'phase': 'iter_steps',
            'pruning_ratio': best_ratio, 'max_pruning_ratio': best_max_ratio,
            'iterative_steps': best_iter_steps,
            'accuracy': best_iter_result['accuracy'],
            'score': best_iter_result['score'],
            'kl_divergence': best_iter_result.get('kl_divergence'),
            'size_mb': best_iter_result['size_mb'],
            'latency_ms': best_iter_result['latency_ms'],
            'iteration': 0,
        })
    else:
        # Phase 3 produced nothing (search aborted before it could run) —
        # fall back to whatever Phase 1/1b/2 already established.
        best_iter_steps = 1

    # ── Phase 4: Final Selection (pool ENTIRE cache, filter, pick best) ───────
    # Completely rewritten (bug #5): previously this only re-validated the
    # single funnel winner from Phases 1-3 with a binary accept/reject and NO
    # fallback — meaning a config sitting in the cache the whole time with a
    # small, acceptable accuracy drop (e.g. a low ratio anchor) could never
    # win if the funnel happened to walk toward a different, accuracy-
    # destroying region (which is exactly what CQI_W_SIZE > CQI_W_ACCURACY
    # tends to do).  Now mirrors find_optimal_epsilon_smart's Phase 3 exactly:
    # pool every cached evaluation, filter to the ones within
    # accuracy_drop_threshold, and pick the highest score among survivors —
    # falling back to "best available regardless of threshold" only if the
    # survivor pool is empty.
    print(f"{'=' * 70}")
    print("PRUNING SEARCH — PHASE 4: FINAL SELECTION")
    print(f"{'=' * 70}\n")

    all_cached = [v for v in cache.values() if isinstance(v, dict) and 'score' in v and 'pruning_ratio' in v]

    if not all_cached:
        print("  ⚠️  No pruning configuration was successfully evaluated at all.")
        return None, {k: dict(v) for k, v in cache.items() if isinstance(v, dict)}, history

    _strict = [v for v in all_cached if (baseline_accuracy - v.get('accuracy', 0)) <= accuracy_drop_threshold]
    if _strict:
        candidates = _strict
        _tier_used = "strict"
    else:
        print(f"  ⚠️  No config within {accuracy_drop_threshold:.1f}pp — returning best available.")
        candidates = all_cached
        _tier_used = "score-only (no config met threshold)"

    candidates_sorted = sorted(candidates, key=lambda v: v['score'], reverse=True)
    winner = candidates_sorted[0]

    print(f"  {'Rank':>4}  {'Ratio':>8}  {'MaxR':>6}  {'Steps':>5}  {'Accuracy':>9}  {'Size':>8}  {'Score':>8}")
    print(f"  {'-' * 60}")
    for rank, v in enumerate(candidates_sorted[:5], start=1):
        print(f"  #{rank:>3}   {v['pruning_ratio']:.4f}  {v.get('max_pruning_ratio',0):.2f}   "
              f"{v.get('iterative_steps',1):>5}  {v['accuracy']:>7.2f}%  "
              f"{v['size_mb']:>6.2f} MB  {v['score']:>8.3f}")
    print()

    warning_msg: Optional[str]    = None
    optimal_params: Optional[dict] = None
    acc_drop_final = baseline_accuracy - winner['accuracy']

    if winner['score'] < 1.0:
        warning_msg = (f"Best config (ratio={winner['pruning_ratio']:.4f}) scores "
                        f"{winner['score']:.3f} < 1.0 — pruning makes this model "
                        f"worse overall. Skipping pruning.")
        print(f"  ❌  {warning_msg}\n")
    elif acc_drop_final > accuracy_drop_threshold and _tier_used != "strict":
        warning_msg = (f"Best available config still drops accuracy "
                        f"{acc_drop_final:.2f}pp (threshold: "
                        f"{accuracy_drop_threshold:.1f}pp, tier={_tier_used}). "
                        f"Skipping pruning.")
        print(f"  ❌  {warning_msg}\n")
    else:
        optimal_params = {
            'pruning_ratio':     winner['pruning_ratio'],
            'pruning_max_ratio': winner.get('max_pruning_ratio', max_pruning_ratio),
            'iterative_steps':   winner.get('iterative_steps', 1),
        }
        print(f"  ✅ Optimal pruning params: {optimal_params}  "
              f"(accuracy={winner['accuracy']:.2f}%, drop={acc_drop_final:.2f}pp, "
              f"score={winner['score']:.3f})\n")

    all_results: dict = {
        k: dict(v) for k, v in cache.items()
        if isinstance(v, dict) and 'score' in v
    }
    if warning_msg:
        all_results['warning'] = warning_msg

    return optimal_params, all_results, history