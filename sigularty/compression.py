"""
compression.py
====================
Seven model compression and optimisation techniques for PyTorch models.

Techniques (in pipeline order)
--------------------------------
  BN Fusion          — fold BatchNorm into preceding Conv/Linear (zero accuracy cost)
  Sensitivity Anal.  — per-layer importance scoring to guide pruning ratios
  Structured Pruning — activation-statistics filter removal via Torch-Pruning
  Adaptive LRF       — per-layer SVD factorization with analytically computed epsilons
  Weight Clustering  — k-means quantisation of weight tensors
  KD Fine-tuning     — knowledge distillation from original model for accuracy recovery
  Quantization       — dynamic INT8 / static INT8 / FP16 / GPTQ-INT4 precision reduction

Public API
----------
  apply_bn_fusion(model)                               -> nn.Module        [Tier 1]
  compute_layer_sensitivity(model, dataloader, ...)    -> dict             [Tier 2A]
  sensitivity_to_pruning_ratios(sensitivities, ...)    -> dict             [Tier 2A]
  fine_tune_with_distillation(student, teacher, ...)   -> nn.Module        [Tier 2B]
  compute_adaptive_epsilons(model, ...)                -> dict             [Tier 3A]
  apply_adaptive_lrf(model, adaptive_epsilons, ...)    -> nn.Module        [Tier 3A]
  apply_gptq_quantization(model, dataloader, ...)      -> nn.Module        [Tier 3B]
  apply_structured_pruning(model, dataloader, ...)     -> nn.Module
  apply_low_rank_factorization(model, ...)             -> nn.Module
  apply_weight_clustering(model, dataloader, ...)      -> nn.Module
  apply_quantization(model, dataloader, mode, ...)     -> nn.Module
  apply_compression_pipeline(model, dataloader, ...)   -> nn.Module

Pipeline execution order:
  BN Fusion → Sensitivity Analysis → Structured Pruning → Adaptive/Standard LRF
  → Weight Clustering → KD Fine-tune → Quantization (GPTQ or standard)

Design rules:
  - Input models are NEVER mutated; deep copies are made internally.
  - All hyperparameters are explicit function arguments; no hidden globals.
  - No global state.
  - helper_functions.py utilities are NOT re-implemented here.

Major fine-tuning is ALWAYS knowledge distillation (KD)
---------------------------------------------------------
  Every recovery fine-tune in this file — after Structured Pruning, after
  Low-Rank Factorization, after Weight Clustering — uses the same shared
  KD recovery loop (_kd_recovery_fine_tune / fine_tune_with_distillation),
  teaching against the original, pre-compression model.  Plain cross-entropy
  fine-tuning is never used for these steps: the teacher's soft-label signal
  recovers more accuracy per epoch on the small calibration sets used during
  compression than hard labels alone.

Per-technique accuracy gating (global, applies to every technique)
----------------------------------------------------------------------
  apply_compression_pipeline() can optionally measure accuracy before and
  after EVERY technique (when test_loader is supplied) and revert any single
  technique whose OWN marginal accuracy drop exceeds accuracy_drop_threshold
  — independently of every other technique's budget.  BN Fusion (lossless by
  construction) and Sensitivity Analysis (read-only, no model mutation) are
  never gated.  The kept/reverted decision for each gated technique is
  recorded on the returned model as `._gating_report` (dict of
  {technique_label: bool_kept}), and the final post-pipeline accuracy as
  `._gating_accuracy`, so callers (run_compression_pipeline in
  helper_functions.py) can build an honest "techniques actually used" list
  and continue gating GPTQ / Quantization / the final KD step, which run
  outside this function entirely.
"""

import copy
from typing import List, Optional

import numpy as np
import torch
import torchmetrics
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ============================================================================
# ── TECHNIQUE 0: STRUCTURED PRUNING (via Torch-Pruning) ─────────────────────
# Theory: structured pruning removes entire output channels (filters) from
# Conv2d layers, producing genuinely smaller tensors that run faster on any
# hardware without requiring sparse matrix support.  Unstructured pruning
# (zeroing individual weights) needs sparse kernels to realise any speedup;
# structured pruning is immediately effective on dense hardware.
#
# Why activation statistics beat L1/L2 magnitude for unknown architectures:
#   L1/L2 sums absolute weight values per filter.  This measures the weight
#   magnitude of a learned parameter, not how much that filter actually
#   contributes to the representation on real data.  A filter with large
#   weights may produce near-zero activations if its input feature distribution
#   makes it irrelevant on the actual dataset.  For unknown customer
#   architectures we cannot assume anything about regularization, initialization,
#   or training objective.  A model trained with L2 weight decay will have
#   uniformly small weights regardless of importance; one trained without it
#   may have large filters that do very little.  L1/L2 produces opposite
#   importance rankings for these two models despite identical functionality.
#   Mean absolute activation on calibration data directly measures how much
#   each filter changes the representation on real inputs — the only metric
#   that generalises across arbitrary architectures without assumptions.
#
# Why Torch-Pruning:
#   Manual shape propagation (the previous approach) required a DFS-based
#   heuristic to find successor layers and skip SE blocks, skip connections,
#   and the final conv before the classifier.  Each new architecture required
#   new heuristics and each heuristic produced new shape mismatch crashes.
#   Torch-Pruning builds a proper computational dependency graph from a
#   real forward pass trace, then propagates channel removals through ALL
#   dependent layers atomically.  This handles skip connections, SE blocks,
#   multi-path architectures, and arbitrary customer models correctly without
#   any architecture-specific code.
#
# Global vs local pruning:
#   Local (uniform) pruning removes the same fraction of filters per layer.
#   This hits critical early layers equally hard as redundant later layers,
#   causing disproportionate accuracy drops.
#   Global pruning sets one target sparsity across the whole network and lets
#   importance scores decide per-layer ratios automatically — redundant layers
#   are pruned heavily, critical layers are left mostly intact.
# ============================================================================


class _ActivationImportance:
    """
    Activation-statistics importance metric for Torch-Pruning.

    Collects mean absolute activation per output channel by registering
    forward hooks on all Conv2d and Linear leaf layers, running calibration
    batches, then averaging.  This measures functional contribution on real
    data rather than weight magnitude.

    Usage:
        imp = _ActivationImportance()
        imp.register_hooks(model)
        try:
            # Run calibration batches through the model
            with torch.no_grad():
                for X, _ in calibration_batches:
                    model(X)
        finally:
            imp.remove_hooks()   # always remove even if batches fail
        # imp is now ready to be passed to a Torch-Pruning pruner

    Subclasses tp.importance.Importance so Torch-Pruning's MetaPruner
    accepts it as the importance argument.
    """

    def __init__(self) -> None:
        self._act_sum: dict  = {}   # id(module) -> cumulative importance tensor
        self._counts:  dict  = {}   # id(module) -> accumulation count
        self._hooks:   list  = []   # (hook_handle, module_ref) pairs
        # Keep module refs to prevent Python from reusing id() for new objects
        self._mod_refs: list = []

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on every Conv2d and Linear leaf module."""
        def _make_hook(mod_id: int):
            def _hook(module: nn.Module, inp, output: torch.Tensor) -> None:
                with torch.no_grad():
                    out = output.detach().float()
                    # Dispatch on MODULE TYPE, not tensor ndim. Conv2d's
                    # output layout is always [N, C, H, W] (channels at
                    # dim=1) -- unchanged below. nn.Linear transforms only
                    # the LAST dimension no matter how many leading dims the
                    # input has, so for Linear the channel dim is always
                    # dim=-1: [N,C] for a flat classifier, [N,L,C] for any
                    # transformer's sequence output (BERT/RoBERTa/ALBERT/
                    # DistilBERT/ViT), [N,H,W,C] for Swin's channels-LAST
                    # layout. The previous `if out.ndim == 4` branch assumed
                    # every 4D tensor was Conv2d's NCHW -- wrong for Swin --
                    # and `ndim == 3`, the standard shape for every
                    # transformer's Linear output, matched neither branch
                    # and silently recorded nothing at all: every attention/
                    # FFN Linear layer in any sequence model got zero
                    # importance data, and Torch-Pruning's own MetaPruner
                    # unconditionally skips any group with no importance
                    # score (`if imp is None: continue` in its own source)
                    # -- so those layers were never eligible for pruning in
                    # the first place, regardless of pruning_ratio.
                    if isinstance(module, nn.Conv2d):
                        if out.ndim != 4:
                            return          # unexpected shape — skip
                        imp = out.abs().mean(dim=(0, 2, 3))       # → [C]
                    else:                    # nn.Linear (only other hooked type)
                        if out.ndim < 2:
                            return          # unexpected shape — skip
                        imp = out.abs().mean(dim=tuple(range(out.ndim - 1)))  # → [C]
                    imp_cpu = imp.cpu()
                    if mod_id in self._act_sum:
                        self._act_sum[mod_id] = self._act_sum[mod_id] + imp_cpu
                        self._counts[mod_id] += 1
                    else:
                        self._act_sum[mod_id] = imp_cpu.clone()
                        self._counts[mod_id]  = 1
            return _hook

        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and not list(module.children()):
                h = module.register_forward_hook(_make_hook(id(module)))
                self._hooks.append(h)
                self._mod_refs.append(module)

    def remove_hooks(self) -> None:
        """Remove all registered hooks.  Safe to call multiple times."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._mod_refs.clear()

    def _scores(self, module: nn.Module) -> Optional[torch.Tensor]:
        """Return averaged importance scores for a module, or None if uncollected."""
        mod_id = id(module)
        if mod_id not in self._act_sum or self._counts.get(mod_id, 0) == 0:
            return None
        return self._act_sum[mod_id] / self._counts[mod_id]

    def get_all_scores(self) -> dict:
        """Return {id(module): averaged_scores} for all collected modules."""
        return {
            k: self._act_sum[k] / self._counts[k]
            for k in self._act_sum
            if self._counts.get(k, 0) > 0
        }

    @torch.no_grad()
    def __call__(self, group, **kwargs) -> Optional[torch.Tensor]:
        """
        Return per-channel importance scores for a torch-pruning dependency group.

        Torch-Pruning calls this once per group during pruner.step().
        A group bundles all layers whose channels are coupled by the
        dependency graph.  We find every output-channel-pruning dep in the
        group, look up activation scores, and return their mean.

        Returns None when no activation data is available for this group
        (torch-pruning will then fall back to its own default handling).
        """
        import torch_pruning as tp   # deferred — only imported when pruning runs

        group_imp: list = []

        for dep, idxs in group:
            module    = dep.target.module
            handler   = dep.handler
            # Identify output-channel pruning handlers by name (version-robust)
            hname = getattr(handler, '__name__', '')
            if 'out_channel' not in hname and 'out_feature' not in hname:
                continue

            scores = self._scores(module)
            if scores is None:
                continue

            # idxs may be a list, range, or tensor — normalise to a list of ints
            idx_list = [int(i) for i in idxs]
            idx_list = [i for i in idx_list if i < len(scores)]
            if not idx_list:
                continue

            group_imp.append(scores[idx_list])

        if not group_imp:
            return None   # no activation data for this group

        # Coupled layers must agree on channel count — use min length then average
        min_len = min(len(s) for s in group_imp)
        stacked = torch.stack([s[:min_len] for s in group_imp], dim=0)
        return stacked.mean(dim=0)


def _run_behavioral_probe(
    original_model: nn.Module,
    pruned_model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_batches: int = 10,
) -> dict:
    """
    Compare original and pruned model outputs on calibration data to assess
    how much the pruning changed the model's functional behaviour.

    Detection logic:
      If the output is 2D (batch × C) and each row approximately sums to 1
      after softmax (i.e., values look like logits for a classifier),
      use KL divergence between softmax(original) and softmax(pruned).
      Otherwise (embeddings, regression), use mean cosine similarity.

    KL divergence severity thresholds (lower is better):
      < 0.01  → negligible   (pruning had almost no effect)
      < 0.05  → acceptable   (safe to deploy)
      < 0.15  → moderate     (consider reducing pruning_ratio)
      ≥ 0.15  → high         (model behaviour changed significantly)

    Args:
        original_model: Unmodified baseline model.
        pruned_model:   Pruned model to compare against.
        dataloader:     DataLoader for calibration inputs.
        device:         'cuda' or 'cpu'.
        num_batches:    Number of batches to probe over.

    Returns:
        Dict with keys: metric, value, severity, recommendation.
    """
    orig_outputs:   list = []
    pruned_outputs: list = []

    original_model.eval()
    pruned_model.eval()

    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            X = X.to(device)
            try:
                o = original_model(X)
                p = pruned_model(X)
                orig_outputs.append(o.float().cpu())
                pruned_outputs.append(p.float().cpu())
            except Exception:
                continue

    if not orig_outputs:
        return {
            'metric':         'none',
            'value':          None,
            'severity':       'unknown',
            'recommendation': 'Probe could not run — no valid batches completed.',
        }

    orig_cat   = torch.cat(orig_outputs,  dim=0)
    pruned_cat = torch.cat(pruned_outputs, dim=0)

    is_classifier = (orig_cat.ndim == 2)

    if is_classifier:
        p_orig   = torch.softmax(orig_cat,   dim=-1).clamp(min=1e-8)
        p_pruned = torch.softmax(pruned_cat, dim=-1).clamp(min=1e-8)
        kl = (p_orig * (p_orig / p_pruned).log()).sum(dim=-1).mean().item()
        metric = 'kl_divergence'
        value  = kl
        if kl < 0.01:
            severity = 'negligible'
            recommendation = 'Pruning had negligible impact. Safe to deploy as-is.'
        elif kl < 0.05:
            severity = 'acceptable'
            recommendation = 'Small behaviour change. Run evaluation to confirm accuracy.'
        elif kl < 0.15:
            severity = 'moderate'
            recommendation = 'Noticeable behaviour shift. Consider reducing pruning_ratio or adding fine-tuning.'
        else:
            severity = 'high'
            recommendation = 'Significant behaviour change. Reduce pruning_ratio or increase fine-tuning epochs.'
    else:
        orig_flat   = orig_cat.view(orig_cat.size(0), -1)
        pruned_flat = pruned_cat.view(pruned_cat.size(0), -1)
        cos = nn.functional.cosine_similarity(orig_flat, pruned_flat, dim=1).mean().item()
        metric = 'cosine_similarity'
        value  = cos
        if cos > 0.99:
            severity = 'negligible'
            recommendation = 'Outputs nearly identical. Safe to deploy.'
        elif cos > 0.95:
            severity = 'acceptable'
            recommendation = 'Minor output drift. Evaluate on downstream task.'
        elif cos > 0.85:
            severity = 'moderate'
            recommendation = 'Moderate output drift. Consider reducing pruning_ratio.'
        else:
            severity = 'high'
            recommendation = 'Significant output drift. Reduce pruning_ratio or increase fine-tuning.'

    print(f"\n  [Probe] Metric: {metric}  Value: {value:.4f}  Severity: {severity.upper()}")
    severity_icons = {'negligible': '✅', 'acceptable': '✅', 'moderate': '⚠️', 'high': '❌'}
    print(f"  {severity_icons.get(severity, '?')} {recommendation}")

    return {
        'metric':         metric,
        'value':          value,
        'severity':       severity,
        'recommendation': recommendation,
    }


# ============================================================================
# ── SHARED KD RECOVERY FINE-TUNE ─────────────────────────────────────────────
# Used by Structured Pruning AND Low-Rank Factorization (Adaptive LRF too).
# Always knowledge distillation against a frozen teacher — never plain
# cross-entropy.  Written once here so the epoch-zero safety check and the
# loss formula stay identical across every caller instead of drifting apart.
# ============================================================================

def _kd_recovery_fine_tune(
    model: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    fine_tune_epochs: int,
    fine_tune_lr: float,
    device: str,
    num_classes: int,
    temperature: float = 4.0,
    alpha: float = 0.7,
    technique_label: str = "Fine-tune",
    test_loader: Optional[DataLoader] = None,
    baseline_accuracy: Optional[float] = None,
    accuracy_drop_threshold: Optional[float] = None,
    early_abort_threshold: Optional[float] = None,
) -> tuple:
    """
    Knowledge-distillation recovery fine-tune.

    `model` is fine-tuned IN-PLACE (it is already a fresh deepcopy by the
    time any caller in this file reaches this function — mirroring the old
    _fine_tune_after_pruning's contract).  `teacher` is used read-only under
    torch.no_grad() and is put into eval() mode (a benign mode flag, not a
    weight mutation — every technique in this codebase already does this to
    models it doesn't otherwise own, e.g. _run_behavioral_probe).

    Per-epoch accuracy is computed on the TRAINING/calibration batches being
    fine-tuned on (torchmetrics.Accuracy, same convention as the rest of this
    pipeline) — this is the same metric already printed during pruning's and
    clustering's fine-tune loops, NOT a held-out test-set accuracy.

    Safety check (always active): if epoch 1 reports EXACTLY 0.0% training
    accuracy, a major warning is printed and ALL remaining epochs are
    skipped — this is almost always a structural break (shape mismatch,
    dead/NaN gradients, wrong num_classes) rather than ordinary slow
    convergence, and continuing would waste compute on a fine-tune that
    cannot recover.  The (still-broken) model is returned regardless; the
    caller's accuracy-drop gate is responsible for reverting the technique
    that led to this fine-tune.

    Early-abort check (opt-in — only runs when test_loader, baseline_accuracy,
    accuracy_drop_threshold, AND early_abort_threshold are ALL supplied,
    and only when the epoch-1-exactly-0.0% case above did NOT already fire):
    after epoch 1, measures TEST accuracy (a genuinely held-out set, unlike
    the training-batch accuracy used above) and compares the drop against
    the absolute original model's baseline_accuracy. If that drop already
    exceeds early_abort_threshold (a direct percentage-point value, e.g.
    30.0 — NOT a multiplier on accuracy_drop_threshold), the remaining
    epochs are skipped: recovering that much ground in what's left is
    judged implausible, so finishing would only spend compute on a result
    the caller's accuracy-drop gate would very likely revert anyway. This
    costs exactly one extra measure_accuracy() pass over test_loader, and
    only after epoch 1 — negligible next to a real training epoch.
    accuracy_drop_threshold itself is NOT used in the comparison (that
    would make this a multiplier again) — it's accepted purely so callers
    can pass it straight through without a separate plumbing path; only
    early_abort_threshold is compared against.

    Returns:
        (model, history) where history is [{epoch, loss, acc}, ...].
    """
    print(f"\n  [{technique_label}] KD fine-tune: {fine_tune_epochs} epoch(s)  "
          f"T={temperature}  alpha={alpha}  lr={fine_tune_lr}")
    model.to(device)
    teacher.eval()   # frozen teacher; every teacher forward pass below is no_grad

    optimizer   = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)
    ce_loss     = nn.CrossEntropyLoss()
    kl_loss     = nn.KLDivLoss(reduction='batchmean')
    accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    history: list = []

    model.train()
    for epoch in tqdm(range(fine_tune_epochs), desc=f"  {technique_label}"):
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            try:
                student_logits = model(x)
            except RuntimeError as e:
                # Shape mismatch during forward pass — skip this batch rather
                # than crash the whole fine-tune (matches the old pruning
                # fine-tune's per-batch tolerance).
                if 'shape' in str(e).lower():
                    continue
                raise

            with torch.no_grad():
                teacher_logits = teacher(x)

            hard_loss     = ce_loss(student_logits, y)
            p_teacher     = torch.softmax(teacher_logits / temperature, dim=1)
            log_p_student = torch.log_softmax(student_logits / temperature, dim=1)
            soft_loss     = kl_loss(log_p_student, p_teacher) * (temperature ** 2)
            loss          = alpha * hard_loss + (1.0 - alpha) * soft_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy_fn(student_logits.argmax(1), y).item()
            n_batches  += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        avg_acc  = (total_acc  / n_batches) * 100 if n_batches > 0 else 0.0
        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'acc': avg_acc})
        print(f"    Epoch {epoch + 1}/{fine_tune_epochs}  "
              f"loss={avg_loss:.4f}  acc={avg_acc:.2f}%")

        if epoch == 0 and avg_acc == 0.0:
            print(
                f"\n  🚨 [{technique_label}] MAJOR WARNING: epoch 1 produced EXACTLY "
                f"0.0% accuracy. This almost always means a structural break "
                f"(shape mismatch, dead/NaN gradients, wrong num_classes) rather "
                f"than slow convergence. Skipping remaining "
                f"{fine_tune_epochs - 1} epoch(s) to avoid wasting compute — "
                f"the accuracy-drop gate will revert this technique.\n"
            )
            break
        elif (
            epoch == 0
            and fine_tune_epochs > 1
            and test_loader is not None
            and baseline_accuracy is not None
            and early_abort_threshold is not None
        ):
            # Deferred import — measure_accuracy lives in helper_functions.py;
            # only imported when this opt-in check actually runs, matching
            # this file's existing lazy-import convention for anything from
            # helper_functions.py (see apply_compression_pipeline).
            from sigularty.helper_functions import measure_accuracy as _measure_accuracy_ea
            test_acc = _measure_accuracy_ea(model, test_loader, device)
            model.train()   # measure_accuracy leaves the model in eval() mode —
                             # switch back before the next epoch's training loop
            drop = baseline_accuracy - test_acc
            if drop > early_abort_threshold:
                print(
                    f"\n  ⏭️  [{technique_label}] EARLY ABORT: after epoch 1, held-out "
                    f"test accuracy is {test_acc:.2f}% vs. the original model's "
                    f"{baseline_accuracy:.2f}% — a {drop:.2f}pp drop, exceeding the "
                    f"{early_abort_threshold:.1f}pp early-abort threshold. Recovering "
                    f"that much ground in the remaining {fine_tune_epochs - 1} "
                    f"epoch(s) is judged implausible, so skipping them rather than "
                    f"spending compute on a result the accuracy-drop gate would very "
                    f"likely revert anyway.\n"
                )
                break
            else:
                print(f"    [early-abort check] epoch 1 test-acc drop {drop:.2f}pp "
                      f"is within the {early_abort_threshold:.1f}pp threshold — "
                      f"continuing.")

    print(f"  [{technique_label}] Done.")
    return model, history


def _fine_tune_after_pruning(
    model: nn.Module,
    dataloader: DataLoader,
    fine_tune_epochs: int,
    fine_tune_lr: float,
    device: str,
    num_classes: int,
) -> tuple:
    """
    DEPRECATED — kept only as a documented historical marker.

    Pruning's recovery fine-tune now always goes through
    _kd_recovery_fine_tune() (knowledge distillation against the pre-pruning
    model), called directly from apply_structured_pruning().  This plain
    cross-entropy version is no longer called anywhere in this file.
    """
    raise NotImplementedError(
        "_fine_tune_after_pruning has been replaced by _kd_recovery_fine_tune "
        "(KD-based recovery against the original model). "
        "apply_structured_pruning() no longer calls this function."
    )


def _torch_pruning_incompatible_result(model: nn.Module, exc: Exception) -> nn.Module:
    """
    Graceful-skip result for apply_structured_pruning() when Torch-Pruning
    cannot build a dependency graph for this architecture AT ALL.

    Confirmed root cause (verified directly, not assumed): HuggingFace's
    Conv1D layers -- used by GPT-2-family models (distilgpt2's
    attn.c_attn/attn.c_proj/mlp.c_fc/mlp.c_proj) -- are not a type
    Torch-Pruning recognizes. It falls back to a generic "unwrapped
    parameters" path that corrupts its internal channel-shape tracking
    badly enough that graph construction itself raises TypeError -- this
    was confirmed to happen identically whether triggered via
    _find_residual_coupled_groups's own graph build or MetaPruner's own
    (independently reproduced both ways), so this single helper is called
    from both of those call sites rather than duplicating the skip logic.

    Returns the model COMPLETELY UNCHANGED (a deepcopy, zero modification)
    with a MINIMAL ._pruning_report -- deliberately not the full report
    shape a real pruning attempt produces (no importance scores, no
    behavioral probe, nothing meaningful was computed) -- marked with
    `torch_pruning_incompatible: True`. Every caller that can reach this
    (find_optimal_pruning_params's search loop, apply_compression_
    pipeline's direct pruning step, run_compression_pipeline's final
    report-building) checks this flag and treats the technique as if
    use_pruning had been False for the entire run, not as "ran, kept,
    zero effect" -- see the flag's check sites in optimization.py,
    apply_compression_pipeline() below, and run_compression_pipeline() in
    helper_functions.py.

    Args:
        model: The ORIGINAL (pre-pruning) model passed into
               apply_structured_pruning -- deepcopied fresh here rather
               than reusing whatever partially-modified state `pruned`
               (the function's working copy) was in when the exception
               fired, since some Torch-Pruning internals mutate the graph
               object's associated model in place even on a path that
               ultimately raises.
        exc:   The caught TypeError, included in the printed explanation.

    Returns:
        Unchanged deepcopy of `model` with the minimal marked report.
    """
    print(
        f"\n  🛑 [Structured Pruning] Torch-Pruning could not build a "
        f"dependency graph for this model's architecture -- "
        f"{type(exc).__name__}: {exc}\n"
        f"     This is commonly caused by non-standard layer types "
        f"Torch-Pruning doesn't recognize (e.g. HuggingFace's Conv1D, used "
        f"by GPT-2-family models) that corrupt its internal shape tracking "
        f"badly enough that graph construction itself fails -- independent "
        f"of pruning_ratio or any other setting; every trial would fail "
        f"identically. Skipping Structured Pruning for this run; the model "
        f"is returned unchanged.\n"
    )
    unchanged = copy.deepcopy(model)
    unchanged._pruning_report = {
        'layers_pruned':              [],
        'layers_skipped':             [],
        'torch_pruning_incompatible': True,
    }
    return unchanged


def _find_residual_coupled_groups(
    model: nn.Module,
    example_input: torch.Tensor,
    ignored_layers: list,
) -> List[List[nn.Module]]:
    """
    Architecture-agnostic detection of residual/skip-connection-coupled
    dependency groups, using Torch-Pruning's own dependency graph — no
    module-name matching of any kind. Replaces the old 'block.3' MBConv-
    specific string check, which only matched EfficientNet/MobileNet-style
    naming and produced zero matches on ResNet, ResNeXt, WideResNet,
    DenseNet, ConvNeXt, or any other residual family (confirmed directly:
    it printed "0 block.3 layers" on every trial of a real ResNet-50 run —
    see README's "Architecture-Agnostic Residual Group Detection" section
    for the full empirical story).

    Mechanism:
      Torch-Pruning classifies every simple pointwise operation — ReLU,
      other activations, AND elementwise addition — under a single
      OPTYPE.ELEMENTWISE category, so checking for ELEMENTWISE alone is NOT
      sufficient to find residual joins specifically: a plain
      Conv→BN→ReLU→Conv chain also passes through an ELEMENTWISE node (the
      ReLU), and would false-positive on every ordinary sequential layer in
      the network if that were the only check. The signal that correctly
      separates "this group is coupled because of a genuine tensor
      addition" from "this group is coupled because of an ordinary
      sequential dependency that happens to pass through an activation" is
      each node's own PyTorch grad_fn: a real residual/skip addition shows
      up as an 'AddBackward'-class autograd function, an activation shows
      up as something like 'ReluBackward0'. Filtering on the grad_fn class
      name (not just the OPTYPE) is what makes this detector precise.

      Verified empirically (see README): on ResNet-50, finds exactly the 4
      stage-boundary groups (conv3 + downsample.0 in every Bottleneck
      block, correctly coupled across each stage); on EfficientNet-B0,
      finds a strict superset of what the old 'block.3' check covered (the
      same 14 project convs, plus 14 more modules on the expand-conv side
      of the coupling it was silently missing); on VGG-16 (no skip
      connections at all), finds zero groups — the correct negative
      control.

    Args:
        model:          The (already deep-copied) model being pruned.
        example_input:  A single example input tensor for graph tracing —
                        reuses the same tensor apply_structured_pruning
                        already derives from the dataloader, no extra data
                        loading needed.
        ignored_layers: Layers to exclude (the output head) — matches the
                        ignored_layers list already passed to MetaPruner,
                        so the output head is never swept into a protected
                        group by mistake (a group can still legitimately
                        include the output head as a passive INPUT-side
                        member, since its input dimension must track
                        whatever the final stage's output width is — this
                        filters it out of the returned member lists).

    Returns:
        List of groups, each group a list of the real nn.Conv2d/nn.Linear
        modules whose channel count is coupled through at least one
        genuine residual/skip-connection addition. Empty list for
        architectures with no skip connections at all (e.g. VGG-style
        plain sequential CNNs).
    """
    residual_groups: List[List[nn.Module]] = []
    seen_signatures: set = set()

    import torch_pruning as tp   # lazy import — matches apply_structured_pruning's
                                  # own convention; this function is only ever
                                  # called from a context where torch-pruning has
                                  # already been confirmed installed, but has its
                                  # own scope and cannot see that local import.
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)
    for group in DG.get_all_groups(ignored_layers=ignored_layers):
        group_list = list(group)
        has_residual_join = any(
            node.type == tp.ops.OPTYPE.ELEMENTWISE
            and node.grad_fn is not None
            and 'Add' in type(node.grad_fn).__name__
            for dep, _idxs in group_list
            for node in (dep.source, dep.target)
        )
        if not has_residual_join:
            continue

        members = list(dict.fromkeys(
            dep.target.module for dep, _idxs in group_list
            if isinstance(dep.target.module, (nn.Conv2d, nn.Linear))
            and dep.target.module not in ignored_layers
        ))
        if not members:
            continue
        signature = tuple(sorted(id(m) for m in members))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        residual_groups.append(members)

    return residual_groups


def apply_structured_pruning(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    pruning_ratio: float   = 0.3,
    model_type: str        = 'unknown',
    num_classes: int       = 10,
    fine_tune_epochs: int  = 3,
    fine_tune_lr: float    = 0.0001,
    num_calibration_batches: int = 50,
    iterative_steps: int     = 1,
    round_to: Optional[int]  = None,
    isomorphic: bool         = False,
    max_pruning_ratio: float = 1.0,
    residual_max_ratio: Optional[float] = None,
    sensitivity_map: Optional[dict] = None,
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.7,
    test_loader: Optional[DataLoader] = None,
    baseline_accuracy: Optional[float] = None,
    accuracy_drop_threshold: Optional[float] = None,
    early_abort_threshold: Optional[float] = None,
    capture_pre_finetune_accuracy: bool = False,
) -> nn.Module:
    """
    Remove low-importance filters using Torch-Pruning with activation-statistics
    importance and global pruning.

    Returns a new model — the input is NEVER mutated.

    Technique overview:
      1. Deep copy the model — Torch-Pruning modifies in-place.
      3. Collect activation statistics via forward hooks on calibration batches.
         Mean absolute activation per output channel per layer — direct proxy
         for functional contribution on real data.
      4. Build a Torch-Pruning MetaPruner that traces the computational
         dependency graph from a real forward pass.  The graph covers skip
         connections, SE blocks, multi-path branches — no architecture-specific
         heuristics needed.
      5. Execute global pruning: one target sparsity across the whole network,
         activation scores decide per-layer ratios automatically.  The final
         Linear classifier head is automatically excluded from pruning.
      6. Optional fine-tuning to recover lost accuracy — ALWAYS knowledge
         distillation against `model` (this function's own pre-pruning
         parameter) as a frozen teacher.  See _kd_recovery_fine_tune().
      7. Behavioral probe comparing original vs pruned outputs.
         Result attached as ._pruning_report.

    Safety clamping by model_type:
      'classifier' → up to 0.5   (classifiers tolerate aggressive pruning)
      'embedding'  → clamped 0.2  (silent downstream task failure risk)
      'generative' → clamped 0.1  (generative models extremely sensitive)
      'unknown'    → clamped 0.2  (conservative default for customer models)

    Torch-Pruning hyperparameters exposed:
      iterative_steps: Prune in N equal steps rather than one shot.
                       More steps = more stable accuracy (but slower).
                       Default 1 = single-shot pruning.
      round_to:        Round pruned channel counts to multiples of this value.
                       Use 8 or 16 for Tensor Core alignment on GPU.
                       None = no rounding (default).
      isomorphic:      Force the same pruning structure across all coupled
                       groups (useful for parallel branches in custom models).
                       Default False.

    Args:
        model:                   Input model. NOT mutated; deep copy returned.
                                 ALSO used as the frozen KD teacher for the
                                 recovery fine-tune below (always — pruning
                                 is the first technique in the pipeline, so
                                 its own `model` parameter genuinely is the
                                 original, uncompressed reference).
        dataloader:              DataLoader for calibration and optional fine-tune.
        device:                  'cuda' or 'cpu'.
        pruning_ratio:           Target fraction of channels to remove globally.
        model_type:              Architecture class — controls safety clamping.
        num_classes:             Output class count (used by fine-tune accuracy).
        fine_tune_epochs:        Fine-tune epochs after pruning.  0 = skip.
        fine_tune_lr:            Fine-tune learning rate.
        num_calibration_batches: Batches for activation collection and probe.
        iterative_steps:         Number of pruning iterations (torch-pruning).
        round_to:                Round pruned channels to this multiple (torch-pruning).
        isomorphic:              Force isomorphic pruning (torch-pruning).
        max_pruning_ratio:       Hard cap: no single layer/coupled-group pruned
                                 more than this fraction of its channels — via
                                 pruning_ratio_dict (see the "WHAT WE OBSERVED"
                                 comment block below for why pruning_ratio_dict,
                                 not Torch-Pruning's own max_pruning_ratio kwarg,
                                 is the mechanism that actually enforces this).
                                 Also the fallback value for residual_max_ratio
                                 when that argument is None.
        residual_max_ratio:      Ceiling specifically for auto-detected residual/
                                 skip-connection-coupled groups — the layers
                                 whose channel count IS the residual stream for
                                 an entire network stage, so collapsing them
                                 damages every downstream block coupled to that
                                 stream, not just one layer's worth of capacity.
                                 None (default) = fall back to max_pruning_ratio.
                                 Set explicitly to give residual-critical layers
                                 a DIFFERENT ceiling than everything else. See
                                 _find_residual_coupled_groups() for the
                                 detection mechanism (architecture-agnostic,
                                 no module-name matching).
        sensitivity_map:         Optional output of compute_layer_sensitivity().
                                If provided, per-layer pruning ratios are computed
                                from sensitivity scores instead of uniform global pruning.
                                Overrides activation-statistics importance for ratio allocation.
        kd_temperature:          Softmax temperature for the KD recovery fine-tune.
        kd_alpha:                Hard-label (CE) weight for the KD recovery fine-tune;
                                 1-alpha is the distillation weight.
        test_loader:             Held-out DataLoader for the early-abort check.
                                 Same contract as apply_low_rank_factorization's.
        baseline_accuracy:       Original model's accuracy %, for the early-abort
                                 check's drop calculation.
        accuracy_drop_threshold: Accepted for API symmetry; not used directly
                                 in the early-abort comparison itself.
        early_abort_threshold:  Direct pp value — see _kd_recovery_fine_tune's
                                 docstring. None (default) = disabled.
        capture_pre_finetune_accuracy: If True (and test_loader is supplied),
                                 measures accuracy once right after pruning
                                 but before the recovery fine-tune starts,
                                 and attaches it as both
                                 ._pruning_report['pre_finetune_accuracy']
                                 and ._pre_finetune_accuracy on the returned
                                 model. Used by apply_compression_pipeline's
                                 per-technique impact reporting to separate
                                 pruning's own structural effect from its
                                 bundled fine-tune's contribution. Default
                                 False so this stays a pipeline-only cost —
                                 optimization.py's search functions call this
                                 function with test_loader already set (for
                                 the early-abort check above) but never pass
                                 this flag, so search trials never pay for or
                                 produce the extra measurement.

    Returns:
        Pruned (and optionally fine-tuned) nn.Module with ._pruning_report.

    Raises:
        ImportError: If torch-pruning is not installed.
    """
    try:
        import torch_pruning as tp
        import torch_pruning.dependency.constants as _tp_constants
        _tp_constants.MAX_RECURSION_DEPTH = 100_000
    except ImportError:
        raise ImportError(
            "torch-pruning is required for structured pruning.\n"
            "Install with: pip install torch-pruning"
        )

    # Guard: ratio >= 1.0 is nonsensical — clamp to 0.95
    if pruning_ratio >= 1.0:
        print(f"  ⚠️  pruning_ratio={pruning_ratio:.2f} is >= 1.0 (removes everything). "
              f"Clamped to 0.95. Check your PRUNING_RATIO constant in main.py.")
        pruning_ratio = 0.95

    # NOTE: there used to be a guard here — "if pruning_ratio > max_pruning_ratio:
    # pruning_ratio = max_pruning_ratio" — that has been REMOVED. That guard
    # silently collapsed the requested GLOBAL ratio down to the cap value
    # whenever the cap was set lower than the target, which defeated the
    # entire point of having an independent per-group ceiling: a search
    # sweeping max_pruning_ratio from 0.2 to 1.0 was never actually testing
    # "same overall compression, capped per layer" — it was testing "a
    # smaller overall compression target," since the global ratio itself got
    # overwritten before Torch-Pruning ever saw it. The global target is now
    # always honored as requested; max_pruning_ratio / residual_max_ratio
    # protect specific vulnerable groups via pruning_ratio_dict below
    # instead, letting the solver redistribute the rest of the budget onto
    # everything else. See README's "max_pruning_ratio Was Not Doing What
    # Its Name Said" section for the full empirical story.

    # Resolve the residual-group protection ratio: an explicit
    # residual_max_ratio wins if supplied, otherwise fall back to
    # max_pruning_ratio (same fallback pattern PRUNING_RESIDUAL_MAX_RATIO
    # uses in main.py).
    effective_residual_ratio = (
        residual_max_ratio if residual_max_ratio is not None else max_pruning_ratio
    )

    print(f"\n[Structured Pruning] Starting  ratio={pruning_ratio:.2f}  "
          f"model_type='{model_type}'  device={device}  "
          f"iterative_steps={iterative_steps}  round_to={round_to}  "
          f"isomorphic={isomorphic}  max_pruning_ratio={max_pruning_ratio}  "
          f"residual_max_ratio={effective_residual_ratio:.2f}"
          f"{' (from max_pruning_ratio)' if residual_max_ratio is None else ''}")

    # ── Deep copy — Torch-Pruning modifies in-place ───────────────────────────
    pruned = copy.deepcopy(model)
    pruned.to(device)
    pruned.eval()

    # ── Derive example input from the first dataloader batch ─────────────────
    # Torch-Pruning needs a real forward pass to trace the dependency graph.
    # We must NOT hardcode a shape — customer models have unknown input dims.
    try:
        _first_batch, _ = next(iter(dataloader))
        example_input   = _first_batch[:1].to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not derive example_input from dataloader: {exc}"
        ) from exc

    # ── Build pruning_ratio_dict and ignored_layers ─────────────────────────────
    #
    # WHAT WE OBSERVED, AND WHAT ACTUALLY FIXES IT:
    #   A pure activation-magnitude importance ranking can concentrate almost
    #   the entire prune budget onto ONE coupled dependency group even at a
    #   mild overall target — e.g. a real trained ResNet-50 lost 92% of its
    #   layer4 output-conv/downsample group's channels at a 20% GLOBAL target,
    #   while every other layer lost only 2-25%. Empirically verified (see
    #   README's "Architecture-Agnostic Residual Group Detection" section):
    #   this is NOT a defect in Torch-Pruning's own global-pruning algorithm —
    #   a fresh, untrained model with a generic magnitude importance signal
    #   does not reproduce this pattern at all. It comes from how a genuinely
    #   TRAINED residual network behaves: late-stage residual branches often
    #   learn to contribute a small correction relative to the accumulated
    #   identity stream they're added to, and mean-absolute-activation
    #   faithfully measures that small OWN contribution — then gets misread
    #   as "this channel is disposable," when collapsing it actually narrows
    #   the residual stream for the ENTIRE stage, damaging every downstream
    #   block coupled to it, not just one layer's worth of capacity.
    #
    #   We also empirically confirmed that Torch-Pruning's own
    #   max_pruning_ratio KWARG to MetaPruner does NOT reliably hold a
    #   specific vulnerable group under a ceiling by itself — a deliberately
    #   weakened test group still got pruned all the way to the overall
    #   global target, not held near the requested cap. The mechanism that
    #   DOES reliably work is pruning_ratio_dict: explicitly setting a target
    #   ratio per module (or group of modules) so Torch-Pruning's global
    #   solver treats that ratio as fixed for those modules and redistributes
    #   the rest of the budget elsewhere — confirmed to hold a deliberately
    #   weakened group to exactly its requested ratio while the network still
    #   reached a healthy overall reduction.
    #
    # THREE LAYER TYPES TO HANDLE:
    #
    # 1. Output head (last Linear) → ignored_layers
    #    Must not change shape at all — num_classes must match.
    #
    # 2. SE attention layers (SqueezeExcitation children) → ratio=0.0
    #    SE is correctly coupled by torch-pruning (fc1/fc2 move with expand conv),
    #    but under heavy global pressure the SE bottleneck drops to near-zero.
    #    Removed entirely from the global budget.
    #
    # 3. Residual/skip-connection-coupled groups → ratio=effective_residual_ratio
    #    ARCHITECTURE-AGNOSTIC detection via _find_residual_coupled_groups():
    #    walks Torch-Pruning's own dependency graph looking for genuine
    #    elementwise ADD nodes (distinguished from ReLU and other activations,
    #    which Torch-Pruning ALSO classifies under the same ELEMENTWISE
    #    category, by inspecting each node's actual autograd grad_fn class
    #    name). This replaces the old 'block.3' string-match, which only
    #    matched EfficientNet/MobileNet's specific MBConv naming convention
    #    and produced ZERO matches on ResNet, ResNeXt, WideResNet, DenseNet,
    #    ConvNeXt, or any other residual family — confirmed directly: it
    #    printed "0 block.3 layers" on every trial of a real ResNet-50 run.
    #    Verified empirically (see README): finds all 4 of ResNet-50's stage
    #    groups correctly; finds a strict superset of what 'block.3' covered
    #    on EfficientNet-B0 (the same 14 project convs, plus 14 more it was
    #    silently missing); correctly finds ZERO groups on VGG-16 (no skip
    #    connections exist there at all — clean negative control).
    #    Fix: cap each detected group at effective_residual_ratio via
    #    pruning_ratio_dict. This keeps them in the global budget (they ARE
    #    prunable) but prevents catastrophic collapse.

    # 1. Output head
    ignored_layers: list = []
    last_linear: Optional[nn.Module] = None
    for m in pruned.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        ignored_layers.append(last_linear)
        print(f"  Protected — output head   : Linear({last_linear.in_features}, "
              f"{last_linear.out_features})")

    # 2 + 3. SE layers (ratio=0) + residual-coupled groups (ratio=effective_residual_ratio)
    pruning_ratio_dict: dict = {}
    se_count = 0

    for name, m in pruned.named_modules():
        cls = type(m).__name__

        # SE block: set all Conv2d/Linear children to ratio=0
        if 'squeeze' in cls.lower() and 'excit' in cls.lower():
            se_count += 1
            for child in m.modules():
                if isinstance(child, (nn.Conv2d, nn.Linear)):
                    pruning_ratio_dict[child] = 0.0

    print(f"  Protected — SE attention  : {se_count} modules (ratio=0.0)")

    try:
        residual_groups = _find_residual_coupled_groups(pruned, example_input, ignored_layers)
    except TypeError as exc:
        return _torch_pruning_incompatible_result(model, exc)
    residual_member_count = 0
    for group in residual_groups:
        for m in group:
            if m not in pruning_ratio_dict:   # don't override SE=0 if overlap
                pruning_ratio_dict[m] = effective_residual_ratio
                residual_member_count += 1
    print(f"  Protected — residual groups: {len(residual_groups)} coupled group(s) found, "
          f"{residual_member_count} layer(s) capped at ratio≤{effective_residual_ratio:.2f}")

    # ── Apply sensitivity-guided ratios if available ─────────────────────────
    # sensitivity_map contains per-layer importance scores (higher = more important).
    # Convert to per-layer pruning ratios: critical layers get low ratios (conservatively
    # pruned), redundant layers get high ratios (aggressively pruned).
    if sensitivity_map:
        print(f"  [Sensitivity-guided] Converting {len(sensitivity_map)} layer scores to pruning ratios...")
        sensitivity_ratios = sensitivity_to_pruning_ratios(
            sensitivity_map,
            max_pruning_ratio=max_pruning_ratio,
            min_pruning_ratio=0.0,
        )
        # Build lookup: name -> module for LEAF Conv2d/Linear only.
        # named_modules() includes container modules (Sequential, Bottleneck, etc.)
        # whose names can shadow leaf names. Filtering to leaves-only (no children)
        # prevents a container string key leaking into pruning_ratio_dict, which
        # would cause MetaPruner to crash with:
        #   AttributeError: 'str' object has no attribute 'modules'
        name_to_module = {
            name: mod
            for name, mod in pruned.named_modules()
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and not list(mod.children())
        }

        matched = 0
        for layer_name, ratio in sensitivity_ratios.items():
            mod = name_to_module.get(layer_name)
            if mod is not None and isinstance(mod, (nn.Conv2d, nn.Linear)):
                # Don't override SE (ratio=0) or project conv caps already set above
                if mod not in pruning_ratio_dict:
                    pruning_ratio_dict[mod] = ratio
                    matched += 1
        print(f"  [Sensitivity-guided] Matched and applied ratios to "
              f"{matched}/{len(sensitivity_ratios)} layers.")

    # ── Safety guard: purge any non-Module keys from pruning_ratio_dict ──────
    # Unmatched sensitivity names (strings) must never reach MetaPruner —
    # it iterates keys with .modules() and crashes on non-Module objects.
    pruning_ratio_dict = {
        k: v for k, v in pruning_ratio_dict.items()
        if isinstance(k, nn.Module)
    }

    # ── Collect activation statistics via forward hooks ───────────────────────
    # Hooks accumulate mean(|activation|) per channel over calibration batches.
    # try/finally guarantees removal even if a batch fails or an exception occurs.
    print(f"  Collecting activation statistics ({num_calibration_batches} batches)...")
    importance = _ActivationImportance()
    importance.register_hooks(pruned)
    try:
        with torch.no_grad():
            for batch_idx, (X, _) in enumerate(dataloader):
                if batch_idx >= num_calibration_batches:
                    break
                try:
                    pruned(X.to(device))
                except Exception:
                    pass   # non-fatal — partial data is better than no data
    finally:
        importance.remove_hooks()   # always remove — never leave stale hooks
    print(f"  Activation statistics collected for "
          f"{len(importance.get_all_scores())} layers.")

    # ── Build Torch-Pruning MetaPruner ────────────────────────────────────────
    # MetaPruner + custom Importance = global pruning with activation stats.
    # The dependency graph is traced from example_input, handling skip
    # connections, SE blocks, parallel branches automatically.
    # round_to=None passes nothing if unset (MetaPruner handles None gracefully).
    # Swin: register relative_position_bias_table with dim=1 (num_heads)
    # so Torch-Pruning prunes it in sync with attention heads, not by guessing.
    # Raise recursion limit for Swin's deeply nested window-attention graph tracing.
    import sys as _sys

    # ── Detect ViT architecture ───────────────────────────────────────────────
    # ViT has two problems with Torch-Pruning:
    #
    # 1. class_token [1,1,768] and pos_embedding [1,197,768] must be registered
    #    as unwrapped_parameters with dim=2 (the embedding dim) so Torch-Pruning
    #    slices them in sync with conv_proj's output channels.
    #
    # 2. self.hidden_dim is a Python int attribute set to 768 at construction.
    #    _process_input does: x.reshape(n, self.hidden_dim, n_h * n_w)
    #    This is NOT derived from conv_proj.out_channels at runtime — it is a
    #    fixed constant. After pruning, we must patch it to match conv_proj's
    #    new out_channels or every forward pass crashes with a shape error.
    #
    # Solution: register class_token + pos_embedding with dim=2 so Torch-Pruning
    # prunes them correctly, then patch hidden_dim after pruning.step().

    _unwrapped: list = []
    _is_vit = False

    for _name, _param in pruned.named_parameters():
        if 'relative_position_bias_table' in _name:
            # Swin: register with dim=1 (num_heads dimension)
            _unwrapped.append((_param, 1))
        elif _name == 'class_token':
            # ViT: class_token [1, 1, hidden_dim] — prune along dim=2
            _unwrapped.append((_param, 2))
            _is_vit = True
        elif _name == 'encoder.pos_embedding':
            # ViT: pos_embedding [1, num_patches+1, hidden_dim] — prune along dim=2
            _unwrapped.append((_param, 2))
            _is_vit = True

    if _is_vit:
        print(f"  ViT architecture : class_token + pos_embedding registered "
              f"with dim=2 (embedding dim). hidden_dim will be patched after pruning.")

    # Any-architecture attention-head-count guard. EVERY multi-head-attention
    # implementation (torchvision's ViT/Swin, HF's BertSelfAttention,
    # RobertaSelfAttention, AlbertAttention, DistilBertSelfAttention, ...)
    # reshapes its projection output as (..., num_heads, head_dim) at every
    # forward pass, using a FIXED python-int head count read from config --
    # never re-derived from the live tensor shape. If pruning changes the
    # projection's output width to something not divisible by that fixed
    # head count, every forward pass afterward crashes on .view(...).
    # round_to forces Torch-Pruning to only remove channels in multiples of
    # whatever value we give it, so this must be set BEFORE pruning happens.
    # Previously this only ever fired for ViT (checking one attribute name,
    # num_heads) because ViT's Linear layers were the only ones besides
    # Conv2d whose importance actually got collected under the OLD
    # _ActivationImportance hook -- every attention/FFN Linear layer in a
    # sequence model produces a [batch, seq_len, hidden] (3D) output, which
    # that hook silently ignored, so those layers were never selected for
    # pruning and this invariant never had a chance to matter for them. Now
    # that the hook handles 3D (and Swin's channels-last 4D) outputs
    # correctly (see _ActivationImportance.register_hooks), those layers
    # ARE eligible for real pruning, so they need the same protection --
    # just under their own family's attribute name.
    if round_to is None:
        _head_count = None
        _head_count_attr = None
        for _m in pruned.modules():
            for _attr in ('num_heads', 'num_attention_heads', 'n_heads'):
                _nh = getattr(_m, _attr, None)
                if isinstance(_nh, int) and _nh > 1:
                    _head_count, _head_count_attr = _nh, _attr
                    break
            if _head_count is not None:
                break
        if _head_count is not None:
            round_to = _head_count
            print(f"  round_to auto-set to {round_to} (found "
                  f"{type(_m).__name__}.{_head_count_attr}={round_to} -- "
                  f"keeps attention dimensions divisible by the head count "
                  f"so every forward pass afterward stays valid).")

    swin_count = sum(1 for _, d in _unwrapped if d == 1)
    if swin_count:
        print(f"  Swin bias tables : {swin_count} relative_position_bias_table "
              f"params registered (dim=1)")

    _old_rlimit = _sys.getrecursionlimit()
    _sys.setrecursionlimit(max(_old_rlimit, 5000))

    pruner_kwargs: dict = dict(
        model              = pruned,
        example_inputs     = example_input,
        importance         = importance,
        global_pruning     = True,
        pruning_ratio      = pruning_ratio,
        pruning_ratio_dict = pruning_ratio_dict,
        max_pruning_ratio  = max_pruning_ratio,
        ignored_layers     = ignored_layers,
        iterative_steps    = iterative_steps,
        isomorphic         = isomorphic,
    )
    if round_to is not None:
        pruner_kwargs['round_to'] = round_to
    if _unwrapped:
        pruner_kwargs['unwrapped_parameters'] = _unwrapped

    try:
        pruner = tp.pruner.MetaPruner(**pruner_kwargs)
    except TypeError as exc:
        _sys.setrecursionlimit(_old_rlimit)
        return _torch_pruning_incompatible_result(model, exc)
    _sys.setrecursionlimit(_old_rlimit)

    # ── Snapshot original channel counts for the report ───────────────────────
    # Includes BOTH Conv2d (out_channels) and Linear (out_features) so
    # architectures with no Conv2d at all -- any transformer -- get
    # accurate pruned/skipped tracking too. Previously Conv2d-only, which
    # meant this report (and layers_pruned_count downstream, which the
    # search's own "all anchors pruned 0 layers -> give up" early-exit
    # reads) stayed blind to real pruning happening on Linear layers --
    # confirmed directly: with the _ActivationImportance hook fix above
    # making real Linear-layer pruning actually happen, the search's early
    # exit still fired incorrectly on a genuinely-prunable BERT-shaped
    # model, because it was reading this same Conv2d-only count.
    original_ch: dict = {
        name: (mod.out_channels if isinstance(mod, nn.Conv2d) else mod.out_features)
        for name, mod in pruned.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear))
    }

    # ── Execute pruning ───────────────────────────────────────────────────────
    print(f"  Pruning ({iterative_steps} iterative step(s))...")
    for step_idx in range(iterative_steps):
        pruner.step()
        if iterative_steps > 1:
            print(f"    Step {step_idx + 1}/{iterative_steps} done")
    print("  [Pruning] Torch-Pruning step(s) complete.")

    # ── ViT hidden_dim patch ──────────────────────────────────────────────────
    # After pruning, conv_proj.out_channels reflects the new embedding dimension.
    # self.hidden_dim is a Python int used in _process_input's reshape:
    #   x = x.reshape(n, self.hidden_dim, n_h * n_w)
    # It must be updated or every forward pass crashes with a shape error.
    if _is_vit and hasattr(pruned, 'hidden_dim') and hasattr(pruned, 'conv_proj'):
        new_hidden_dim = pruned.conv_proj.out_channels
        old_hidden_dim = pruned.hidden_dim
        if new_hidden_dim != old_hidden_dim:
            pruned.hidden_dim = new_hidden_dim
            print(f"  ViT hidden_dim patched: {old_hidden_dim} → {new_hidden_dim}")

    # ── Generalized attention head-size patch ────────────────────────────────
    # Every multi-head-attention implementation caches num_heads * head_dim
    # as fixed python ints at construction time (all_head_size /
    # attention_head_size / scaling / hidden_size or dim, depending on the
    # family -- see BertSelfAttention, RobertaSelfAttention, AlbertAttention,
    # DistilBertSelfAttention). round_to above keeps the pruned width a
    # multiple of the head count, but does NOT itself update these cached
    # values -- confirmed directly: even with round_to correctly set, a
    # forward pass still reshaped using the STALE, pre-pruning head_dim and
    # crashed with a shape mismatch, because attention_head_size is a fixed
    # int computed once in __init__, never re-derived from the live
    # projection width. This is the direct generalization of the ViT
    # hidden_dim patch immediately above to every other architecture with
    # the same kind of cached, not-auto-derived shape invariant.
    _patched_attn_count = 0
    for _attn_name, _attn_m in pruned.named_modules():
        _head_count = None
        for _hc_attr in ('num_attention_heads', 'num_heads', 'n_heads'):
            _v = getattr(_attn_m, _hc_attr, None)
            if isinstance(_v, int) and _v > 1:
                _head_count = _v
                break
        if _head_count is None or not hasattr(_attn_m, 'attention_head_size'):
            continue
        # First nn.Linear CHILD (not grandchild) of this module is one of
        # its own Q/K/V-style projections -- definition order puts it first
        # in every family checked (BERT/RoBERTa/ALBERT: query; DistilBERT:
        # q_lin) -- and its current out_features is the ground truth for
        # this module's actual post-pruning projection width, however
        # Torch-Pruning's global solver happened to size it.
        _first_linear = next(
            (c for c in _attn_m.children() if isinstance(c, nn.Linear)), None
        )
        if _first_linear is None:
            continue
        _new_all_head_size = _first_linear.out_features
        if _new_all_head_size % _head_count != 0:
            print(f"  ⚠️  {_attn_name}: post-prune width {_new_all_head_size} "
                  f"not divisible by head count {_head_count} -- leaving "
                  f"cached attention_head_size unpatched; this module may "
                  f"still crash on forward.")
            continue
        _new_head_size = _new_all_head_size // _head_count
        if _new_head_size != _attn_m.attention_head_size:
            _attn_m.attention_head_size = _new_head_size
            if hasattr(_attn_m, 'all_head_size'):
                _attn_m.all_head_size = _new_all_head_size
            if hasattr(_attn_m, 'scaling'):
                _attn_m.scaling = _new_head_size ** -0.5
            if hasattr(_attn_m, 'hidden_size') and isinstance(_attn_m.hidden_size, int):
                _attn_m.hidden_size = _new_all_head_size
            if hasattr(_attn_m, 'dim') and isinstance(_attn_m.dim, int):
                _attn_m.dim = _new_all_head_size
            _patched_attn_count += 1
    if _patched_attn_count:
        print(f"  Attention head-size patch: updated {_patched_attn_count} "
              f"module(s) whose cached attention_head_size/all_head_size/"
              f"scaling no longer matched their pruned projection width.")

    # ── Build pruning report layer stats ──────────────────────────────────────
    layers_pruned:  list = []
    layers_skipped: list = []

    for name, mod in pruned.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            continue
        orig_ch = original_ch.get(name)
        if orig_ch is None:
            continue
        new_ch = mod.out_channels if isinstance(mod, nn.Conv2d) else mod.out_features
        if new_ch < orig_ch:
            layers_pruned.append({
                'name':               name,
                'original_channels':  orig_ch,
                'remaining_channels': new_ch,
                'fraction_removed':   (orig_ch - new_ch) / orig_ch,
            })
            print(f"  ✓ {name}: {orig_ch} → {new_ch} filters "
                  f"({(orig_ch - new_ch) / orig_ch * 100:.0f}% removed)")
        else:
            layers_skipped.append(name)

    print(f"\n  Pruned: {len(layers_pruned)}  Skipped: {len(layers_skipped)}")

    # ── Collect importance scores for visualization histogram ─────────────────
    all_scores_map = importance.get_all_scores()
    all_importance_scores: list = []
    for mod in pruned.modules():
        if isinstance(mod, nn.Conv2d) and mod.groups == 1:
            scores = all_scores_map.get(id(mod))
            if scores is not None:
                all_importance_scores.extend(scores.tolist())

    # Approximate the importance threshold: minimum score of kept channels
    # across all pruned layers (the boundary between removed and kept).
    importance_threshold = 0.0
    if layers_pruned and all_importance_scores:
        sorted_scores = sorted(all_importance_scores)
        n_removed     = sum(
            l['original_channels'] - l['remaining_channels'] for l in layers_pruned
        )
        if 0 < n_removed < len(sorted_scores):
            importance_threshold = sorted_scores[n_removed - 1]

    # ── Pre-fine-tune accuracy snapshot (for per-technique impact reporting) ──
    # Captures pruning's own raw structural effect — the accuracy of the
    # pruned-but-not-yet-recovered model — separately from whatever the
    # recovery fine-tune below contributes. Opt-in via
    # capture_pre_finetune_accuracy (see its docstring entry above for why
    # this is a separate flag from test_loader's presence): only measured
    # when the caller explicitly requested it AND at least one layer was
    # actually pruned (if nothing was pruned, this number would just equal
    # whatever the post-fine-tune accuracy turns out to be anyway).
    pre_finetune_accuracy: Optional[float] = None
    if capture_pre_finetune_accuracy and test_loader is not None and layers_pruned:
        from sigularty.helper_functions import measure_accuracy as _measure_accuracy_pft
        pre_finetune_accuracy = _measure_accuracy_pft(pruned, test_loader, device)
        print(f"  [Impact] Pre-fine-tune accuracy: {pre_finetune_accuracy:.2f}% "
              f"(raw structural effect, before recovery fine-tune)")

    # ── Optional fine-tuning — ALWAYS knowledge distillation against `model` ──
    # (this function's own pre-pruning parameter, i.e. the absolute original
    # model, since pruning is always the first technique in the pipeline).
    fine_tune_history: list = []
    if fine_tune_epochs > 0 and layers_pruned:
        pruned, fine_tune_history = _kd_recovery_fine_tune(
            pruned, teacher=model, dataloader=dataloader,
            fine_tune_epochs=fine_tune_epochs, fine_tune_lr=fine_tune_lr,
            device=device, num_classes=num_classes,
            temperature=kd_temperature, alpha=kd_alpha,
            technique_label="Pruning Fine-tune",
            test_loader=test_loader, baseline_accuracy=baseline_accuracy,
            accuracy_drop_threshold=accuracy_drop_threshold,
            early_abort_threshold=early_abort_threshold,
        )

    # ── Behavioral probe ──────────────────────────────────────────────────────
    print("\n  [Pruning] Running behavioral probe...")
    probe_model      = copy.deepcopy(model).to(device)
    behavioral_probe = _run_behavioral_probe(
        probe_model, pruned, dataloader, device,
        num_batches=min(num_calibration_batches, 10),
    )

    # ── Attach pruning report ─────────────────────────────────────────────────
    pruned._pruning_report = {
        'pruning_ratio':         pruning_ratio,
        'model_type':            model_type,
        'layers_pruned':         layers_pruned,
        'layers_skipped':        layers_skipped,
        'all_importance_scores': all_importance_scores,
        'importance_threshold':  importance_threshold,
        'behavioral_probe':      behavioral_probe,
        'fine_tune_epochs':      fine_tune_epochs,
        'fine_tune_history':     fine_tune_history,
        'iterative_steps':       iterative_steps,
        'round_to':              round_to,
        'isomorphic':            isomorphic,
        'original_params':       sum(p.numel() for p in model.parameters()),
        'pruned_params':         sum(p.numel() for p in pruned.parameters()),
        'pre_finetune_accuracy': pre_finetune_accuracy,
    }
    # Top-level attribute (in addition to the report dict above) so
    # apply_compression_pipeline's impact reporting can look this up the
    # same way regardless of which technique produced it — LRF and
    # Clustering attach the identical attribute name on their own reports.
    pruned._pre_finetune_accuracy = pre_finetune_accuracy

    return pruned


# ============================================================================
# ── TECHNIQUE 1: LOW-RANK FACTORIZATION ─────────────────────────────────────
# Theory: every weight matrix W can be approximated as U_r * S_r * V_r^T using
# its top-r singular values.  Replacing one layer with two smaller layers
# reduces FLOPs from O(in*out) to O(in*r + r*out) when r << min(in, out).
# ============================================================================

def _decompose_linear_layer(layer: nn.Linear, epsilon: float) -> nn.Sequential:
    """
    Replace Linear(in, out) with Linear(in, rank) → Linear(rank, out)
    via truncated SVD.

    rank = max(1, int(min(in_features, out_features) * epsilon))

    Args:
        layer:   Source nn.Linear layer (not modified).
        epsilon: Rank ratio in (0.0, 1.0].  Lower → smaller rank → more compression.

    Returns:
        nn.Sequential of two Linear layers whose composition approximates
        the original weight matrix.
    """
    W = layer.weight.data                   # [out_features, in_features]
    rank = max(1, int(min(layer.in_features, layer.out_features) * epsilon))

    # Full SVD on the weight matrix
    U, S, V = torch.svd(W)

    # Truncate to rank-r approximation and absorb singular values into U
    U_r = U[:, :rank] @ torch.diag(S[:rank])   # [out_features, rank]
    V_r = V[:, :rank].t()                       # [rank, in_features]

    layer1 = nn.Linear(layer.in_features, rank, bias=False)
    layer2 = nn.Linear(rank, layer.out_features, bias=True)

    layer1.weight.data = V_r
    layer2.weight.data = U_r

    if layer.bias is not None:
        layer2.bias.data = layer.bias.data

    return nn.Sequential(layer1, layer2)


def _decompose_conv_layer(conv_layer: nn.Conv2d, epsilon: float) -> nn.Sequential:
    """
    Replace Conv2d(in, out, k) with Conv2d(in, rank, k) → Conv2d(rank, out, 1)
    via truncated SVD on the reshaped kernel.

    The kernel W [out_ch, in_ch, kh, kw] is reshaped to [out_ch, in_ch*kh*kw],
    decomposed, then re-folded into two conv kernels.

    NOTE: depthwise and grouped convolutions (groups > 1) are NOT decomposed
    here — the caller skips them before calling this function.

    Args:
        conv_layer: Source nn.Conv2d layer with groups=1 (not modified).
        epsilon:    Rank ratio in (0.0, 1.0].

    Returns:
        nn.Sequential of two Conv2d layers that approximate the original.
    """
    W = conv_layer.weight.data              # [out_ch, in_ch, kh, kw]
    out_ch, in_ch, kh, kw = W.shape
    rank = max(1, int(min(in_ch, out_ch) * epsilon))

    # Reshape to 2-D for SVD: [out_ch, in_ch*kh*kw]
    W_2d = W.view(out_ch, -1)
    U, S, V = torch.svd(W_2d)

    U_r = U[:, :rank] @ torch.diag(S[:rank])   # [out_ch, rank]
    V_r = V[:, :rank].t()                       # [rank, in_ch*kh*kw]

    # Conv1 applies the full-kernel spatial filtering at reduced rank
    conv1 = nn.Conv2d(
        in_ch, rank,
        kernel_size=(kh, kw),
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        bias=False,
    )
    conv1.weight.data = V_r.view(rank, in_ch, kh, kw)

    # Conv2 is a 1×1 conv that mixes the rank channels into the output channels
    conv2 = nn.Conv2d(rank, out_ch, kernel_size=1, bias=conv_layer.bias is not None)
    conv2.weight.data = U_r.view(out_ch, rank, 1, 1)

    if conv_layer.bias is not None:
        conv2.bias.data = conv_layer.bias.data

    return nn.Sequential(conv1, conv2)


def _count_multihead_attention(model: nn.Module) -> int:
    """Count nn.MultiheadAttention modules anywhere in `model`."""
    return sum(1 for m in model.modules() if isinstance(m, nn.MultiheadAttention))


def apply_low_rank_factorization(
    model: nn.Module,
    epsilon: float = 0.5,
    min_layer_size: int = 10,
    min_rank: int = 2,
    skip_large_kernels: bool = False,
    dataloader: Optional[DataLoader] = None,
    device: Optional[str] = None,
    num_classes: Optional[int] = None,
    fine_tune_epochs: int = 0,
    fine_tune_lr: float = 0.0001,
    kd_teacher: Optional[nn.Module] = None,
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.7,
    test_loader: Optional[DataLoader] = None,
    baseline_accuracy: Optional[float] = None,
    accuracy_drop_threshold: Optional[float] = None,
    early_abort_threshold: Optional[float] = None,
    capture_pre_finetune_accuracy: bool = False,
) -> nn.Module:
    """
    Apply SVD-based low-rank factorization to all eligible Conv2d and Linear layers.

    Skipped layers:
      - Any layer with in/out dimensions <= min_layer_size (already small).
      - Any layer where the computed rank < min_rank.
      - Depthwise/grouped convolutions (groups > 1).
      - When skip_large_kernels=True: any Conv2d with kernel size > 1×1.
      - Any layer that is already a Sequential (from prior factorization).
      - The model's output head (final nn.Linear) — see below.

    nn.MultiheadAttention safety (whole-model skip):
      nn.MultiheadAttention.forward() calls F.multi_head_attention_forward()
      with self.out_proj.weight / .bias accessed DIRECTLY as raw tensors —
      it never calls out_proj.forward(), so wrapping out_proj in nn.Sequential
      (what _decompose_linear_layer does) breaks EVERY forward pass with
      AttributeError: 'Sequential' object has no attribute 'weight'.  If any
      nn.MultiheadAttention module exists anywhere in the model, this
      function returns the model COMPLETELY UNCHANGED (a deepcopy, but with
      zero factorization applied anywhere) rather than attempting partial
      factorization elsewhere and crashing on the attention layer.  The
      pipeline-level caller (run_compression_pipeline in helper_functions.py)
      additionally detects this case BEFORE calling LRF at all and disables
      it (and the epsilon search) entirely, to avoid wasting search budget —
      this function's own check is a defensive backstop for direct API use.

    Output-head protection:
      The final nn.Linear in the model (the classification head) is always
      skipped, regardless of epsilon/min_layer_size/min_rank.  This layer is
      typically well under 1% of total parameters — factorizing it saves
      essentially nothing but risks disproportionate accuracy loss, since
      every feature the network has learned funnels through this one
      decision boundary.

    Optional KD recovery fine-tune (always knowledge distillation, never
    plain cross-entropy):
      If fine_tune_epochs > 0 AND dataloader/device/num_classes are all
      provided, the factorized model is fine-tuned via
      _kd_recovery_fine_tune() against kd_teacher (defaults to `model`
      itself — the pre-factorization reference) for fine_tune_epochs epochs.
      If any of dataloader/device/num_classes is missing, fine-tuning is
      skipped with a warning rather than crashing (keeps this function
      usable standalone without forcing every caller to supply training
      infrastructure).

    WHY skip_large_kernels matters for latency:
      Factorizing a Conv2d(C_in, C_out, k×k) into two convolutions:
        Conv2d(C_in, rank, k×k) → Conv2d(rank, C_out, 1×1)
      replaces one CUDA kernel launch with TWO sequential launches.
      On GPU at batch_size=1, each kernel launch has ~0.3-0.5ms fixed overhead.
      For 3×3 convolutions the rank reduction is small (rank ≈ min(C_in, C_out) × ε)
      but the extra kernel launch always pays full overhead cost.
      Net result: latency INCREASES even though parameter count decreases.

      For 1×1 convolutions (pointwise): factorization is beneficial because
        Conv2d(C_in, C_out, 1×1) → Conv2d(C_in, rank, 1×1) → Conv2d(rank, C_out, 1×1)
      each matmul is genuinely smaller, and batch matrix multiplications are
      efficient on GPU Tensor Cores even at small sizes.

      Rule of thumb:
        LRF_SKIP_LARGE_KERNELS = False  for MobileNet/EfficientNet (already mostly 1×1)
        LRF_SKIP_LARGE_KERNELS = True   for ResNet/VGG/DenseNet (heavy 3×3 usage)

    Args:
        model:               Input model. NOT mutated; a deep copy is returned.
        epsilon:             Rank ratio in (0.0, 1.0].
        min_layer_size:      Skip layers with dim <= this.
        min_rank:            Skip layers where computed rank < this.
        skip_large_kernels:  If True, skip all Conv2d with kernel > 1×1.
                             Prevents latency regression on ResNet-style models.
        dataloader:          Training/calibration DataLoader for the optional
                             KD recovery fine-tune.  Required if fine_tune_epochs>0.
        device:              Compute device for the optional fine-tune.
        num_classes:         Output class count for the optional fine-tune's
                             accuracy metric.
        fine_tune_epochs:    KD recovery epochs after factorization.  0 = skip
                             (default — preserves old no-fine-tune behaviour
                             for direct callers who don't opt in).
        fine_tune_lr:        Learning rate for the optional fine-tune.
        kd_teacher:          Frozen KD teacher.  None = use `model` itself.
        kd_temperature:      Softmax temperature for the optional fine-tune.
        kd_alpha:            Hard-label (CE) weight for the optional fine-tune.
        test_loader:         Held-out DataLoader for the early-abort check
                             below. Only used if fine_tune_epochs > 1 AND
                             baseline_accuracy/early_abort_threshold are also
                             supplied — otherwise has no effect.
        baseline_accuracy:   Original (pre-compression) model's accuracy %,
                             for the early-abort check's drop calculation.
        accuracy_drop_threshold: Accepted for API symmetry with the search
                             functions that call this. NOT used directly here
                             (that would make early_abort_threshold a
                             multiplier again) — only early_abort_threshold
                             is compared against.
        early_abort_threshold: Direct pp value. If the fine-tune's epoch-1
                             held-out test-accuracy drop already exceeds
                             this, remaining epochs are skipped. None
                             (default) = disabled, fine-tune always runs to
                             completion. See _kd_recovery_fine_tune's
                             docstring for the full mechanism.
        capture_pre_finetune_accuracy: If True (and test_loader/device are
                             supplied), measures accuracy once right after
                             factorization but before the recovery fine-tune
                             starts, and attaches it as both
                             ._lrf_report['pre_finetune_accuracy'] and
                             ._pre_finetune_accuracy on the returned model
                             (also set, to False/None-equivalents, on the
                             nn.MultiheadAttention unchanged-return path).
                             Used by apply_compression_pipeline's per-
                             technique impact reporting to separate LRF's
                             own structural effect — which is exactly zero
                             on architectures where every eligible layer
                             gets skipped — from its bundled fine-tune's
                             contribution. Default False so this stays a
                             pipeline-only cost: optimization.py's search
                             functions call this function with test_loader
                             already set (for the early-abort check above)
                             but never pass this flag, so search trials
                             never pay for or produce the extra measurement.

    Returns:
        New nn.Module with factorized layers (or an unchanged deepcopy if
        nn.MultiheadAttention was detected).
    """
    _mha_count = _count_multihead_attention(model)
    if _mha_count > 0:
        print(f"\n[Low-Rank Factorization] ⚠️  Detected {_mha_count} "
              f"nn.MultiheadAttention module(s) in this model. LRF cannot "
              f"safely factorize their out_proj Linear layers — "
              f"nn.MultiheadAttention.forward() accesses out_proj.weight/.bias "
              f"directly (bypassing out_proj's own forward() entirely), and "
              f"wrapping out_proj in nn.Sequential breaks that access "
              f"completely. Returning the model UNCHANGED (no factorization "
              f"applied anywhere in this model).\n")
        _unchanged = copy.deepcopy(model)
        _mha_pre_finetune_accuracy: Optional[float] = None
        if capture_pre_finetune_accuracy and test_loader is not None and device is not None:
            from sigularty.helper_functions import measure_accuracy as _measure_accuracy_pft
            _mha_pre_finetune_accuracy = _measure_accuracy_pft(_unchanged, test_loader, device)
        _unchanged._lrf_report = {
            'factorized_count':      0,
            'skipped_count':         None,
            'mha_skip':              True,
            'mha_count':             _mha_count,
            'pre_finetune_accuracy': _mha_pre_finetune_accuracy,
        }
        _unchanged._pre_finetune_accuracy = _mha_pre_finetune_accuracy
        return _unchanged

    _last_linear_name: Optional[str] = None
    for _name, _module in model.named_modules():
        if isinstance(_module, nn.Linear):
            _last_linear_name = _name

    # Structural counters — mirrors the pattern apply_adaptive_lrf already
    # uses for its own _adaptive_factorize walk. Previously this function
    # only ever printed which layers were factorized/skipped and never
    # returned those counts as data, so callers had no way to tell "0
    # eligible layers" apart from "several layers factorized, moderate
    # effect" without parsing console output. Every branch that reaches a
    # leaf Linear/Conv2d decision point increments exactly one of these,
    # including the branches that previously had no print statement at all
    # (e.g. a Linear layer too small to meet min_layer_size) — those are
    # still silently skipped (no new print added, output is unchanged) but
    # now correctly counted.
    factorized_count = 0
    skipped_count    = 0

    def _factorize_module(module: nn.Module, parent_name: str = "") -> None:
        """Recursively walk the module tree and replace eligible layers."""
        nonlocal factorized_count, skipped_count
        for name, child in list(module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            # Skip if already a Sequential (from prior LRF pass)
            if isinstance(child, nn.Sequential):
                # Recurse into Sequential to factorize any nested layers
                _factorize_module(child, full_name)
                continue

            if isinstance(child, nn.Linear):
                if full_name == _last_linear_name:
                    print(f"  Skipping  {full_name}: "
                          f"Linear({child.in_features}, {child.out_features}) — "
                          f"output head, protected from factorization")
                    skipped_count += 1
                elif child.in_features > min_layer_size and child.out_features > min_layer_size:
                    rank = max(1, int(min(child.in_features, child.out_features) * epsilon))
                    if rank < min_rank:
                        print(f"  Skipping  {full_name}: "
                              f"Linear({child.in_features}, {child.out_features}) → "
                              f"rank {rank} < min_rank {min_rank} (would collapse layer)")
                        skipped_count += 1
                    else:
                        print(f"  Factorizing {full_name}: "
                              f"Linear({child.in_features}, {child.out_features}) → rank {rank}")
                        setattr(module, name, _decompose_linear_layer(child, epsilon))
                        factorized_count += 1
                else:
                    skipped_count += 1

            elif isinstance(child, nn.Conv2d):
                # Skip depthwise and grouped convolutions
                if child.groups > 1:
                    print(f"  Skipping  {full_name}: "
                          f"Conv2d({child.in_channels}, {child.out_channels}, "
                          f"groups={child.groups}) — depthwise/grouped, not factorizable")
                    skipped_count += 1
                    continue

                # Skip non-1×1 kernels when requested (prevents latency regression)
                is_pointwise = (child.kernel_size == (1, 1))
                if skip_large_kernels and not is_pointwise:
                    print(f"  Skipping  {full_name}: "
                          f"Conv2d({child.in_channels}, {child.out_channels}, "
                          f"{child.kernel_size}) — kernel>1×1, skip_large_kernels=True")
                    skipped_count += 1
                    continue

                if child.in_channels > min_layer_size and child.out_channels > min_layer_size:
                    rank = max(1, int(min(child.in_channels, child.out_channels) * epsilon))
                    if rank < min_rank:
                        print(f"  Skipping  {full_name}: "
                              f"Conv2d({child.in_channels}, {child.out_channels}) → "
                              f"rank {rank} < min_rank {min_rank} (would collapse layer)")
                        skipped_count += 1
                    else:
                        print(f"  Factorizing {full_name}: "
                              f"Conv2d({child.in_channels}, {child.out_channels}, "
                              f"{child.kernel_size}) → rank {rank}")
                        setattr(module, name, _decompose_conv_layer(child, epsilon))
                        factorized_count += 1
                else:
                    skipped_count += 1

            elif len(list(child.children())) > 0:
                # Recurse into sub-modules (e.g., Sequential, MBConv blocks)
                _factorize_module(child, full_name)

    print("\n[Low-Rank Factorization] Starting factorization...")
    print("  Note: depthwise/grouped convolutions are skipped (groups>1 not factorizable)")
    if _last_linear_name is not None:
        print(f"  Note: output head '{_last_linear_name}' is protected from factorization")
    model_copy = copy.deepcopy(model)
    _factorize_module(model_copy)
    print(f"[Low-Rank Factorization] Done.  Factorized: {factorized_count}  "
          f"Skipped: {skipped_count}\n")

    # ── Pre-fine-tune accuracy snapshot (for per-technique impact reporting) ──
    # Opt-in via capture_pre_finetune_accuracy — see its docstring entry
    # above. This is the number that catches LRF being structurally inert
    # on an architecture (factorized_count == 0) while its bundled fine-
    # tune below still moves accuracy — without this snapshot that movement
    # would be misattributed to LRF itself.
    pre_finetune_accuracy: Optional[float] = None
    if capture_pre_finetune_accuracy and test_loader is not None and device is not None:
        from sigularty.helper_functions import measure_accuracy as _measure_accuracy_pft
        pre_finetune_accuracy = _measure_accuracy_pft(model_copy, test_loader, device)
        print(f"  [Impact] Pre-fine-tune accuracy: {pre_finetune_accuracy:.2f}% "
              f"(raw structural effect, before recovery fine-tune)")

    if fine_tune_epochs > 0:
        if dataloader is not None and device is not None and num_classes is not None:
            _teacher = kd_teacher if kd_teacher is not None else model
            model_copy, _ = _kd_recovery_fine_tune(
                model_copy, teacher=_teacher, dataloader=dataloader,
                fine_tune_epochs=fine_tune_epochs, fine_tune_lr=fine_tune_lr,
                device=device, num_classes=num_classes,
                temperature=kd_temperature, alpha=kd_alpha,
                technique_label="LRF Fine-tune",
                test_loader=test_loader, baseline_accuracy=baseline_accuracy,
                accuracy_drop_threshold=accuracy_drop_threshold,
                early_abort_threshold=early_abort_threshold,
            )
        else:
            print("  ⚠️  [LRF] fine_tune_epochs > 0 but dataloader/device/"
                  "num_classes not all provided — skipping LRF fine-tune.")

    model_copy._lrf_report = {
        'factorized_count':      factorized_count,
        'skipped_count':         skipped_count,
        'mha_skip':              False,
        'pre_finetune_accuracy': pre_finetune_accuracy,
    }
    model_copy._pre_finetune_accuracy = pre_finetune_accuracy

    return model_copy


# ============================================================================
# ── TECHNIQUE 2: WEIGHT CLUSTERING ──────────────────────────────────────────
# Theory: k-means groups the N unique weight values into k centroids.  Each
# weight is replaced by its nearest centroid.  The model can then be stored
# as (k centroids) + (N uint8 indices), giving ~4x compression for k=256.
# Fine-tuning after clustering nudges centroid values toward a better optimum.
# ============================================================================

def _pack_indices_4bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack a 1-D tensor of unsigned cluster-assignment indices in [0, 15] two
    per byte (low nibble = even position, high nibble = odd position).

    Unlike GPTQ's _float_to_int4_packed (which packs SIGNED quantized
    weight VALUES in [-8, 7]), these are plain unsigned k-means cluster
    INDICES — no sign handling needed, just direct nibble packing. A
    separate helper rather than reusing _float_to_int4_packed because the
    two encodings mean genuinely different things (a value vs a lookup key)
    even though the bit-packing mechanics look superficially similar.

    Args:
        indices: 1-D integer tensor, values in [0, 15].

    Returns:
        1-D uint8 tensor of length ceil(len(indices) / 2).
    """
    n      = indices.numel()
    idx_u8 = indices.to(torch.uint8)
    packed = torch.zeros((n + 1) // 2, dtype=torch.uint8, device=indices.device)

    even = idx_u8[0::2]
    packed[: even.numel()] = even & 0x0F
    if n > 1:
        odd = idx_u8[1::2]
        packed[: odd.numel()] |= (odd & 0x0F) << 4
    return packed


def _unpack_indices_4bit(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    """
    Reverse of _pack_indices_4bit.

    Returns an int64 tensor of length n_elements with values in [0, 15],
    ready to use as an index tensor into a centroids tensor — advanced
    indexing requires an integer dtype, and int64 is what PyTorch expects
    by default.
    """
    lo  = packed & 0x0F
    hi  = (packed >> 4) & 0x0F
    out = torch.empty(2 * packed.numel(), dtype=torch.uint8, device=packed.device)
    out[0::2] = lo
    out[1::2] = hi
    return out[:n_elements].to(torch.int64)


class _ClusteredWeightBase(nn.Module):
    """
    Shared machinery for weight-shared ("trained quantization") layers —
    subclassed by _ClusteredLinear and _ClusteredConv2d below. Handles
    centroid/assignment storage, weight reconstruction, and eval-mode
    caching identically for both; only forward() and the layer-shape
    constructor arguments differ between the two subclasses.

    Replaces the old apply_weight_clustering behaviour of writing centroid
    values back as a plain float32 tensor. That old approach had two
    compounding problems, both fixed here:

    1. get_model_size_mb (numel() * element_size()) could never see any
       reduction from it — clustered weights were still N float32 values,
       same dtype, same count, just restricted to k distinct values. Zero
       measured compression, regardless of how small k was.
    2. The very next step in the real pipeline is always an unconstrained
       KD fine-tune (plain per-parameter Adam). With nothing tying
       same-centroid weights together, every individual weight drifted
       independently — the k-value structure was almost certainly gone
       within a few epochs, well before GPTQ or quantization ever saw it.

    The fix: `centroids` (length k) is the ONLY trainable piece; the
    per-weight `assignment` (which centroid each original weight position
    uses) is a FIXED, non-trainable buffer, packed 4-bit (k<=16) or plain
    uint8 (k>16). The weight is reconstructed each forward call as
    `centroids[assignment]`, reshaped to the original shape.

    Why fine-tuning now preserves the k-value structure: backprop through
    `centroids[assignment]` (PyTorch advanced indexing) naturally SUMS the
    gradient contribution from every weight position sharing a centroid —
    this is the exact same mechanism that already makes nn.Embedding train
    correctly when a token ID repeats within a batch. No custom backward
    code is needed for this; it's a free consequence of how indexing's
    backward pass works, not something bolted on here.

    Why get_model_size_mb now sees a real reduction: `centroids` is k
    float32 values (tiny — 16 floats is 64 bytes) and `assignment` is
    packed into k<=16 ? 4 bits : 8 bits per original weight element, both
    real tensors, both counted (get_model_size_mb sums parameters() AND
    buffers()).

    GPTQ compatibility: exposes `.weight` (property, reconstructed dense
    tensor), `.bias` (real nn.Parameter, never clustered), and whatever
    shape attributes the subclass needs (in_features/out_features for
    Linear; in_channels/out_channels/kernel_size/etc. for Conv2d) — the
    same attribute names apply_gptq_quantization already expects from a
    plain nn.Linear. This means GPTQ can run AFTER clustering exactly as
    the fixed pipeline order already has it: it reads a real dense weight
    from this layer, Hessian-corrects it, and replaces the whole layer
    with a fresh _Int4Linear — clustering acts as a pre-conditioning /
    denoising step for whatever GPTQ does next on Linear layers, not a
    competing claim on the same layers. For Conv2d (which GPTQ never
    touches — it's a Linear-only, per-column Hessian algorithm), this
    compact storage IS the final, lasting compression.
    """

    def _init_clustered(
        self,
        centroids: torch.Tensor,
        assignment: torch.Tensor,
        weight_shape: tuple,
        bias: Optional[torch.Tensor],
    ) -> None:
        self.num_clusters  = int(centroids.numel())
        self._weight_shape = tuple(weight_shape)
        self._n_elements   = int(assignment.numel())
        self._use_4bit     = self.num_clusters <= 16

        self.centroids = nn.Parameter(centroids.clone().float())
        if self._use_4bit:
            self.register_buffer('assignment_packed', _pack_indices_4bit(assignment))
        else:
            self.register_buffer('assignment_packed', assignment.to(torch.uint8))
        # Real, trainable parameter — bias is never clustered. MUST be
        # wrapped in nn.Parameter explicitly: the `bias` argument here has
        # already been through `.detach().clone()` at the call site (see
        # _cluster_single_layer), which returns a plain torch.Tensor, not
        # an nn.Parameter — plain attribute assignment of a plain Tensor
        # does NOT auto-register it in self._parameters (unlike
        # _Int8Linear, which assigns the LIVE bias Parameter straight
        # through with no detach/clone, so it keeps its subclass and gets
        # auto-registered for free). An unregistered bias silently never
        # received gradients despite this comment's original claim, and
        # silently failed to follow a whole-model dtype cast — invisible
        # in ordinary use (forward() below re-casts .to(x.dtype) on every
        # call) but fatal the moment something reads .weight/.bias
        # directly without going through forward(), which is exactly what
        # nn.MultiheadAttention.forward() does for its out_proj sub-layer.
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
        self._weight_cache: Optional[torch.Tensor] = None

    def _reconstruct_weight(self) -> torch.Tensor:
        if self._use_4bit:
            idx = _unpack_indices_4bit(self.assignment_packed, self._n_elements)
        else:
            idx = self.assignment_packed.to(torch.int64)
        return self.centroids[idx].view(self._weight_shape)

    def _get_weight(self) -> torch.Tensor:
        """
        Return the reconstructed weight, using a cache in eval mode.

        Same caching contract as _Int4Linear: in eval mode, build once and
        reuse until training resumes or the module moves device/dtype; in
        train mode, always reconstruct fresh (centroids change every
        optimizer step) and leave the cache empty so the next eval() call
        is guaranteed to rebuild from the current centroids rather than
        serving something stale from before this training round.

        Invalidates on a dtype change as well as a device change. Found
        alongside the bias-registration fix above: centroids (a real
        Parameter) always tracks a whole-model .to(dtype) call correctly,
        but a cache warmed BEFORE that call — e.g. by an accuracy/latency
        measurement taken right after clustering, whose model object then
        gets deepcopied into a later quantization step, cache and all —
        would otherwise keep serving the pre-cast dtype indefinitely,
        since only .device was ever compared here. A stale-dtype cache
        reproduces the same class of mismatch the bias bug did, just
        through the cache instead of through non-registration.
        """
        if not self.training:
            if (self._weight_cache is not None
                    and (self._weight_cache.device != self.centroids.device
                         or self._weight_cache.dtype != self.centroids.dtype)):
                self._weight_cache = None
            if self._weight_cache is None:
                self._weight_cache = self._reconstruct_weight()
            return self._weight_cache
        self._weight_cache = None
        return self._reconstruct_weight()

    def train(self, mode: bool = True):
        """Override train() to invalidate the weight cache when switching modes."""
        if mode:
            self._weight_cache = None
        return super().train(mode)

    @property
    def weight(self) -> torch.Tensor:
        """
        Reconstructed (centroid-expanded) weight tensor, exposed as
        .weight — required because nn.MultiheadAttention.forward directly
        accesses self.out_proj.weight as a raw tensor argument to
        F.multi_head_attention_forward, bypassing out_proj.forward()
        entirely (same reasoning as _Int4Linear.weight), and because
        apply_gptq_quantization reads a real dense .weight from whatever
        layer it's about to Hessian-correct.
        """
        return self._get_weight()


class _ClusteredLinear(_ClusteredWeightBase):
    """Weight-shared Linear layer — see _ClusteredWeightBase for the full mechanism."""

    def __init__(
        self,
        centroids: torch.Tensor,
        assignment: torch.Tensor,
        weight_shape: tuple,
        bias: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self._init_clustered(centroids, assignment, weight_shape, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.linear(x, w, b)

    def extra_repr(self) -> str:
        cached = self._weight_cache is not None
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, k={self.num_clusters}, '
                f'packing={"4bit" if self._use_4bit else "uint8"}, '
                f'cache={"warm" if cached else "cold"}')


class _ClusteredConv2d(_ClusteredWeightBase):
    """
    Weight-shared Conv2d layer — see _ClusteredWeightBase for the full
    mechanism. Conv2d is never touched by GPTQ (a Linear-only, per-column
    Hessian algorithm), so for Conv2d layers this compact storage IS the
    final, lasting compression — unlike Linear layers, which may go on to
    be replaced by GPTQ's own INT4 packing afterward.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        assignment: torch.Tensor,
        weight_shape: tuple,        # (out_ch, in_ch/groups, kh, kw)
        bias: Optional[torch.Tensor],
        stride, padding, dilation, groups: int,
    ) -> None:
        super().__init__()
        self.out_channels = weight_shape[0]
        self.in_channels   = weight_shape[1] * groups
        self.kernel_size   = (weight_shape[2], weight_shape[3])
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation
        self.groups   = groups
        self._init_clustered(centroids, assignment, weight_shape, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return nn.functional.conv2d(
            x, w, b, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )

    def extra_repr(self) -> str:
        cached = self._weight_cache is not None
        return (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
                f'stride={self.stride}, padding={self.padding}, groups={self.groups}, '
                f'bias={self.bias is not None}, k={self.num_clusters}, '
                f'packing={"4bit" if self._use_4bit else "uint8"}, '
                f'cache={"warm" if cached else "cold"}')


def _get_nested_layer(model: nn.Module, dotted_name: str) -> nn.Module:
    """
    Traverse a dotted attribute path to retrieve a nested sub-module.

    Example: _get_nested_layer(model, 'layer1.0.conv1')
    """
    layer = model
    for part in dotted_name.split('.'):
        layer = getattr(layer, part)
    return layer


def _cluster_single_layer(
    model: nn.Module,
    layer_name: str,
    num_clusters: int,
) -> Optional[dict]:
    """
    Apply k-means weight clustering to one layer, REPLACING it in-place
    within `model`'s module tree with a _ClusteredLinear / _ClusteredConv2d.

    This replaces the old behaviour of writing centroid values back as a
    plain float32 tensor (same dtype, same element count — get_model_size_mb
    could never see any reduction from that, and the very next unconstrained
    fine-tune step would silently un-cluster every weight independently,
    since nothing tied same-centroid weights together — see
    _ClusteredWeightBase's docstring for the full explanation of both
    problems and the fix). The number of clusters is clamped to
    min(num_clusters, unique_weight_values) to handle layers that already
    have very few distinct values.

    Args:
        model:        Model containing the layer (modified in-place — the
                      target layer is REPLACED in its parent module, not
                      mutated).
        layer_name:   Dotted attribute path to the target layer.
        num_clusters: Maximum k for k-means.

    Returns:
        Dict with real compression info, or None if the layer is
        ineligible (already wrapped by an earlier technique, no weight,
        fewer than 1 distinct value, unsupported type).
    """
    parts  = layer_name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    layer = getattr(parent, parts[-1])

    # Skip if layer is Sequential (from LRF) or doesn't have weight
    if isinstance(layer, nn.Sequential) or not hasattr(layer, 'weight') or layer.weight is None:
        return None
    # Only plain, un-touched Conv2d/Linear are eligible — a layer already
    # wrapped by an earlier technique (shouldn't happen given the fixed
    # pipeline order, since Clustering runs before GPTQ/Quantization, but
    # guards direct/library callers who invoke this out of order) is left
    # alone rather than clustering something that isn't a real dense layer.
    if not isinstance(layer, (nn.Conv2d, nn.Linear)):
        return None

    try:
        w = layer.weight.data.detach()
        shape = tuple(w.shape)
        flat_np = w.cpu().numpy().flatten().reshape(-1, 1)
    except (AttributeError, RuntimeError):
        return None

    unique_vals = len(np.unique(flat_np))
    k = min(num_clusters, unique_vals)
    if k < 1:
        return None

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(flat_np)

    centroids  = torch.from_numpy(km.cluster_centers_.flatten()).float()
    assignment = torch.from_numpy(km.labels_.astype(np.int64))
    bias = layer.bias.detach().clone() if getattr(layer, 'bias', None) is not None else None

    if isinstance(layer, nn.Linear):
        new_layer = _ClusteredLinear(
            centroids=centroids, assignment=assignment, weight_shape=shape,
            bias=bias, in_features=layer.in_features, out_features=layer.out_features,
        )
    else:  # nn.Conv2d
        new_layer = _ClusteredConv2d(
            centroids=centroids, assignment=assignment, weight_shape=shape,
            bias=bias, stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, groups=layer.groups,
        )
    new_layer = new_layer.to(layer.weight.device)
    setattr(parent, parts[-1], new_layer)

    bits_per_index = 4 if k <= 16 else 8
    real_ratio = 32 / bits_per_index   # real storage ratio vs float32 for the
                                        # index array — the k centroids
                                        # themselves are a fixed, tiny
                                        # overhead (e.g. 16 floats = 64
                                        # bytes) not worth folding into this
    print(f"  ✓ {layer_name}: shape={shape}  k={k}  "
          f"packing={'4-bit' if k <= 16 else 'uint8'}  "
          f"real storage ratio≈{real_ratio:.1f}x  "
          f"(unique values before clustering: {unique_vals})")

    return {
        'n_clusters':        k,
        'weight_shape':      shape,
        'original_unique':   unique_vals,
        # NOW a real, measured storage ratio (32 / bits_per_index), not the
        # old unique_vals/k figure — that number was purely about how many
        # DISTINCT VALUES existed, which never translated into fewer bytes
        # since the old code stored every value as a full float32 regardless.
        'compression_ratio': real_ratio,
    }



def _fine_tune_after_clustering(
    model: nn.Module,
    dataloader: DataLoader,
    fine_tune_epochs: int,
    fine_tune_lr: float,
    device: str,
    num_classes: int,
) -> nn.Module:
    """
    Fine-tune a clustered model for a small number of epochs to recover accuracy.

    Plain cross-entropy fallback path — only reached when apply_weight_clustering
    is called WITHOUT a kd_teacher.  In the real product pipeline kd_teacher is
    always supplied (the original model), so this branch is not exercised there;
    it exists for direct/library use.

    Uses a lower learning rate than pre-training so centroid values are nudged
    rather than fully re-learned (which would destroy the clustering benefit).

    Includes the same epoch-1-exactly-0.0%-accuracy safety check used by every
    other fine-tune loop in this file.

    Args:
        model:             Clustered model to fine-tune (modified in-place).
        dataloader:        Training data loader.
        fine_tune_epochs:  Number of training epochs.
        fine_tune_lr:      Learning rate.
        device:            'cuda' or 'cpu'.
        num_classes:       Number of output classes (for torchmetrics).

    Returns:
        Fine-tuned model (same object, returned for chaining).
    """
    print(f"\n  [Fine-tune] {fine_tune_epochs} epoch(s) at lr={fine_tune_lr}")
    model.to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)
    loss_fn     = nn.CrossEntropyLoss()
    accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

    model.train()
    for epoch in tqdm(range(fine_tune_epochs), desc="  Fine-tune"):
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc  += accuracy_fn(logits.argmax(1), y).item()
            n_batches  += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        avg_acc  = (total_acc  / n_batches) * 100 if n_batches > 0 else 0.0
        print(f"    Epoch {epoch + 1}/{fine_tune_epochs}  "
              f"loss={avg_loss:.4f}  acc={avg_acc:.2f}%")

        if epoch == 0 and avg_acc == 0.0:
            print(
                f"\n  🚨 [Clustering Fine-tune] MAJOR WARNING: epoch 1 produced "
                f"EXACTLY 0.0% accuracy — this is almost always a structural "
                f"break, not slow convergence. Skipping remaining "
                f"{fine_tune_epochs - 1} epoch(s).\n"
            )
            break

    print("  [Fine-tune] Done.")
    return model


def apply_weight_clustering(
    model: nn.Module,
    dataloader: DataLoader,
    num_clusters: int = 16,
    fine_tune_epochs: int = 5,
    fine_tune_lr: float = 0.0001,
    device: str = "cpu",
    num_classes: int = 10,
    layers_to_cluster: Optional[List[str]] = None,
    kd_teacher: Optional[nn.Module] = None,   # if set, uses KD instead of CE fine-tune
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.7,
    test_loader: Optional[DataLoader] = None,
    capture_pre_finetune_accuracy: bool = False,
) -> nn.Module:
    """
    Apply k-means weight clustering to all Conv2d and Linear layers.

    Each layer is REPLACED with a _ClusteredLinear / _ClusteredConv2d
    (see _ClusteredWeightBase's docstring for the full mechanism) — real,
    trainable centroids plus a fixed, compactly-packed per-weight
    assignment. This is a real behaviour change from clustering's old
    approach of writing centroid values back as plain float32: that
    version produced zero measured size reduction (get_model_size_mb
    couldn't see it — same dtype, same element count) and its k-value
    structure was silently destroyed by the very next unconstrained
    fine-tune step, before GPTQ or quantization ever saw it. Optionally
    fine-tunes the clustered model to recover lost accuracy — because
    `centroids` is the ONLY trainable piece now (assignment is fixed),
    this fine-tune can no longer un-cluster the model; gradients for
    same-centroid weights are naturally summed by PyTorch's own indexing
    backward pass, the same mechanism nn.Embedding already relies on.

    Compression intuition (now real, not theoretical):
      k<=16  → weights stored as 4-bit packed indices → 8x storage vs float32
      k>16   → weights stored as uint8 indices        → 4x storage vs float32
      (plus a fixed, tiny k-centroid table — e.g. k=16 is 16 floats = 64 bytes,
      negligible next to the index savings on any layer worth clustering)

    GPTQ compatibility: for Linear layers that ALSO go through GPTQ
    afterward (the fixed pipeline order runs Clustering before GPTQ),
    _ClusteredLinear exposes .weight/.bias/.in_features/.out_features
    exactly like a real nn.Linear, so apply_gptq_quantization reads a real
    dense weight from it, Hessian-corrects it, and replaces the whole
    layer with a fresh _Int4Linear — clustering's role for those layers
    becomes pre-conditioning (a smoother, fine-tuned starting point for
    GPTQ's own correction), not a competing claim on final storage. For
    Conv2d layers (which GPTQ never touches — it's a Linear-only, per-
    column Hessian algorithm), _ClusteredConv2d's compact storage IS the
    final, lasting compression.

    Args:
        model:             Input model. NOT mutated; a deep copy is returned.
        dataloader:        DataLoader used for fine-tuning (ignored if
                           fine_tune_epochs == 0).
        num_clusters:      k for k-means.  Try 8, 16, 32, 64, 128, 256.
        fine_tune_epochs:  Epochs of fine-tuning after clustering.  0 = skip.
        fine_tune_lr:      Learning rate for fine-tuning.
        device:            Device for fine-tuning ('cuda' or 'cpu').
        num_classes:       Number of output classes (used by fine-tune accuracy).
        layers_to_cluster: Explicit list of dotted layer names to cluster.
                           None = automatically cluster all Conv2d and Linear layers.
        kd_teacher:        If provided, fine-tuning uses knowledge distillation
                           against this frozen teacher (always preferred — the
                           real pipeline always supplies the original model
                           here).  If None, falls back to plain cross-entropy
                           via _fine_tune_after_clustering().
        test_loader:       Optional held-out DataLoader. Only consumed when
                           capture_pre_finetune_accuracy=True (see below) —
                           unlike apply_structured_pruning / apply_low_rank_
                           factorization, this function has no early-abort
                           mechanism of its own, so test_loader has exactly
                           one job here.
        capture_pre_finetune_accuracy: If True (and test_loader is supplied),
                           measures accuracy once right after clustering but
                           before the recovery fine-tune starts, and attaches
                           it as both ._clustering_report['pre_finetune_
                           accuracy'] and ._pre_finetune_accuracy on the
                           returned model. Used by apply_compression_
                           pipeline's per-technique impact reporting to
                           separate clustering's own structural effect from
                           its bundled fine-tune's contribution. Default
                           False so this stays a pipeline-only cost.

    Returns:
        New nn.Module with clustered layers (real _ClusteredLinear /
        _ClusteredConv2d, optionally fine-tuned).
    """
    clustered = copy.deepcopy(model)

    if layers_to_cluster is None:
        layers_to_cluster = [
            name for name, mod in clustered.named_modules()
            if isinstance(mod, (nn.Conv2d, nn.Linear))
        ]

    print(f"\n[Weight Clustering] Clustering {len(layers_to_cluster)} layers "
          f"with k={num_clusters}...")

    layer_details: dict = {}
    for name in layers_to_cluster:
        info = _cluster_single_layer(clustered, name, num_clusters)
        if info is not None:
            layer_details[name] = info

    if layer_details:
        avg_ratio = float(np.mean([v['compression_ratio'] for v in layer_details.values()]))
        print(f"[Weight Clustering] Done.  Layers clustered: {len(layer_details)}, "
              f"avg weight-value compression: {avg_ratio:.2f}x")
    else:
        print("[Weight Clustering] No eligible layers found.")

    # ── Pre-fine-tune accuracy snapshot (for per-technique impact reporting) ──
    # Opt-in via capture_pre_finetune_accuracy — see its docstring entry
    # above. measure_accuracy() moves the model to `device` internally, so
    # no explicit .to(device) call is needed here first. Captured as a
    # local variable (not attached to `clustered` yet) because
    # fine_tune_with_distillation below REASSIGNS clustered to a brand new
    # deep-copied object rather than mutating in place — attaching the
    # report at the very end, after that reassignment has already
    # happened, avoids depending on whether any particular fine-tune path
    # mutates in place or not.
    pre_finetune_accuracy: Optional[float] = None
    if capture_pre_finetune_accuracy and test_loader is not None and layer_details:
        from sigularty.helper_functions import measure_accuracy as _measure_accuracy_pft
        pre_finetune_accuracy = _measure_accuracy_pft(clustered, test_loader, device)
        print(f"  [Impact] Pre-fine-tune accuracy: {pre_finetune_accuracy:.2f}% "
              f"(raw structural effect, before recovery fine-tune)")

    if fine_tune_epochs > 0:
        if kd_teacher is not None:
            print(f"  [Clustering] Using KD fine-tune (T={kd_temperature:.1f}, α={kd_alpha:.2f})")
            clustered = fine_tune_with_distillation(
                student=clustered,
                teacher=kd_teacher,
                dataloader=dataloader,
                num_classes=num_classes,
                epochs=fine_tune_epochs,
                lr=fine_tune_lr,
                device=device,
                temperature=kd_temperature,
                alpha=kd_alpha,
            )
        else:
            clustered = _fine_tune_after_clustering(
                clustered,
                dataloader=dataloader,
                fine_tune_epochs=fine_tune_epochs,
                fine_tune_lr=fine_tune_lr,
                device=device,
                num_classes=num_classes,
            )

    clustered._clustering_report = {
        'num_clusters':          num_clusters,
        'layers_clustered':      len(layer_details),
        'layer_details':         layer_details,
        'pre_finetune_accuracy': pre_finetune_accuracy,
    }
    clustered._pre_finetune_accuracy = pre_finetune_accuracy

    return clustered


# ============================================================================
# ── TECHNIQUE 3: QUANTIZATION ────────────────────────────────────────────────
# Theory:
#   Dynamic  — scales activations at runtime; only Linear weights go to INT8.
#              Zero calibration cost.  Best for RNN/Transformer-heavy models.
#   Static   — pre-computes activation scales from calibration data; more
#              aggressive.  Requires representative batches. CPU only (fbgemm).
#   FP16     — casts all parameters to float16.  Halves memory footprint.
#              Needs GPU for actual speedup (FP16 CUDA cores).
# ============================================================================

def _manual_dynamic_quantize(model: nn.Module) -> nn.Module:
    """
    Manually quantize all nn.Linear weights to INT8 and store the float scale.

    Why this exists instead of torch.ao.quantization.quantize_dynamic:
      quantize_dynamic in PyTorch 2.x raises
        "apply_dynamic is not implemented for this packed parameter type"
      when the model contains nn.Sequential wrappers around Linear layers —
      which is exactly what LRF produces. The PyTorch dispatch path does not
      recognise those container types.

    This function does the same thing quantize_dynamic does for Linear layers:
      1. Walk every nn.Linear leaf in the module tree.
      2. Compute per-tensor scale = max(|W|) / 127.
      3. Store weight as torch.int8 in a new parameter.
      4. Store scale as a float buffer.
      5. Override forward() to dequantize on the fly before the matmul.

    Result: weights occupy 1 byte per element (vs 4 for float32), inference
    dequantizes to float32 just before each Linear computation — identical
    behaviour to dynamic quant, ~4x weight memory reduction on Linear layers.

    Args:
        model: float32 model (deep-copied by the caller before this is called).

    Returns:
        Same model object with all nn.Linear layers replaced by _Int8Linear.
    """
    class _Int8Linear(nn.Module):
        """Drop-in replacement for nn.Linear with INT8-stored weights."""
        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            w = linear.weight.data.float()
            scale = w.abs().max() / 127.0
            scale = scale.clamp(min=1e-8)           # avoid div-by-zero
            w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
            # Store as non-trainable buffers so state_dict serialises them
            # and .to(device) moves them correctly.
            self.register_buffer('weight_int8', w_int8)
            self.register_buffer('scale', scale.reshape(1))
            self.bias         = linear.bias           # keep original bias (float32)
            self.in_features  = linear.in_features
            self.out_features = linear.out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Dequantize weight back to float32 for the actual matmul.
            # Memory footprint is INT8 at rest; compute is float32.
            w_float = self.weight_int8.float() * self.scale
            return nn.functional.linear(x, w_float, self.bias)

        def extra_repr(self) -> str:
            return (f'in_features={self.in_features}, '
                    f'out_features={self.out_features}, '
                    f'scale={self.scale.item():.6f}')

    def _replace_linears(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            # Recognizes _ClusteredLinear alongside nn.Linear: a Linear
            # layer that went through Clustering but was never subsequently
            # replaced by GPTQ (e.g. USE_GPTQ=False) would otherwise be
            # silently skipped here — _Int8Linear's constructor already
            # works unchanged on it (reads .weight/.bias/.in_features/
            # .out_features, all of which _ClusteredLinear exposes exactly
            # like a real nn.Linear).
            if isinstance(child, (nn.Linear, _ClusteredLinear)):
                setattr(module, name, _Int8Linear(child))
            else:
                _replace_linears(child)     # recurse into Sequential, Bottleneck, etc.

    _replace_linears(model)
    return model


class _QuantWrapper(nn.Module):
    """
    Thin wrapper that injects QuantStub/DeQuantStub around a model for
    static quantization preparation and conversion.
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant   = torch.ao.quantization.QuantStub()
        self.model   = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def _apply_static_quantization(
    model: nn.Module,
    calibration_loader: DataLoader,
    num_calibration_batches: int,
) -> nn.Module:
    """
    Apply static INT8 quantization with QuantStub/DeQuantStub calibration.

    Compatibility note:
    Models with residual additions (plain Python '+') crash after static
    quantization because tensors become QuantizedCPU dtype, but 'aten::add'
    has no QuantizedCPU kernel:
        RuntimeError: Could not run 'aten::add.out' from 'QuantizedCPU' backend.
    Fixing this requires replacing every '+' with FloatFunctional, which means
    rewriting the architecture to use FloatFunctional for residual additions.

    Workaround: attach a dynamic-quantized copy as ._eval_model.
    - The static INT8 wrapped model is kept for SIZE measurement
      (Conv2d/Linear weights are genuinely INT8 after conversion).
    - measure_accuracy() and measure_latency() detect ._eval_model and
      use it instead — dynamic INT8 only touches Linear layers so
      residual additions stay float32 and work fine.

    Args:
        model:                   Float32 model on CPU.
        calibration_loader:      DataLoader providing representative inputs.
        num_calibration_batches: Number of calibration batches.

    Returns:
        Static INT8 model (for size) with ._eval_model set (for inference).
    """
    # Dynamic-quantized copy for safe inference (residual-compatible).
    # Uses _manual_dynamic_quantize instead of quantize_dynamic because
    # the latter crashes on nn.Sequential wrappers left by LRF.
    eval_model = _manual_dynamic_quantize(copy.deepcopy(model))

    # Static INT8 model for accurate size measurement
    wrapped = _QuantWrapper(model)
    wrapped.eval()
    wrapped.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    torch.ao.quantization.prepare(wrapped, inplace=True)

    print(f"  [Static] Calibrating on {num_calibration_batches} batches...")
    with torch.no_grad():
        for i, (X, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            wrapped(X)

    torch.ao.quantization.convert(wrapped, inplace=True)
    print("  [Static] INT8 conversion done.")
    print("  [Static] Attaching dynamic-quant eval model (residual-safe).")

    wrapped._eval_model = eval_model
    return wrapped


def apply_quantization(
    model: nn.Module,
    dataloader: DataLoader,
    mode: str = "dynamic",
    num_calibration_batches: int = 100,
) -> nn.Module:
    """
    Reduce model parameter precision via quantization.

    Modes:
      "dynamic"  — INT8 dynamic quantization of all nn.Linear layers.
                   No calibration data needed.  Works on CPU and GPU.
                   After LRF the nested Linear layers inside Sequential
                   are still found and quantized recursively.
      "static"   — INT8 static quantization with observer-based calibration.
                   Wraps the model in _QuantWrapper for QuantStub injection.
                   Requires CPU (fbgemm backend).
      "fp16"     — Cast all parameters to torch.float16.
                   Halves in-memory size.  Full speedup requires a GPU with
                   native FP16 support (Pascal or later).

    Stacking note: quantization must always be applied LAST.  Applying it
    before LRF or clustering would invalidate the float32 assumptions of
    SVD and k-means.

    Args:
        model:                   Input model. NOT mutated; a deep copy is returned.
        dataloader:              Calibration DataLoader (used only for "static").
        mode:                    One of "dynamic", "static", "fp16".
        num_calibration_batches: Number of calibration batches for "static" mode.

    Returns:
        Quantized model.

    Raises:
        ValueError: If mode is not one of the supported values.
    """
    valid_modes = {"dynamic", "static", "fp16"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    try:
        torch.backends.quantized.engine = 'fbgemm'
    except Exception:
        pass  # non-CPU environments may not support fbgemm; proceed anyway

    print(f"\n[Quantization] Applying '{mode}' quantization...")
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    if mode == "dynamic":
        # torch.ao.quantization.quantize_dynamic raises
        #   "apply_dynamic is not implemented for this packed parameter type"
        # when the model went through LRF, because LRF wraps layers in
        # nn.Sequential and PyTorch 2.x dispatch can't handle those containers.
        # Fix: manual walk — same end result, no broken dispatch.
        quantized = _manual_dynamic_quantize(model_copy)

    elif mode == "static":
        model_copy = model_copy.cpu()   # fbgemm backend requires CPU
        quantized  = _apply_static_quantization(
            model_copy, dataloader, num_calibration_batches
        )

    else:  # "fp16"
        # torch.float16 = FP16.  Note: CPU inference with FP16 is unsupported
        # on most platforms; move to CUDA for actual speedup.
        quantized = model_copy.to(torch.float16)

    print(f"[Quantization] '{mode}' quantization done.\n")
    return quantized


# ============================================================================
# ── UNIFIED COMPRESSION PIPELINE ────────────────────────────────────────────
# Applies enabled techniques in the mandatory order:
#   BN Fusion → Sensitivity Analysis → Pruning → LRF → Clustering →
#   (in-pipeline KD, dormant in the real product) → GPTQ → Quantization
#
# Each technique operates on the output of the previous one.
#
# Per-technique accuracy gating (test_loader/accuracy_drop_threshold):
#   When test_loader is supplied, every MUTATING technique (Pruning, LRF,
#   Clustering, the dormant in-pipeline KD step, GPTQ, Quantization) is
#   measured before and after it runs.  If its OWN marginal accuracy drop
#   exceeds accuracy_drop_threshold, that technique's result is discarded
#   and the pipeline reverts to the pre-technique model — each technique
#   gets an independent budget, not a shared/cumulative one.  BN Fusion and
#   Sensitivity Analysis are never gated (lossless / read-only respectively).
#   The kept/reverted outcome of every gated step is recorded in
#   `._gating_report` on the returned model; the final accuracy (after
#   whichever techniques survived) is `._gating_accuracy`.
# ============================================================================

def apply_compression_pipeline(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    # ── Technique enable/disable flags ──────────────────────────────────────
    use_bn_fusion: bool   = True,
    use_sensitivity: bool = False,
    use_pruning: bool     = False,
    use_low_rank: bool    = True,
    use_clustering: bool  = True,
    use_kd_finetune: bool = False,
    use_quantization: bool = True,
    use_gptq: bool        = False,
    # ── BN Fusion (no hyperparameters — always safe) ─────────────────────────
    # ── Sensitivity Analysis hyperparameters ─────────────────────────────────
    sensitivity_batches: int = 10,
    # ── Structured Pruning hyperparameters ───────────────────────────────────
    pruning_ratio: float             = 0.3,
    pruning_model_type: str          = 'unknown',
    pruning_num_classes: int         = 10,
    pruning_fine_tune_epochs: int    = 3,
    pruning_fine_tune_lr: float      = 0.0001,
    pruning_cal_batches: int         = 50,
    pruning_iterative_steps: int     = 1,
    pruning_round_to: Optional[int]  = None,
    pruning_isomorphic: bool         = False,
    pruning_max_ratio: float         = 0.5,
    pruning_residual_max_ratio: Optional[float] = None,
    # ── Low-Rank Factorization hyperparameters ───────────────────────────────
    lrf_epsilon: float        = 0.5,
    lrf_min_layer_size: int   = 64,
    lrf_min_rank: int         = 2,
    lrf_skip_large_kernels: bool = False,
    lrf_adaptive: bool        = False,     # use per-layer adaptive epsilons
    lrf_energy_threshold: float = 0.99,   # for adaptive mode
    lrf_fine_tune_epochs: int  = 0,       # 0 = no LRF recovery fine-tune (opt-in)
    lrf_fine_tune_lr: float    = 0.0001,
    lrf_num_classes: int       = 10,
    # ── Weight Clustering hyperparameters ────────────────────────────────────
    cluster_num_clusters: int       = 16,
    cluster_fine_tune_epochs: int   = 5,
    cluster_fine_tune_lr: float     = 0.0001,
    cluster_num_classes: int        = 10,
    cluster_layers: Optional[List[str]] = None,
    # ── Knowledge Distillation Fine-tune hyperparameters ─────────────────────
    kd_teacher: Optional[nn.Module]  = None,   # if None, uses original model
    kd_epochs: int                   = 3,
    kd_lr: float                     = 0.0001,
    kd_temperature: float            = 4.0,
    kd_alpha: float                  = 0.7,
    kd_num_classes: int              = 10,
    kd_max_batches: int              = 50,     # max batches per epoch (0 = full dataloader)
    # ── Quantization hyperparameters ─────────────────────────────────────────
    quant_mode: str                    = "dynamic",
    quant_num_calibration_batches: int = 100,
    # ── GPTQ hyperparameters ─────────────────────────────────────────────────
    gptq_bits: int                  = 4,
    gptq_cal_batches: int           = 16,
    gptq_block_size: int            = 128,
    # ── Per-technique accuracy gating ────────────────────────────────────────
    test_loader: Optional[DataLoader] = None,
    accuracy_drop_threshold: float    = 10.0,
    # NEW: early-abort protection for Pruning's and LRF's recovery fine-tunes,
    # applied here in the REAL pipeline (not just the hyperparameter searches
    # in optimization.py, which have their own independent wiring for this).
    early_abort_threshold: Optional[float] = None,
    # ── Per-technique impact reporting ────────────────────────────────────────
    original_size_mb: Optional[float]    = None,
    original_latency_ms: Optional[float] = None,
    input_shape: tuple                   = (1, 3, 224, 224),
    input_dtype: Optional[torch.dtype]   = None,
) -> nn.Module:
    """
    Apply any combination of compression techniques in the correct order.

    Stacking order is fixed (cannot be changed):
      Structured Pruning  →  Low-Rank Factorization  →  Weight Clustering  →  Quantization

    Pruning runs first because LRF wraps layers in nn.Sequential containers.
    Running pruning after LRF would break shape propagation.

    Technique compatibility:
      Structured Pruning     : ✅  Conv2d layers (groups==1) only
      Low-Rank Factorization : ✅  Targets Conv2d + Linear; auto-skips whole-model
                                   if nn.MultiheadAttention is present (see
                                   apply_low_rank_factorization's docstring)
      Weight Clustering      : ✅  Works on any layer type
      Quantization dynamic   : ✅  CPU only (targets nn.Linear recursively)
      Quantization static    : ✅  CPU (fbgemm); wraps full model
      Quantization fp16      : ✅  GPU recommended for speedup

    Args:
        model:                       Input model. NOT mutated.
        dataloader:                  DataLoader for all calibration and fine-tuning.
        device:                      'cuda' or 'cpu'.
        use_pruning:                 Enable Structured Pruning. Default False (opt-in).
        use_low_rank:                Enable Low-Rank Factorization.
        use_clustering:              Enable Weight Clustering.
        use_quantization:            Enable Quantization.
        pruning_ratio:               Fraction of filters to remove per layer.
        pruning_model_type:          Architecture class for safety clamping.
        pruning_num_classes:         Output classes for pruning fine-tune accuracy.
        pruning_fine_tune_epochs:    Fine-tune epochs after pruning (0=skip).
        pruning_fine_tune_lr:        Fine-tune learning rate for pruning.
        pruning_cal_batches:         Calibration batches for importance collection.
        pruning_iterative_steps:     Number of torch-pruning iterative steps (1=single-shot).
        pruning_round_to:            Round pruned channels to this multiple (8/16 for Tensor Cores).
        pruning_isomorphic:          Force same pruning structure across coupled groups.
        pruning_max_ratio:           Hard cap per group (prevents catastrophic collapse).
                                     Also the fallback for pruning_residual_max_ratio
                                     when that argument is None.
        pruning_residual_max_ratio:  Ceiling specifically for auto-detected
                                     residual/skip-connection-coupled groups.
                                     None (default) = fall back to
                                     pruning_max_ratio. See
                                     _find_residual_coupled_groups() and
                                     apply_structured_pruning()'s docstring.
        lrf_epsilon:                 LRF rank ratio. Lower → more compression.
        lrf_min_layer_size:          LRF: skip layers with dim <= this.
        lrf_min_rank:                LRF: skip layers where rank < this (prevents rank-1 collapse).
        lrf_skip_large_kernels:      LRF: skip Conv2d with kernel>1×1.
                                     Use True for ResNet/VGG/DenseNet to prevent latency regression.
                                     Use False for EfficientNet/MobileNet (already mostly 1×1).
        lrf_fine_tune_epochs:        LRF recovery fine-tune epochs (0=skip, default).
                                     Always knowledge distillation against
                                     `model` (or kd_teacher if provided).
        lrf_fine_tune_lr:            Learning rate for the LRF recovery fine-tune.
        lrf_num_classes:             Output class count for the LRF fine-tune's
                                     accuracy metric.
        cluster_num_clusters:        k-means k.
        cluster_fine_tune_epochs:    Fine-tune epochs after clustering (0=skip).
        cluster_fine_tune_lr:        Fine-tune learning rate.
        cluster_num_classes:         Output class count (fine-tune metric).
        cluster_layers:              Explicit layer list; None = all Conv2d+Linear.
        quant_mode:                  "dynamic" | "static" | "fp16".
        quant_num_calibration_batches: Calibration batches for static mode.
        test_loader:                 If provided, enables per-technique accuracy
                                     gating (see module docstring). None = no
                                     gating, old unconditional behaviour.
        accuracy_drop_threshold:     Max allowed marginal accuracy drop (pp)
                                     per technique when gating is enabled.
        early_abort_threshold:       Direct pp value passed to Pruning's and
                                     LRF's own recovery fine-tunes: after
                                     epoch 1, abort remaining epochs if the
                                     held-out test-accuracy drop vs. the
                                     original model already exceeds this.
                                     None (default) = disabled.
        original_size_mb/original_latency_ms: The ABSOLUTE original model's
                                     size/latency, measured ONCE by the
                                     caller (typically run_compression_
                                     pipeline, right alongside its own
                                     baseline accuracy measurement) and
                                     passed through here so every per-
                                     technique impact report's "cumulative
                                     vs. original" numbers compare against
                                     the SAME reference throughout. None =
                                     this function measures its own fallback
                                     baseline internally (on the post-BN-
                                     Fusion model, since BN Fusion always
                                     runs first and is lossless) — only
                                     relevant for direct/library callers who
                                     don't already have these cached; the
                                     real pipeline always supplies both.
                                     Only consulted when test_loader is also
                                     supplied (gating enabled); otherwise no
                                     impact reports are produced at all, so
                                     these baselines are never measured or
                                     used.
        input_shape/input_dtype:     Passed straight through to every
                                     impact report's latency measurement.

    Returns:
        Compressed model (new object; input is not modified).
        If use_pruning=True, the returned model has ._pruning_report attached.
        If test_loader was provided, the returned model has ._gating_report
        (dict of {technique_label: bool_kept}) and ._gating_accuracy
        (float, final accuracy after every gated step) attached, and a
        detailed [Impact] block is printed for every gated technique that
        ran (structural effect, algorithm-vs-fine-tune split where
        applicable, marginal and cumulative accuracy/size/latency) — see
        report_technique_impact() in helper_functions.py for exactly what
        each of those numbers means and why both marginal and cumulative
        are shown side by side rather than just one.

    Raises:
        ValueError: If no technique is enabled.
    """
    if not any([use_bn_fusion, use_pruning, use_low_rank, use_clustering,
                use_quantization, use_gptq]):
        raise ValueError(
            "At least one compression technique must be enabled."
        )

    # Deferred import — measure_accuracy/gate_technique_accuracy live in
    # helper_functions.py.  helper_functions.py already imports compression.py
    # only inside function bodies (never at module load time); doing the same
    # here keeps both import directions lazy and avoids a circular import at
    # module load time.
    _gating_enabled = test_loader is not None
    if _gating_enabled:
        from sigularty.helper_functions import measure_accuracy as _measure_accuracy
        from sigularty.helper_functions import gate_technique_accuracy as _gate
        from sigularty.helper_functions import report_technique_impact as _report_impact
        from sigularty.helper_functions import get_model_size_mb as _get_model_size_mb
        from sigularty.helper_functions import measure_latency as _measure_latency

        def _resolve_post_accuracy(raw_model: nn.Module, kept: bool, kept_accuracy: float) -> float:
            """
            Return the RAW (un-gated) technique output's accuracy for impact
            reporting. When the technique was kept, kept_accuracy already IS
            raw_model's accuracy (gate_technique_accuracy just measured it
            as part of its own decision) — reused for free. When reverted,
            kept_accuracy instead reflects the PRE-technique model, so this
            pays for exactly one extra measure_accuracy() call to report
            honestly on what the technique actually produced, even though
            the pipeline is discarding it.
            """
            if kept:
                return kept_accuracy
            return _measure_accuracy(raw_model, test_loader, device)

    # ── Build ordered list of active technique labels ─────────────────────────
    techniques_applied: List[str] = []
    if use_bn_fusion:
        techniques_applied.append("BN Fusion")
    if use_sensitivity and use_pruning:
        techniques_applied.append("Sensitivity Analysis")
    if use_pruning:
        techniques_applied.append(f"Structured Pruning (ratio={pruning_ratio})")
    if use_low_rank:
        lrf_label = "Adaptive LRF" if lrf_adaptive else f"Low-Rank Factorization (e={lrf_epsilon})"
        techniques_applied.append(lrf_label)
    if use_clustering:
        techniques_applied.append(f"Weight Clustering (k={cluster_num_clusters})")
    if use_kd_finetune:
        techniques_applied.append(f"KD Fine-tune (T={kd_temperature}, a={kd_alpha})")
    if use_quantization:
        techniques_applied.append(f"Quantization ({quant_mode})")
    if use_gptq:
        techniques_applied.append(f"GPTQ INT{gptq_bits}")

    total = len(techniques_applied)
    print("\n" + "=" * 70)
    print("COMPRESSION PIPELINE")
    print("=" * 70)
    print(f"  Order : {' -> '.join(techniques_applied)}")
    if _gating_enabled:
        print(f"  Per-technique accuracy gate: {accuracy_drop_threshold:.1f}pp "
              f"(each technique reverts independently if it alone drops "
              f"accuracy more than this)")
    print("=" * 70)

    compressed = copy.deepcopy(model)
    step_num   = 1
    gating_report: dict = {}
    current_accuracy: Optional[float] = None
    # Captured once, right after BN fusion (lossless), and never reassigned —
    # current_accuracy gets updated after every gate below to reflect
    # whatever survived, but the early-abort check (passed to Pruning's and
    # LRF's recovery fine-tunes) needs the ABSOLUTE ORIGINAL baseline, not
    # a marginal pre-technique value, exactly like baseline_accuracy means
    # everywhere else in this toolkit (run_compression_pipeline's
    # orig_accuracy, the search functions' baseline_accuracy parameter).
    _absolute_baseline_accuracy: Optional[float] = None

    # ── Step 1: BN Fusion ────────────────────────────────────────────────────
    # Always first. Zero accuracy cost, reduces kernel launches, eliminates
    # BN parameters before they are processed by downstream techniques.
    # NEVER gated — it is mathematically lossless by construction.
    if use_bn_fusion:
        print(f"\n[Pipeline Step {step_num}/{total}] BN Fusion")
        compressed = apply_bn_fusion(compressed)
        step_num += 1

    if _gating_enabled:
        print("\n[Gating] Measuring pre-pipeline baseline accuracy "
              f"(global per-technique threshold = {accuracy_drop_threshold:.1f}pp)...")
        current_accuracy = _measure_accuracy(compressed, test_loader, device)
        _absolute_baseline_accuracy = current_accuracy
        print(f"[Gating] Baseline: {current_accuracy:.2f}%")

        # ── Baseline size/latency for per-technique impact reporting ─────────
        # Reuses caller-supplied values when available (run_compression_
        # pipeline already measures these once on the true original model,
        # before BN Fusion even runs) rather than re-measuring — latency in
        # particular is not free. Falls back to measuring here (on the
        # post-BN-Fusion model) only for direct/library callers who don't
        # supply them; BN Fusion is lossless so this fallback is a
        # reasonable stand-in, just not identical to the true pre-BN-Fusion
        # original the real pipeline always has available.
        _absolute_baseline_size = (
            original_size_mb if original_size_mb is not None
            else _get_model_size_mb(compressed)
        )
        if original_latency_ms is not None:
            _absolute_baseline_latency = original_latency_ms
        else:
            _absolute_baseline_latency = _measure_latency(
                compressed, input_shape, device,
                num_iterations=10, warmup=3, input_dtype=input_dtype,
            )['mean_ms']
    else:
        _absolute_baseline_size: Optional[float] = None
        _absolute_baseline_latency: Optional[float] = None

    # ── Step 2: Sensitivity Analysis (optional, informs pruning) ─────────────
    # Run before pruning to compute per-layer importance scores.
    # Only meaningful when pruning is also enabled.  Never gated — read-only,
    # never mutates the model that flows downstream.
    sensitivity_map: Optional[dict] = None
    if use_sensitivity and use_pruning:
        print(f"\n[Pipeline Step {step_num}/{total}] Layer Sensitivity Analysis")
        sensitivity_map = compute_layer_sensitivity(
            compressed, dataloader=dataloader, device=device,
            num_batches=sensitivity_batches,
        )
        step_num += 1

    # ── Step 3: Structured Pruning (gated) ───────────────────────────────────
    # Must run BEFORE LRF — LRF wraps layers in nn.Sequential which breaks
    # Torch-Pruning's dependency graph walk.
    # If sensitivity_map exists, pass it to guide per-layer pruning ratios.
    if use_pruning:
        print(f"\n[Pipeline Step {step_num}/{total}] Structured Pruning")
        _pre_model, _pre_acc = compressed, current_accuracy
        _pruned = apply_structured_pruning(
            compressed,
            dataloader=dataloader,
            device=device,
            pruning_ratio=pruning_ratio,
            model_type=pruning_model_type,
            num_classes=pruning_num_classes,
            fine_tune_epochs=pruning_fine_tune_epochs,
            fine_tune_lr=pruning_fine_tune_lr,
            num_calibration_batches=pruning_cal_batches,
            iterative_steps=pruning_iterative_steps,
            round_to=pruning_round_to,
            isomorphic=pruning_isomorphic,
            max_pruning_ratio=pruning_max_ratio,
            residual_max_ratio=pruning_residual_max_ratio,
            sensitivity_map=sensitivity_map,
            kd_temperature=kd_temperature,
            kd_alpha=kd_alpha,
            test_loader=test_loader,
            baseline_accuracy=_absolute_baseline_accuracy,
            accuracy_drop_threshold=accuracy_drop_threshold,
            early_abort_threshold=early_abort_threshold,
            capture_pre_finetune_accuracy=_gating_enabled,
        )
        _pr = getattr(_pruned, '_pruning_report', {}) or {}
        if _pr.get('torch_pruning_incompatible', False):
            # Torch-Pruning could not build a dependency graph for this
            # architecture at all (explanation already printed inside
            # apply_structured_pruning). Treated exactly like use_pruning
            # had been False for this run: no gating, no impact report, no
            # gating_report entry -- so run_compression_pipeline's final
            # techniques_used list and plot_pruning_report call both
            # correctly leave it out entirely, rather than showing a
            # misleading "kept, zero-effect" entry for a technique that
            # never actually ran.
            print(f"  [Pipeline] Structured Pruning skipped for this run -- "
                  f"continuing without it.")
            compressed = _pre_model
        elif _gating_enabled:
            compressed, current_accuracy, _kept = _gate(
                "Structured Pruning", _pre_model, _pruned, test_loader, device,
                accuracy_drop_threshold, pre_accuracy=_pre_acc,
            )
            gating_report['Structured Pruning'] = _kept

            _n_pruned  = len(_pr.get('layers_pruned', []))
            _n_skipped = len(_pr.get('layers_skipped', []))
            _report_impact(
                "Structured Pruning",
                pre_model=_pre_model, post_model=_pruned,
                pre_accuracy=_pre_acc,
                post_accuracy=_resolve_post_accuracy(_pruned, _kept, current_accuracy),
                was_kept=_kept,
                pre_device=device, post_device=device,
                original_accuracy=_absolute_baseline_accuracy,
                original_size_mb=_absolute_baseline_size,
                original_latency_ms=_absolute_baseline_latency,
                input_shape=input_shape, input_dtype=input_dtype,
                pre_finetune_accuracy=_pr.get('pre_finetune_accuracy'),
                structural_summary=(
                    f"{_n_pruned}/{_n_pruned + _n_skipped} Conv2d/Linear layers pruned "
                    f"(rest: no channels removed by the global solver)"
                ),
                structural_zero=(_n_pruned == 0),
            )
        else:
            compressed = _pruned
        step_num += 1

    # ── Step 4: Low-Rank Factorization (adaptive or standard, gated) ─────────
    if use_low_rank:
        print(f"\n[Pipeline Step {step_num}/{total}] {'Adaptive ' if lrf_adaptive else ''}Low-Rank Factorization")
        _pre_model, _pre_acc = compressed, current_accuracy
        _lrf_teacher = kd_teacher if kd_teacher is not None else model
        if lrf_adaptive:
            adaptive_eps = compute_adaptive_epsilons(
                compressed,
                energy_threshold=lrf_energy_threshold,
                min_layer_size=lrf_min_layer_size,
                skip_large_kernels=lrf_skip_large_kernels,
            )
            _lrf_result = apply_adaptive_lrf(
                compressed,
                adaptive_epsilons=adaptive_eps,
                global_epsilon=lrf_epsilon,
                min_layer_size=lrf_min_layer_size,
                min_rank=lrf_min_rank,
                skip_large_kernels=lrf_skip_large_kernels,
                dataloader=dataloader, device=device, num_classes=lrf_num_classes,
                fine_tune_epochs=lrf_fine_tune_epochs, fine_tune_lr=lrf_fine_tune_lr,
                kd_teacher=_lrf_teacher, kd_temperature=kd_temperature, kd_alpha=kd_alpha,
                test_loader=test_loader,
                baseline_accuracy=_absolute_baseline_accuracy,
                accuracy_drop_threshold=accuracy_drop_threshold,
                early_abort_threshold=early_abort_threshold,
                capture_pre_finetune_accuracy=_gating_enabled,
            )
        else:
            _lrf_result = apply_low_rank_factorization(
                compressed,
                epsilon=lrf_epsilon,
                min_layer_size=lrf_min_layer_size,
                min_rank=lrf_min_rank,
                skip_large_kernels=lrf_skip_large_kernels,
                dataloader=dataloader, device=device, num_classes=lrf_num_classes,
                fine_tune_epochs=lrf_fine_tune_epochs, fine_tune_lr=lrf_fine_tune_lr,
                kd_teacher=_lrf_teacher, kd_temperature=kd_temperature, kd_alpha=kd_alpha,
                test_loader=test_loader,
                baseline_accuracy=_absolute_baseline_accuracy,
                accuracy_drop_threshold=accuracy_drop_threshold,
                early_abort_threshold=early_abort_threshold,
                capture_pre_finetune_accuracy=_gating_enabled,
            )
        if _gating_enabled:
            compressed, current_accuracy, _kept = _gate(
                "Low-Rank Factorization", _pre_model, _lrf_result, test_loader, device,
                accuracy_drop_threshold, pre_accuracy=_pre_acc,
            )
            gating_report['Low-Rank Factorization'] = _kept

            _lr = getattr(_lrf_result, '_lrf_report', {}) or {}
            if _lr.get('mha_skip'):
                _lrf_structural = (
                    f"whole-model skip — {_lr.get('mha_count', 0)} "
                    f"nn.MultiheadAttention module(s) detected, incompatible with LRF"
                )
                _lrf_zero = True
            else:
                _f, _s = _lr.get('factorized_count', 0), _lr.get('skipped_count', 0)
                _lrf_structural = f"{_f}/{_f + _s} eligible layers factorized"
                _lrf_zero = (_f == 0)
            _report_impact(
                "Low-Rank Factorization",
                pre_model=_pre_model, post_model=_lrf_result,
                pre_accuracy=_pre_acc,
                post_accuracy=_resolve_post_accuracy(_lrf_result, _kept, current_accuracy),
                was_kept=_kept,
                pre_device=device, post_device=device,
                original_accuracy=_absolute_baseline_accuracy,
                original_size_mb=_absolute_baseline_size,
                original_latency_ms=_absolute_baseline_latency,
                input_shape=input_shape, input_dtype=input_dtype,
                pre_finetune_accuracy=_lr.get('pre_finetune_accuracy'),
                structural_summary=_lrf_structural,
                structural_zero=_lrf_zero,
            )
        else:
            compressed = _lrf_result
        step_num += 1

    # ── Step 5: Weight Clustering (gated) ────────────────────────────────────
    if use_clustering:
        print(f"\n[Pipeline Step {step_num}/{total}] Weight Clustering")
        _pre_model, _pre_acc = compressed, current_accuracy
        _cluster_teacher = kd_teacher if kd_teacher is not None else model
        _clustered = apply_weight_clustering(
            compressed,
            dataloader=dataloader,
            num_clusters=cluster_num_clusters,
            fine_tune_epochs=cluster_fine_tune_epochs,
            fine_tune_lr=cluster_fine_tune_lr,
            device=device,
            num_classes=cluster_num_classes,
            layers_to_cluster=cluster_layers,
            kd_teacher=_cluster_teacher,
            kd_temperature=kd_temperature,
            kd_alpha=kd_alpha,
            test_loader=test_loader,
            capture_pre_finetune_accuracy=_gating_enabled,
        )
        if _gating_enabled:
            compressed, current_accuracy, _kept = _gate(
                "Weight Clustering", _pre_model, _clustered, test_loader, device,
                accuracy_drop_threshold, pre_accuracy=_pre_acc,
            )
            gating_report['Weight Clustering'] = _kept

            _cr = getattr(_clustered, '_clustering_report', {}) or {}
            _n_clustered = _cr.get('layers_clustered', 0)
            _report_impact(
                "Weight Clustering",
                pre_model=_pre_model, post_model=_clustered,
                pre_accuracy=_pre_acc,
                post_accuracy=_resolve_post_accuracy(_clustered, _kept, current_accuracy),
                was_kept=_kept,
                pre_device=device, post_device=device,
                original_accuracy=_absolute_baseline_accuracy,
                original_size_mb=_absolute_baseline_size,
                original_latency_ms=_absolute_baseline_latency,
                input_shape=input_shape, input_dtype=input_dtype,
                pre_finetune_accuracy=_cr.get('pre_finetune_accuracy'),
                structural_summary=f"{_n_clustered} layer(s) clustered (k={cluster_num_clusters})",
                structural_zero=(_n_clustered == 0),
            )
        else:
            compressed = _clustered
        step_num += 1

    # ── Step 6: In-pipeline Knowledge Distillation Fine-tuning (gated) ───────
    # NOTE: in the real product pipeline (run_compression_pipeline in
    # helper_functions.py), use_kd_finetune is ALWAYS False here — the real
    # KD recovery step runs AFTER quantization/GPTQ (outside this function
    # entirely), so it can recover accuracy lost from those too.  This branch
    # exists for direct/library callers of apply_compression_pipeline.
    if use_kd_finetune:
        print(f"\n[Pipeline Step {step_num}/{total}] Knowledge Distillation Fine-tune")
        teacher = kd_teacher if kd_teacher is not None else model
        _pre_model, _pre_acc = compressed, current_accuracy
        try:
            _kd_result = fine_tune_with_distillation(
                student=compressed,
                teacher=teacher,
                dataloader=dataloader,
                num_classes=kd_num_classes,
                epochs=kd_epochs,
                lr=kd_lr,
                device=device,
                temperature=kd_temperature,
                alpha=kd_alpha,
                max_batches=kd_max_batches,
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("  [KD] OOM: teacher+student too large for GPU. "
                      "Falling back to standard cross-entropy fine-tune.")
                torch.cuda.empty_cache()
                # Standard fine-tune fallback using clustering's _fine_tune_after_clustering
                _kd_result = _fine_tune_after_clustering(
                    compressed, dataloader=dataloader,
                    fine_tune_epochs=kd_epochs, fine_tune_lr=kd_lr,
                    device=device, num_classes=kd_num_classes,
                )
            else:
                raise
        if _gating_enabled:
            compressed, current_accuracy, _kept = _gate(
                "KD Fine-tune (in-pipeline)", _pre_model, _kd_result, test_loader, device,
                accuracy_drop_threshold, pre_accuracy=_pre_acc,
            )
            gating_report['KD Fine-tune (in-pipeline)'] = _kept
            _report_impact(
                "KD Fine-tune (in-pipeline)",
                pre_model=_pre_model, post_model=_kd_result,
                pre_accuracy=_pre_acc,
                post_accuracy=_resolve_post_accuracy(_kd_result, _kept, current_accuracy),
                was_kept=_kept,
                pre_device=device, post_device=device,
                original_accuracy=_absolute_baseline_accuracy,
                original_size_mb=_absolute_baseline_size,
                original_latency_ms=_absolute_baseline_latency,
                input_shape=input_shape, input_dtype=input_dtype,
                structural_summary="n/a — KD only fine-tunes existing weights, no structural change",
                structural_zero=False,
            )
        else:
            compressed = _kd_result
        step_num += 1

    # ── Step 7: GPTQ Quantization (gated) ────────────────────────────────────
    # GPTQ must run BEFORE standard quantization.
    # Standard dynamic INT8 replaces every nn.Linear with _Int8Linear, which
    # means named_modules() finds zero nn.Linear instances when GPTQ runs after.
    # GPTQ and standard quantization are mutually exclusive on Linear layers:
    # use GPTQ (INT4, Hessian-corrected) for transformer/MLP-heavy models,
    # use fp16/dynamic for CNN-heavy models where fp16 on conv weights matters.
    if use_gptq and use_quantization and quant_mode == 'dynamic':
        print("  ⚠️  [Pipeline] use_gptq=True with quant_mode='dynamic': dynamic INT8 "
              "replaces nn.Linear with _Int8Linear before GPTQ can see them.\n"
              "       Running GPTQ first, then skipping dynamic quantization on Linear layers.")
    if use_gptq:
        print(f"\n[Pipeline Step {step_num}/{total}] GPTQ INT{gptq_bits} Quantization")
        _pre_model, _pre_acc = compressed, current_accuracy
        _gptq_result = apply_gptq_quantization(
            compressed,
            dataloader=dataloader,
            device=device,
            bits=gptq_bits,
            num_calibration_batches=gptq_cal_batches,
            block_size=gptq_block_size,
        )
        if _gating_enabled:
            compressed, current_accuracy, _kept = _gate(
                f"GPTQ INT{gptq_bits}", _pre_model, _gptq_result, test_loader, device,
                accuracy_drop_threshold, pre_accuracy=_pre_acc,
            )
            gating_report[f'GPTQ INT{gptq_bits}'] = _kept
        else:
            compressed = _gptq_result
        step_num += 1

    # ── Step 8: Standard Quantization (gated) ────────────────────────────────
    # fp16 is safe after GPTQ (casts weights to half, GPTQ correction is preserved).
    # dynamic INT8 after GPTQ is skipped — _Int8Linear would double-quantize.
    if use_quantization:
        if use_gptq and quant_mode == 'dynamic':
            print(f"\n[Pipeline] Skipping dynamic INT8 — GPTQ INT{gptq_bits} already ran on Linear layers.")
        else:
            print(f"\n[Pipeline Step {step_num}/{total}] Quantization ({quant_mode})")
            _pre_model, _pre_acc = compressed, current_accuracy
            _quant_result = apply_quantization(
                compressed,
                dataloader=dataloader,
                mode=quant_mode,
                num_calibration_batches=quant_num_calibration_batches,
            )
            if _gating_enabled:
                # NOTE: this internal Step 8 (unlike run_compression_pipeline's
                # external Phase B handling) does not move dynamic-quantized
                # models to CPU itself — apply_quantization's dynamic branch
                # quantizes in-place on whatever device `compressed` was
                # already on.  Measuring on `device` here matches where
                # _quant_result actually lives.
                compressed, current_accuracy, _kept = _gate(
                    f"Quantization ({quant_mode})", _pre_model, _quant_result, test_loader, device,
                    accuracy_drop_threshold, pre_accuracy=_pre_acc,
                )
                gating_report[f'Quantization ({quant_mode})'] = _kept
            else:
                compressed = _quant_result
            step_num += 1

    print("\n[Pipeline] All techniques applied successfully.")
    print("=" * 70 + "\n")

    if _gating_enabled:
        compressed._gating_report   = gating_report
        compressed._gating_accuracy = current_accuracy

    return compressed


# ============================================================================
# ── TIER 1: BATCHNORM FUSION ─────────────────────────────────────────────────
# Theory:
#   During inference, Conv2d → BatchNorm2d is always two sequential operations:
#     1. y = W*x + b          (convolution)
#     2. z = γ*(y-μ)/σ + β   (batch normalization)
#
#   Since both are linear in x, they can be collapsed into one operation:
#     W' = W * (γ / sqrt(σ² + ε))       per output-channel scale factor
#     b' = (b - μ) * (γ / sqrt(σ² + ε)) + β
#
#   After fusion, BatchNorm is removed entirely.  The result:
#     - One fewer kernel launch per fused pair
#     - One fewer memory read (BN parameters no longer loaded)
#     - Smaller model size (BN parameters eliminated)
#     - ZERO accuracy change (mathematically equivalent)
#
#   This is what TorchScript, TensorRT, and ONNX Runtime all do as their very
#   first optimization pass.  It is the only compression technique that is
#   universally safe — no hyperparameters, no search, no fine-tuning needed.
#
#   Supported patterns:
#     Conv2d → BatchNorm2d     (most common: ResNet, VGG, EfficientNet)
#     Linear → BatchNorm1d     (less common: some MLP architectures)
#
#   NOT supported (and not needed):
#     Grouped convolutions with BN — the scale factor is still per-output-channel
#     so it works, but we verify correctness via groups==1 only to be safe.
#     BN in the middle of residual branches — fusion is still valid, we handle this.
# ============================================================================

def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse Conv2d + BatchNorm2d into a single Conv2d (in-place on copies).

    Math:
      BN forward: z = γ * (y - μ) / sqrt(σ² + ε) + β
      Combined:   z = (γ/σ') * W * x + (γ/σ') * b - (γ/σ') * μ + β
      where σ' = sqrt(σ² + ε)

      Fused weight  : W' = W * (γ / σ')[:, None, None, None]  (broadcast per out-channel)
      Fused bias    : b' = (b_conv - μ) * (γ / σ') + β

    Args:
        conv: Conv2d layer (NOT modified).
        bn:   BatchNorm2d layer (NOT modified).

    Returns:
        New Conv2d with fused weights and bias.
    """
    # Place fused weights/bias on the same device as the source conv layer.
    device = conv.weight.device

    mean  = bn.running_mean.to(device).float()
    var   = bn.running_var.to(device).float()
    gamma = bn.weight.to(device).float()
    beta  = bn.bias.to(device).float()
    eps   = bn.eps

    std = torch.sqrt(var + eps)           # σ' per output channel
    scale = gamma / std                   # γ/σ' per output channel  [C_out]

    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,                        # fused always has bias
    ).to(device)

    # W' = W * scale, broadcast scale over (in_ch, kH, kW) dimensions
    fused_conv.weight.data = (conv.weight.float() * scale[:, None, None, None]).to(conv.weight.dtype)

    # b' = (b_conv - μ) * scale + β
    b_conv = (conv.bias.to(device).float() if conv.bias is not None
              else torch.zeros(conv.out_channels, device=device))
    fused_conv.bias.data = ((b_conv - mean) * scale + beta).to(conv.weight.dtype)

    return fused_conv


def _fuse_linear_bn_pair(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """
    Fuse Linear + BatchNorm1d into a single Linear.

    Same math as conv fusion; Linear weight is [out, in] so scale broadcasts
    over the input dimension only.

    Args:
        linear: Linear layer (NOT modified).
        bn:     BatchNorm1d layer (NOT modified).

    Returns:
        New Linear with fused weights and bias.
    """
    device = linear.weight.device

    mean  = bn.running_mean.to(device).float()
    var   = bn.running_var.to(device).float()
    gamma = bn.weight.to(device).float()
    beta  = bn.bias.to(device).float()
    eps   = bn.eps

    std   = torch.sqrt(var + eps)
    scale = gamma / std

    fused = nn.Linear(linear.in_features, linear.out_features, bias=True).to(device)
    fused.weight.data = (linear.weight.to(device).float() * scale[:, None]).to(linear.weight.dtype)
    b_lin = (linear.bias.to(device).float() if linear.bias is not None
             else torch.zeros(linear.out_features, device=device))
    fused.bias.data   = ((b_lin - mean) * scale + beta).to(linear.weight.dtype)
    return fused


def apply_bn_fusion(model: nn.Module) -> nn.Module:
    """
    Fuse all Conv2d→BatchNorm2d and Linear→BatchNorm1d pairs in the model.

    This is the FIRST step in any compression pipeline — it is applied before
    pruning, LRF, or clustering because:
      1. It eliminates BN parameters that would otherwise be pruned or clustered
         unnecessarily.
      2. It reduces the number of kernel launches, improving baseline latency
         against which all other techniques are measured.
      3. It is mathematically lossless — zero accuracy change.

    The function walks the entire module tree recursively.  For each sequential
    container (nn.Sequential, any module with named children), it looks for
    adjacent Conv2d/Linear + BN pairs and replaces them with a single fused layer.

    Requirements for fusion:
      - BN must be in training=False mode (running stats must be populated).
        We call model.eval() internally before fusing.
      - Conv2d and BatchNorm2d output channels must match (sanity check).

    Args:
        model: Input model.  NOT mutated; a deep copy is returned.

    Returns:
        New nn.Module with all BN layers fused into preceding conv/linear.
        The returned model has ._bn_fusion_report attached:
          {'pairs_fused': int, 'bn_params_removed': int}
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()   # BN running stats must be populated before fusing

    pairs_fused    = 0
    bn_params_removed = 0

    def _fuse_children(module: nn.Module) -> None:
        nonlocal pairs_fused, bn_params_removed

        # Walk named children; collect names in order for pair detection
        children = list(module.named_children())

        # Build set of names to remove (BN layers that get fused)
        fused_names: set = set()

        for i in range(len(children) - 1):
            name_a, layer_a = children[i]
            name_b, layer_b = children[i + 1]

            if name_a in fused_names:
                continue

            # Pattern 1: Conv2d → BatchNorm2d
            if (isinstance(layer_a, nn.Conv2d) and
                isinstance(layer_b, nn.BatchNorm2d) and
                layer_b.running_mean is not None and
                layer_a.out_channels == layer_b.num_features):

                fused = _fuse_conv_bn_pair(layer_a, layer_b)
                setattr(module, name_a, fused)
                setattr(module, name_b, nn.Identity())
                fused_names.add(name_b)
                pairs_fused += 1
                bn_params_removed += (layer_b.weight.numel() + layer_b.bias.numel() +
                                      layer_b.running_mean.numel() + layer_b.running_var.numel())

            # Pattern 2: Linear → BatchNorm1d
            elif (isinstance(layer_a, nn.Linear) and
                  isinstance(layer_b, nn.BatchNorm1d) and
                  layer_b.running_mean is not None and
                  layer_a.out_features == layer_b.num_features):

                fused = _fuse_linear_bn_pair(layer_a, layer_b)
                setattr(module, name_a, fused)
                setattr(module, name_b, nn.Identity())
                fused_names.add(name_b)
                pairs_fused += 1
                bn_params_removed += (layer_b.weight.numel() + layer_b.bias.numel() +
                                      layer_b.running_mean.numel() + layer_b.running_var.numel())

        # Recurse into children that are not themselves being replaced
        for name, child in module.named_children():
            if name not in fused_names:
                _fuse_children(child)

    print("\n[BN Fusion] Scanning for Conv→BN and Linear→BN pairs...")
    _fuse_children(model_copy)

    # Clean up Identity layers from the state dict (they add no computation)
    # Leave them in place — they cost nothing at inference and removing them
    # would require rebuilding Sequential containers (fragile).

    print(f"[BN Fusion] Done.  Pairs fused: {pairs_fused}  "
          f"BN params eliminated: {bn_params_removed:,}\n")

    model_copy._bn_fusion_report = {
        'pairs_fused':        pairs_fused,
        'bn_params_removed':  bn_params_removed,
    }
    return model_copy


# ============================================================================
# ── TIER 2A: LAYER SENSITIVITY ANALYSIS ──────────────────────────────────────
# Theory:
#   Not all layers are equally important.  Pruning 30% of a redundant middle
#   layer may cause 0.1% accuracy drop; pruning 30% of the first conv (which
#   learns low-level edges) may cause 15% accuracy drop.
#
#   Sensitivity analysis measures this empirically:
#     1. For each layer, zero out all its weights (simulate complete removal).
#     2. Measure accuracy on a small calibration set.
#     3. Sensitivity = baseline_accuracy - zeroed_accuracy.
#        High sensitivity → layer is critical → prune conservatively.
#        Low sensitivity  → layer is redundant → prune aggressively.
#
#   This produces a per-layer sensitivity map which we translate into
#   per-layer pruning ratios for torch-pruning's pruning_ratio_dict.
#   The mapping is: higher sensitivity → lower pruning ratio.
#
#   Formula for per-layer ratio:
#     ratio_i = max_ratio * (1 - sensitivity_i / max_sensitivity)
#   This linearly maps [0, max_sensitivity] → [max_ratio, 0].
#   A layer with zero sensitivity gets the maximum pruning ratio.
#   A layer with maximum sensitivity gets ratio≈0 (barely pruned).
#
#   Why this beats uniform pruning:
#     Uniform: 30% from every layer → kills critical layers, wastes budget
#              on already-redundant ones.
#     Sensitivity-guided: 5% from critical layers, 60% from redundant ones →
#              same total FLOPs reduction, 3-5% better accuracy.
# ============================================================================

def compute_layer_sensitivity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_batches: int = 10,
    target_layers: Optional[List[str]] = None,
) -> dict:
    """
    Compute per-layer sensitivity by zeroing each layer and measuring accuracy drop.

    For each Conv2d or Linear layer:
      1. Save original weights.
      2. Zero the layer's weights.
      3. Run num_batches forward passes, measure accuracy.
      4. Restore original weights.
      5. Sensitivity = baseline_acc - zeroed_acc.

    This is done sequentially (one layer at a time), so memory overhead is
    exactly one model copy.  Total cost: N_layers × num_batches forward passes.

    Args:
        model:         Model to analyse (NOT modified).
        dataloader:    Evaluation DataLoader.
        device:        'cuda' or 'cpu'.
        num_batches:   Batches per layer evaluation (10 is usually enough for ranking).
        target_layers: Explicit list of dotted module names to test.
                       None = all Conv2d and Linear layers.

    Returns:
        Dict mapping dotted layer name → sensitivity score (float ≥ 0).
        Higher = more critical = prune less aggressively.
        Also contains '_baseline_accuracy' key.
    """
    model = copy.deepcopy(model).to(device)
    model.eval()

    # ── Measure baseline accuracy on num_batches ─────────────────────────────
    correct = total = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            X, y = X.to(device), y.to(device)
            if X.is_floating_point():
                try:
                    dtype = next(model.parameters()).dtype
                    X = X.to(dtype=dtype)
                except StopIteration:
                    pass
            preds = model(X).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    baseline_acc = (correct / total * 100) if total > 0 else 0.0

    print(f"\n[Sensitivity] Baseline accuracy on {num_batches} batches: {baseline_acc:.1f}%")

    # ── Discover layers to test ───────────────────────────────────────────────
    if target_layers is None:
        target_layers = [
            name for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and not list(m.children())
        ]

    sensitivities = {'_baseline_accuracy': baseline_acc}
    n = len(target_layers)
    print(f"[Sensitivity] Testing {n} layers ({num_batches} batches each)...")

    for idx, layer_name in enumerate(target_layers):
        # Navigate to the layer
        parts  = layer_name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer = getattr(parent, parts[-1])

        # Skip Sequential wrappers (from LRF) and layers without weight
        if isinstance(layer, nn.Sequential) or not hasattr(layer, 'weight'):
            sensitivities[layer_name] = 0.0
            continue

        # Save and zero weights
        try:
            orig_weight = layer.weight.data.clone()
            orig_bias   = layer.bias.data.clone() if hasattr(layer, 'bias') and layer.bias is not None else None
        except (AttributeError, RuntimeError):
            # Layer doesn't have accessible weight; skip
            sensitivities[layer_name] = 0.0
            continue
        layer.weight.data.zero_()
        if layer.bias is not None:
            layer.bias.data.zero_()

        # Measure zeroed accuracy
        c = t = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):
                if i >= num_batches:
                    break
                X, y = X.to(device), y.to(device)
                try:
                    dtype = next(model.parameters()).dtype
                    X = X.to(dtype=dtype)
                except StopIteration:
                    pass
                try:
                    preds = model(X).argmax(1)
                    c += (preds == y).sum().item()
                    t += y.size(0)
                except Exception:
                    pass  # shape mismatch possible if zeroing causes NaN; treat as 0 acc

        zeroed_acc = (c / t * 100) if t > 0 else 0.0
        sensitivity = max(0.0, baseline_acc - zeroed_acc)
        sensitivities[layer_name] = sensitivity

        # Restore weights
        layer.weight.data = orig_weight
        if orig_bias is not None:
            layer.bias.data = orig_bias

        if (idx + 1) % 10 == 0 or idx == n - 1:
            print(f"  [{idx+1}/{n}] {layer_name}: sensitivity={sensitivity:.2f}%  "
                  f"(zeroed acc={zeroed_acc:.1f}%)")

    # Summary
    layer_sens = {k: v for k, v in sensitivities.items() if k != '_baseline_accuracy'}
    if layer_sens:
        max_s = max(layer_sens.values())
        min_s = min(layer_sens.values())
        most_critical = max(layer_sens, key=layer_sens.get)
        most_redundant = min(layer_sens, key=layer_sens.get)
        print(f"\n[Sensitivity] Range: {min_s:.1f}% – {max_s:.1f}%")
        print(f"  Most critical  : {most_critical} (sensitivity={max_s:.1f}%)")
        print(f"  Most redundant : {most_redundant} (sensitivity={min_s:.1f}%)")

    return sensitivities


def sensitivity_to_pruning_ratios(
    sensitivities: dict,
    max_pruning_ratio: float = 1.0,
    min_pruning_ratio: float = 0.0,
) -> dict:
    """
    Convert layer sensitivity scores to per-layer pruning ratios.

    Linear mapping: sensitivity → pruning_ratio
      High sensitivity (critical layer)  → low pruning ratio  (barely touched)
      Low sensitivity  (redundant layer) → high pruning ratio (aggressively pruned)

    Formula:
      ratio_i = max_ratio * (1 - s_i / max_s)
      where s_i = sensitivity of layer i, max_s = max sensitivity across all layers.
      Clipped to [min_pruning_ratio, max_pruning_ratio].

    Args:
        sensitivities:      Output of compute_layer_sensitivity().
        max_pruning_ratio:  Maximum ratio for any single layer.
        min_pruning_ratio:  Minimum ratio (even critical layers get this much pruned).

    Returns:
        Dict {layer_name: pruning_ratio} for use in torch-pruning's pruning_ratio_dict.
    """
    layer_sens = {k: v for k, v in sensitivities.items() if k != '_baseline_accuracy'}
    if not layer_sens:
        return {}

    max_s = max(layer_sens.values())
    if max_s == 0:
        # All layers equally sensitive — use uniform ratios
        return {k: max_pruning_ratio * 0.5 for k in layer_sens}

    ratios = {}
    for name, s in layer_sens.items():
        r = max_pruning_ratio * (1.0 - s / max_s)
        ratios[name] = float(max(min_pruning_ratio, min(max_pruning_ratio, r)))

    return ratios


# ============================================================================
# ── TIER 2B: KNOWLEDGE DISTILLATION FINE-TUNING ──────────────────────────────
# Theory:
#   Standard fine-tuning trains the compressed model on hard labels:
#     loss = CrossEntropy(logits_compressed, y_true)   [one-hot targets]
#
#   Knowledge distillation trains on soft labels from the original model:
#     loss = α * CrossEntropy(logits_compressed, y_true)          [task loss]
#           + (1-α) * T² * KLDiv(                                 [distillation loss]
#                         softmax(logits_teacher / T),
#                         softmax(logits_compressed / T))
#
#   where T is temperature (typically 2–6) and α balances the two losses.
#
#   WHY SOFT LABELS ARE BETTER:
#     Hard labels say "this is a cat" (probability 1.0) and nothing else.
#     The teacher's soft probabilities say "82% cat, 12% tiger, 6% lion" —
#     encoding inter-class similarity learned over the full training run.
#     The compressed model learns from this richer signal even on a small
#     calibration set, recovering 2–5% more accuracy than hard-label fine-tuning
#     with the same number of epochs.
#
#   The T² multiplier compensates for the fact that KL divergence values shrink
#   as T increases (distributions become more uniform at high temperature).
#   Without T², higher temperature would automatically reduce distillation loss
#   contribution regardless of α.
#
#   This is the SINGLE generic KD function used by:
#     - apply_weight_clustering's recovery step (whenever kd_teacher is given —
#       which the real pipeline always does)
#     - apply_compression_pipeline's (dormant in the real product) in-pipeline
#       KD step
#     - run_compression_pipeline's (helper_functions.py) final post-quantization
#       KD recovery step
#   Structured Pruning and Low-Rank Factorization use the separate
#   _kd_recovery_fine_tune() helper above instead, because they need the
#   {epoch, loss, acc} history format their existing reports already expect.
# ============================================================================

def fine_tune_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: str,
    temperature: float = 4.0,
    alpha: float = 0.7,
    max_batches: int = 0,
    test_loader: Optional[DataLoader] = None,
) -> nn.Module:
    """
    Fine-tune a compressed model using knowledge distillation from the original.

    DEDUPLICATION (this used to be a second, independent copy of this
    function — now a thin pass-through to the single canonical
    implementation in helper_functions.py):

      This file used to carry its own full copy of this function, identical
      in intent to the one in helper_functions.py but drifted from it.  Both
      copies unconditionally cast every input batch to float32
      (`X.to(device).float()`) before the forward pass.  That is correct for
      vision pixel tensors, but WRONG for NLP token-ID tensors (torch.long):
      nn.Embedding.forward() does an index lookup and requires Long/Int
      indices, so casting token IDs to float32 crashes deep inside
      torch.embedding() with "Expected tensor for argument #1 'indices' to
      have one of the following scalar types: Long, Int; but got ...
      FloatTensor". Having two copies meant the bug had to be fixed twice —
      exactly the trap that caused it to resurface here after being fixed
      once in helper_functions.py's copy.

      There is now exactly ONE implementation (helper_functions.py), which
      conditionally casts only floating-point batches — matching
      measure_accuracy's existing convention — so NLP integer batches pass
      through untouched. This function is kept only so
      `from sigularty.compression import fine_tune_with_distillation`
      (this module's documented public API) keeps working for any
      direct/library callers, without maintaining a second implementation
      that can silently drift out of sync again.

    The student (compressed) model learns to:
      1. Predict correct class labels (weighted by alpha).
      2. Match the teacher's (original) soft output distribution (weighted by 1-alpha).

    See helper_functions.fine_tune_with_distillation's docstring for the
    full theory, loss formula, per-epoch history format, and epoch-1-0.0%
    safety-check behaviour — all of that lives in one place now.

    Args:
        student:     Compressed model to fine-tune (NOT modified; deep copy made).
        teacher:     Original uncompressed model (never modified).
        dataloader:  Training DataLoader.
        num_classes: Number of output classes.
        epochs:      Number of fine-tuning epochs.
        lr:          Learning rate for Adam.
        device:      'cuda' or 'cpu'.
        temperature: Softmax temperature T.  Higher -> softer distributions ->
                     more inter-class information preserved.  Try 2-6.
        alpha:       Weight for hard-label (task) loss.  1-alpha = distillation weight.
                     alpha=0.7: 70% task loss, 30% distillation loss.
                     alpha=0.0: pure distillation (no hard labels).
        test_loader: Optional evaluation DataLoader.  When provided, test accuracy
                     is reported and tracked at the end of each epoch.

    Returns:
        Fine-tuned student model (new object, student is NOT modified), with
        `._kd_history` attached.
    """
    # Deferred import (inside the function body, not at module load time) —
    # matches this file's existing lazy-import convention with
    # helper_functions.py (see apply_compression_pipeline's docstring note
    # on avoiding circular imports at module load time; helper_functions.py
    # already imports FROM this file only inside function bodies too, so
    # this keeps both import directions lazy in both files).
    from sigularty.helper_functions import fine_tune_with_distillation as _fine_tune_with_distillation_impl
    return _fine_tune_with_distillation_impl(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        device=device,
        temperature=temperature,
        alpha=alpha,
        max_batches=max_batches,
        test_loader=test_loader,
    )


def compute_adaptive_epsilons(
    model: nn.Module,
    energy_threshold: float = 0.99,
    min_layer_size: int = 64,
    skip_large_kernels: bool = False,
) -> dict:
    """
    Compute the minimum epsilon (rank ratio) per layer that retains
    at least energy_threshold of the layer's singular value energy.

    This replaces a single global epsilon with analytically optimal
    per-layer epsilons derived from the weight matrix structure.

    No data or forward passes required — uses only the weight tensors.

    The output head (final nn.Linear) is always excluded from analysis —
    apply_adaptive_lrf() protects it from factorization regardless, so
    computing (and printing) an epsilon for a layer that will never be
    touched is just noise.

    Args:
        model:             Model to analyse (NOT modified).
        energy_threshold:  Fraction of singular energy to retain [0,1].
                           0.99 = retain 99% → very conservative.
                           0.95 = more compression, slight accuracy cost.
        min_layer_size:    Skip layers with dim <= this (too small to benefit).
        skip_large_kernels: Skip Conv2d with kernel > 1×1 (same logic as LRF).

    Returns:
        Dict {layer_name: epsilon} for all eligible layers.
        Also contains '_summary' key with statistics.
        Layers not in the dict should use the global fallback epsilon.
    """
    epsilons: dict = {}
    stats = {'layers_analysed': 0, 'mean_epsilon': 0.0, 'min_epsilon': 1.0, 'max_epsilon': 0.0}

    _last_linear_name: Optional[str] = None
    for _name, _module in model.named_modules():
        if isinstance(_module, nn.Linear):
            _last_linear_name = _name

    print(f"\n[Adaptive ε] Analysing singular value decay  (threshold={energy_threshold:.0%})...")

    for name, module in model.named_modules():
        if name == _last_linear_name:
            continue   # output head — always protected by apply_adaptive_lrf; skip analysis
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if list(module.children()):   # skip non-leaf (Sequential wrappers)
            continue

        if isinstance(module, nn.Conv2d):
            if module.groups > 1:
                continue
            kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) \
                     else (module.kernel_size, module.kernel_size)
            if skip_large_kernels and (kh > 1 or kw > 1):
                continue
            if module.in_channels <= min_layer_size or module.out_channels <= min_layer_size:
                continue
            W = module.weight.data.float().view(module.out_channels, -1)
            dim_min = min(module.in_channels, module.out_channels)
        else:  # nn.Linear
            if module.in_features <= min_layer_size or module.out_features <= min_layer_size:
                continue
            W = module.weight.data.float()
            dim_min = min(module.in_features, module.out_features)

        # SVD — only singular values needed (not full U, V matrices)
        try:
            S = torch.linalg.svdvals(W)   # descending order
        except Exception:
            try:
                _, S, _ = torch.svd(W)
            except Exception:
                continue

        # Compute energy fraction per rank
        energy     = S ** 2
        total_energy = energy.sum().item()
        if total_energy < 1e-12:
            continue
        cumulative = torch.cumsum(energy, dim=0) / total_energy

        # Find minimum rank r* that captures ≥ energy_threshold
        r_star_candidates = (cumulative >= energy_threshold).nonzero(as_tuple=False)
        if len(r_star_candidates) == 0:
            r_star = len(S)          # need all singular values
        else:
            r_star = int(r_star_candidates[0].item()) + 1   # 1-indexed

        epsilon = r_star / dim_min
        epsilon = float(min(1.0, max(0.01, epsilon)))   # clip to valid range

        epsilons[name] = epsilon
        stats['layers_analysed'] += 1
        stats['min_epsilon'] = min(stats['min_epsilon'], epsilon)
        stats['max_epsilon'] = max(stats['max_epsilon'], epsilon)

    if stats['layers_analysed'] > 0:
        stats['mean_epsilon'] = sum(epsilons.values()) / len(epsilons)
        print(f"  Analysed {stats['layers_analysed']} layers")
        print(f"  ε range: {stats['min_epsilon']:.3f} – {stats['max_epsilon']:.3f}  "
              f"(mean={stats['mean_epsilon']:.3f})")
        # Histogram: how compressible is this model?
        bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        counts = [0] * (len(bins) - 1)
        for e in epsilons.values():
            for i in range(len(bins) - 1):
                if bins[i] <= e < bins[i + 1]:
                    counts[i] += 1
                    break
        print("  Distribution: ", end="")
        labels = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "≥0.9"]
        for lbl, cnt in zip(labels, counts):
            if cnt > 0:
                print(f"{lbl}:{cnt}  ", end="")
        print()
    else:
        print("  No eligible layers found.")

    epsilons['_summary'] = stats
    return epsilons


def apply_adaptive_lrf(
    model: nn.Module,
    adaptive_epsilons: dict,
    global_epsilon: float = 0.9,
    min_layer_size: int = 64,
    min_rank: int = 2,
    skip_large_kernels: bool = False,
    dataloader: Optional[DataLoader] = None,
    device: Optional[str] = None,
    num_classes: Optional[int] = None,
    fine_tune_epochs: int = 0,
    fine_tune_lr: float = 0.0001,
    kd_teacher: Optional[nn.Module] = None,
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.7,
    test_loader: Optional[DataLoader] = None,
    baseline_accuracy: Optional[float] = None,
    accuracy_drop_threshold: Optional[float] = None,
    early_abort_threshold: Optional[float] = None,
    capture_pre_finetune_accuracy: bool = False,
) -> nn.Module:
    """
    Apply Low-Rank Factorization using per-layer adaptive epsilons.

    Each layer uses the epsilon from adaptive_epsilons if available,
    falling back to global_epsilon for layers not in the dict.

    This is a drop-in replacement for apply_low_rank_factorization that
    achieves better compression/accuracy tradeoff by using analytically
    computed per-layer rank ratios instead of one global value.

    Carries the SAME safety behaviour as apply_low_rank_factorization:
      - nn.MultiheadAttention anywhere in the model → return the model
        completely UNCHANGED (see that function's docstring for the
        mechanical reason).
      - The output head (final nn.Linear) is always protected.
      - Optional KD recovery fine-tune (always knowledge distillation,
        never plain cross-entropy) when fine_tune_epochs > 0 and
        dataloader/device/num_classes are all provided.

    Args:
        model:              Input model (NOT modified; deep copy returned).
        adaptive_epsilons:  Output of compute_adaptive_epsilons().
        global_epsilon:     Fallback epsilon for layers not in adaptive_epsilons.
        min_layer_size:     Skip layers with dim <= this.
        min_rank:           Skip layers where rank < this.
        skip_large_kernels: Skip Conv2d with kernel > 1×1.
        dataloader:         Training/calibration DataLoader for the optional
                            KD recovery fine-tune.  Required if fine_tune_epochs>0.
        device:             Compute device for the optional fine-tune.
        num_classes:        Output class count for the optional fine-tune's
                            accuracy metric.
        fine_tune_epochs:   KD recovery epochs after factorization.  0 = skip.
        fine_tune_lr:       Learning rate for the optional fine-tune.
        kd_teacher:         Frozen KD teacher.  None = use `model` itself.
        kd_temperature:     Softmax temperature for the optional fine-tune.
        kd_alpha:           Hard-label (CE) weight for the optional fine-tune.
        test_loader:        Held-out DataLoader for the early-abort check.
                            Same contract as apply_low_rank_factorization's.
        baseline_accuracy:  Original model's accuracy %, for the early-abort
                            check's drop calculation.
        accuracy_drop_threshold: Accepted for API symmetry; not used directly
                            in the early-abort comparison itself.
        early_abort_threshold: Direct pp value — see _kd_recovery_fine_tune's
                            docstring. None (default) = disabled.
        capture_pre_finetune_accuracy: Same contract as
                            apply_low_rank_factorization's parameter of the
                            same name — see that docstring for the full
                            explanation. Default False; only
                            apply_compression_pipeline sets this True.

    Returns:
        New nn.Module with per-layer adaptive LRF applied (or an unchanged
        deepcopy if nn.MultiheadAttention was detected).
    """
    _mha_count = _count_multihead_attention(model)
    if _mha_count > 0:
        print(f"\n[Adaptive LRF] ⚠️  Detected {_mha_count} nn.MultiheadAttention "
              f"module(s) in this model. Same incompatibility as standard LRF "
              f"(see apply_low_rank_factorization's docstring) — returning the "
              f"model UNCHANGED.\n")
        _unchanged = copy.deepcopy(model)
        _mha_pre_finetune_accuracy: Optional[float] = None
        if capture_pre_finetune_accuracy and test_loader is not None and device is not None:
            from sigularty.helper_functions import measure_accuracy as _measure_accuracy_pft
            _mha_pre_finetune_accuracy = _measure_accuracy_pft(_unchanged, test_loader, device)
        _unchanged._lrf_report = {
            'factorized_count':      0,
            'skipped_count':         None,
            'mha_skip':              True,
            'mha_count':             _mha_count,
            'pre_finetune_accuracy': _mha_pre_finetune_accuracy,
        }
        _unchanged._pre_finetune_accuracy = _mha_pre_finetune_accuracy
        return _unchanged

    _last_linear_name: Optional[str] = None
    for _name, _module in model.named_modules():
        if isinstance(_module, nn.Linear):
            _last_linear_name = _name

    model_copy = copy.deepcopy(model)

    print(f"\n[Adaptive LRF] Applying per-layer adaptive factorization...")
    print(f"  Global fallback ε={global_epsilon}  "
          f"(used for {sum(1 for k in [n for n, _ in model.named_modules()] if k not in adaptive_epsilons)} layers)")
    if _last_linear_name is not None:
        print(f"  Output head '{_last_linear_name}' is protected from factorization")

    factorized_count = 0
    skipped_count    = 0

    def _adaptive_factorize(module: nn.Module, parent_name: str = "") -> None:
        nonlocal factorized_count, skipped_count

        for name, child in list(module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(child, nn.Linear):
                if full_name == _last_linear_name:
                    skipped_count += 1
                    continue
                if child.in_features <= min_layer_size or child.out_features <= min_layer_size:
                    skipped_count += 1
                    continue
                eps  = adaptive_epsilons.get(full_name, global_epsilon)
                rank = max(1, int(min(child.in_features, child.out_features) * eps))
                if rank < min_rank:
                    skipped_count += 1
                    continue
                print(f"  Linear {full_name}: ε={eps:.3f} → rank {rank}")
                setattr(module, name, _decompose_linear_layer(child, eps))
                factorized_count += 1

            elif isinstance(child, nn.Conv2d):
                if child.groups > 1:
                    skipped_count += 1
                    continue
                kh, kw = child.kernel_size if isinstance(child.kernel_size, tuple) \
                         else (child.kernel_size, child.kernel_size)
                if skip_large_kernels and (kh > 1 or kw > 1):
                    skipped_count += 1
                    continue
                if child.in_channels <= min_layer_size or child.out_channels <= min_layer_size:
                    skipped_count += 1
                    continue
                eps  = adaptive_epsilons.get(full_name, global_epsilon)
                rank = max(1, int(min(child.in_channels, child.out_channels) * eps))
                if rank < min_rank:
                    skipped_count += 1
                    continue
                print(f"  Conv2d {full_name}: ε={eps:.3f} → rank {rank}")
                setattr(module, name, _decompose_conv_layer(child, eps))
                factorized_count += 1

            elif list(child.children()):
                _adaptive_factorize(child, full_name)

    _adaptive_factorize(model_copy)
    print(f"[Adaptive LRF] Done.  Factorized: {factorized_count}  Skipped: {skipped_count}\n")

    # ── Pre-fine-tune accuracy snapshot (for per-technique impact reporting) ──
    # Same opt-in contract as apply_low_rank_factorization's identical block.
    pre_finetune_accuracy: Optional[float] = None
    if capture_pre_finetune_accuracy and test_loader is not None and device is not None:
        from sigularty.helper_functions import measure_accuracy as _measure_accuracy_pft
        pre_finetune_accuracy = _measure_accuracy_pft(model_copy, test_loader, device)
        print(f"  [Impact] Pre-fine-tune accuracy: {pre_finetune_accuracy:.2f}% "
              f"(raw structural effect, before recovery fine-tune)")

    if fine_tune_epochs > 0:
        if dataloader is not None and device is not None and num_classes is not None:
            _teacher = kd_teacher if kd_teacher is not None else model
            model_copy, _ = _kd_recovery_fine_tune(
                model_copy, teacher=_teacher, dataloader=dataloader,
                fine_tune_epochs=fine_tune_epochs, fine_tune_lr=fine_tune_lr,
                device=device, num_classes=num_classes,
                temperature=kd_temperature, alpha=kd_alpha,
                technique_label="Adaptive LRF Fine-tune",
                test_loader=test_loader, baseline_accuracy=baseline_accuracy,
                accuracy_drop_threshold=accuracy_drop_threshold,
                early_abort_threshold=early_abort_threshold,
            )
        else:
            print("  ⚠️  [Adaptive LRF] fine_tune_epochs > 0 but dataloader/device/"
                  "num_classes not all provided — skipping fine-tune.")

    model_copy._lrf_report = {
        'factorized_count':      factorized_count,
        'skipped_count':         skipped_count,
        'mha_skip':              False,
        'pre_finetune_accuracy': pre_finetune_accuracy,
    }
    model_copy._pre_finetune_accuracy = pre_finetune_accuracy

    return model_copy


# ============================================================================
# ── TIER 3B: GPTQ-STYLE QUANTIZATION ─────────────────────────────────────────
# Theory:
#   Standard INT8/INT4 quantization rounds each weight independently:
#     w_q = round(w / scale)
#   The quantization error e_i = w_i - scale * round(w_i / scale) is
#   independent for each weight and accumulates across the layer.
#
#   GPTQ (Frantar et al., 2022) uses the Hessian of the layer's output loss
#   to distribute quantization error optimally:
#     After quantizing weight w_i, the remaining weights w_j (j > i) are
#     updated to compensate:
#       w_j ← w_j - (e_i / H_ii) * H_ij
#     where H is the (approximate) Hessian of the output w.r.t. weights,
#     computed from the activations X as:
#       H ≈ 2 * X^T X    (outer product approximation)
#
#   Why this works:
#     Quantizing w_i introduces error e_i in the layer output.
#     The Hessian tells us which other weights have correlated influence
#     on the output.  Adjusting them compensates for w_i's
#     error → errors partially cancel → 0.5-2% accuracy drop at INT4.
#
#   Why GPTQ matters:
#     INT4 = 8× compression vs float32, 4× vs INT8. At this compression ratio, standard
#     quantization is unusable (5-15% accuracy loss). GPTQ makes INT4 viable with
#     <2% accuracy loss. This is the technique that made 4-bit LLM quantization practical.
#
#   Our implementation targets Linear layers (the dominant component in
#   Transformers and the FC layers of CNNs).  We use the Cholesky-factored
#   Hessian inverse for numerical stability, following the original paper.
#
#   Bits=4 → 8× compression vs FP32, 4× vs INT8.
#   Bits=8 → same as standard INT8 but with error correction.
# ============================================================================

def _collect_input_activations(
    model: nn.Module,
    target_layer_name: str,
    dataloader: DataLoader,
    device: str,
    num_batches: int = 16,
) -> Optional[torch.Tensor]:
    """
    Collect input activations to a named layer via forward hook.

    Used by GPTQ to estimate the Hessian H ≈ 2 * X^T * X.

    Returns:
        Tensor of shape [N, in_features] (all collected inputs concatenated).
        None if the layer was not reached or had wrong shape.
    """
    collected = []

    def _hook(module, inp, out):
        x = inp[0].detach().float()
        # For Conv2d, unfold spatial dimensions into the batch dim
        if x.ndim == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
        elif x.ndim == 2:
            pass  # Linear: already [B, in_features]
        elif x.ndim == 3:
            B, T, C = x.shape
            x = x.reshape(-1, C)   # NLP: [B, seq_len, hidden] → [B*seq_len, hidden]
        collected.append(x.cpu())

    # Find the layer and register hook
    handle = None
    for n, m in model.named_modules():
        if n == target_layer_name:
            handle = m.register_forward_hook(_hook)
            break

    if handle is None:
        return None

    model.eval()
    try:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                X = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                if X.is_floating_point():
                    try:
                        dtype = next(model.parameters()).dtype
                        X = X.to(dtype=dtype)
                    except StopIteration:
                        pass
                try:
                    model(X)
                except Exception:
                    pass
    finally:
        handle.remove()

    if not collected:
        return None

    return torch.cat(collected, dim=0)


def _float_to_int4_packed(
    W: torch.Tensor,
    block_size: int = 128,
) -> tuple:
    """
    Convert a float32 dequantized weight matrix to packed uint8 + per-block scales.

    Packing layout:
      Each byte holds two 4-bit signed integers (int4, range -8..7):
        bits 0-3  = even column  (col 0, 2, 4, ...)
        bits 4-7  = odd  column  (col 1, 3, 5, ...)

    Returns:
        packed : uint8 tensor  (out_features, ceil(in_features / 2))
        scales : float16 tensor (out_features, n_blocks)
    """
    import torch.nn.functional as _F
    out_f, in_f = W.shape
    n_blocks = (in_f + block_size - 1) // block_size

    # Pad in_features to multiple of block_size
    pad   = n_blocks * block_size - in_f
    W_pad = _F.pad(W.float(), (0, pad)) if pad > 0 else W.float()

    W_blocks = W_pad.view(out_f, n_blocks, block_size)

    # Per-block symmetric scale: max_abs / 7  (int4 symmetric range = -7..7)
    w_max  = W_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # (out, blocks, 1)
    scales = (w_max.squeeze(-1) / 7.0).to(torch.float16)                # (out, blocks)

    # Quantize to int4 integers in [-8, 7]
    W_int  = (W_blocks / w_max * 7.0).round().clamp(-8, 7).to(torch.int8)  # (out, blocks, bs)
    W_int  = W_int.view(out_f, n_blocks * block_size)[:, :in_f]            # crop to in_f

    # Bit-pack: even → low nibble, odd → high nibble
    n_packed = (in_f + 1) // 2
    packed   = torch.zeros(out_f, n_packed, dtype=torch.uint8, device=W.device)

    even_u4 = W_int[:, 0::2].to(torch.int16) & 0x0F
    packed[:, :even_u4.shape[1]] = even_u4.to(torch.uint8)

    if in_f > 1:
        odd_u4 = (W_int[:, 1::2].to(torch.int16) & 0x0F) << 4
        packed[:, :odd_u4.shape[1]] |= odd_u4.to(torch.uint8)

    return packed, scales


class _Int4Linear(nn.Module):
    """
    INT4 weight-only quantized Linear layer with packed uint8 storage + weight cache.

    Storage: Each byte holds two 4-bit signed integers (low nibble = even column,
    high nibble = odd column) → true 8× reduction vs float32, 4× vs float16.
    Per-block float16 scales enable per-block dequantisation without a CUDA kernel.

    WHY THE CACHE EXISTS:
    Dequantization runs ~2.4M ops for a 768×3072 layer (unpack + scale + reshape).
    Without caching, every forward pass pays this cost — on a T4 this adds ~7ms
    per layer per batch, causing the 0.73× slowdown observed on ViT-B/16 (37 layers
    × 7ms = 259ms overhead per forward pass).

    The cache stores the dequantized weight (at this module's current working
    dtype — see _working_dtype_ref below) after the first forward call. It is
    invalidated and rebuilt when:
      - The module is moved to a different device (.to() / .cuda())
      - The module's working dtype changes (.to(dtype) / .float() / .half())
      - The module enters training mode (.train()) — weights may be updated

    In eval mode (inference) the cache is built once and reused for all subsequent
    calls, eliminating the per-forward unpack overhead entirely.

    WORKING DTYPE (fixed bug — previously hardcoded to float16 unconditionally):
    _dequantize()'s output, and `bias`, both now follow _working_dtype_ref, a
    zero-element buffer that exists purely so nn.Module.to(dtype) has something
    to update on this otherwise-parameter-free, frozen layer — the same
    mechanism that lets any ordinary buffer (e.g. BatchNorm's running_mean)
    track a dtype cast. Previously both were hardcoded to float16 at
    construction, which silently assumed this layer would always end up
    inside a model that gets cast to fp16 immediately afterward. That
    assumption is false whenever QUANT_MODE != 'fp16', and — more
    importantly — false for the entire window between GPTQ finishing and
    Quantization actually running: this layer already existed at a
    hardcoded float16 while any untouched sibling GPTQ never reaches (e.g.
    nn.MultiheadAttention's own in_proj_weight/in_proj_bias) was still
    float32. gate_technique_accuracy's forward pass runs in exactly that
    window, so the mismatch fired immediately after GPTQ's own gate — no
    fp16 step required — whenever qat=False (i.e. USE_KD_FINETUNE=False)
    on a model containing nn.MultiheadAttention. Starts at float32 by
    default, matching GPTQ's own construction-time precision (the Hessian
    correction that builds `packed`/`scales` always runs in float32), and
    correctly follows any later whole-model .to(dtype) call from there.

    MEMORY COST: one copy of each weight matrix at the current working dtype
    (same element count as the original model, at whatever precision the
    model currently is). The packed uint8 (the compressed storage) is kept
    for serialisation — saving the model to disk still saves the compact
    INT4 form.

    Created by apply_gptq_quantization() when bits=4.
    """

    def __init__(
        self,
        packed:       torch.Tensor,     # uint8   (out, ceil(in/2))
        scales:       torch.Tensor,     # float16 (out, n_blocks)
        bias:         Optional[torch.Tensor],
        in_features:  int,
        out_features: int,
        block_size:   int = 128,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.block_size   = block_size
        self.register_buffer('packed', packed)
        self.register_buffer('scales', scales)
        # `bias` is stored at whatever dtype it arrives in — always
        # float32 in practice, since GPTQ always runs on a still-float32
        # model (Phase B's fixed order runs GPTQ before Quantization; the
        # Hessian correction itself requires float32 precision) — rather
        # than the previous hardcoded `.to(torch.float16)`. That hardcode
        # assumed this layer would always end up inside a model that later
        # gets cast to fp16, which is common but NOT guaranteed: false
        # whenever QUANT_MODE != 'fp16', and — more importantly — false
        # for the entire window between GPTQ finishing and Quantization
        # actually running, during which this layer already existed at
        # its hardcoded float16 while every untouched sibling (e.g.
        # nn.MultiheadAttention's own in_proj_weight/in_proj_bias, which
        # GPTQ never reaches) was still float32. register_buffer (not a
        # bare attribute) still correctly means .to(dtype) on the whole
        # model moves this buffer right along with everything else going
        # forward; only the CONSTRUCTION-TIME value was ever the problem.
        self.register_buffer('bias', bias.clone() if bias is not None else None)
        # Zero-element buffer whose only purpose is tracking this module's
        # current "working" dtype, correctly updated by nn.Module.to() the
        # same way any other buffer is (torch.zeros(0) carries no data, so
        # this costs nothing). _dequantize() below reads this instead of a
        # hardcoded dtype so the dequantized weight always matches
        # whatever dtype the rest of the model was most recently cast to
        # — needed because, unlike shadow_weight in _Int4LinearQAT,
        # nothing else in this frozen/inference-only layer is an
        # nn.Parameter that would naturally track a .to(dtype) call on its
        # own; `bias` above would work as a stand-in EXCEPT it is
        # legitimately None for any bias=False layer, so a dedicated,
        # always-present reference is what _dequantize() can
        # unconditionally rely on. Starts at float32 to match GPTQ's own
        # construction-time precision (see the bias comment above) — the
        # correct value the very first time this layer is ever evaluated,
        # before any .to(dtype) call has happened.
        self.register_buffer('_working_dtype_ref', torch.zeros(0, dtype=torch.float32))
        # Weight cache: None until first eval forward, invalidated on device move,
        # dtype cast, or train() call. Not a buffer — never serialised to disk.
        self._weight_cache: Optional[torch.Tensor] = None

    def _dequantize(self) -> torch.Tensor:
        """
        Unpack uint8 → int4 codes → weight matrix, cast to this module's
        current working dtype (_working_dtype_ref — see __init__). All
        internal arithmetic stays in float32 for accuracy regardless; only
        the final output cast target changed from a hardcoded float16.
        """
        import torch.nn.functional as _F
        p  = self.packed
        lo = (p & 0x0F).to(torch.int8)
        hi = ((p >> 4) & 0x0F).to(torch.int8)
        lo = lo - ((lo >= 8).to(torch.int8) * 16)
        hi = hi - ((hi >= 8).to(torch.int8) * 16)
        W  = torch.empty(self.out_features, lo.shape[1] * 2,
                         dtype=torch.int8, device=p.device)
        W[:, 0::2] = lo
        W[:, 1::2] = hi
        W  = W[:, :self.in_features]
        n_blocks = (self.in_features + self.block_size - 1) // self.block_size
        pad      = n_blocks * self.block_size - self.in_features
        if pad:
            W = _F.pad(W, (0, pad))
        W_f = W.float().view(self.out_features, n_blocks, self.block_size)
        s   = self.scales.float().unsqueeze(-1)
        W_f = (W_f * s).view(self.out_features, n_blocks * self.block_size)
        return W_f[:, :self.in_features].to(self._working_dtype_ref.dtype)

    def _get_weight(self) -> torch.Tensor:
        """
        Return dequantized weight, using the cache in eval mode.

        In eval mode:  build cache on first call, return cached tensor thereafter.
        In train mode: always dequantize fresh (weights may be updated by optimizer).
        Cache is also invalidated if the packed tensor has moved to a new device
        (e.g. after .cuda() or .to(device) was called) OR if this module's
        working dtype has changed (e.g. after .to(torch.float16)) — both are
        checked, since a cache warmed before a dtype-only .to() call would
        otherwise keep serving the pre-cast dtype indefinitely (only device
        was ever compared here previously).
        """
        if not self.training:
            if (self._weight_cache is not None and
                    (self._weight_cache.device != self.packed.device
                     or self._weight_cache.dtype != self._working_dtype_ref.dtype)):
                self._weight_cache = None
            if self._weight_cache is None:
                self._weight_cache = self._dequantize()
            return self._weight_cache
        # Training mode: always fresh (no cache)
        self._weight_cache = None
        return self._dequantize()

    def train(self, mode: bool = True):
        """Override train() to invalidate the weight cache when switching modes."""
        if mode:
            self._weight_cache = None
        return super().train(mode)

    @property
    def weight(self) -> torch.Tensor:
        """
        Dequantized weight tensor (at this module's current working dtype
        — see _working_dtype_ref) exposed as .weight property.

        Required because nn.MultiheadAttention.forward directly accesses
        self.out_proj.weight as a raw tensor argument to F.multi_head_attention_forward,
        bypassing out_proj.forward() entirely. Without this property, replacing
        out_proj with _Int4Linear raises AttributeError.

        Uses the cache in eval mode — same cost as accessing a regular parameter.
        """
        return self._get_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as _F
        W = self._get_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return _F.linear(x, W, b)

    def extra_repr(self) -> str:
        cached = self._weight_cache is not None
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, bits=4, block_size={self.block_size}, '
            f'dtype={self._working_dtype_ref.dtype}, '
            f'cache={"warm" if cached else "cold"}'
        )


class _Int4LinearQAT(nn.Module):
    """
    Trainable variant of _Int4Linear, using a Straight-Through Estimator
    (STE) so a subsequent fine-tune can actually update GPTQ-quantized
    weights — something _Int4Linear structurally cannot allow, since it
    stores weight/scale/bias ALL as non-trainable buffers (a deliberate
    choice there, favouring a fast, frozen inference path — see its own
    docstring).

    Why this exists: the pipeline's final post-quantization KD recovery
    step exists specifically to "recover accuracy lost from every prior
    step at once," explicitly including quantization. But
    torch.optim.Adam(student.parameters(), ...) only ever sees real
    nn.Parameters — with every GPTQ'd Linear layer frozen as _Int4Linear
    buffers, that "recovery" was silently only ever adjusting LayerNorms,
    embeddings, and whatever small head fell below min_layer_size. For a
    model where GPTQ covers most Linear layers — exactly what this
    toolkit's own model registry recommends for every NLP transformer and
    ViT — the step meant to recover GPTQ's cost could never touch the
    thing that caused it.

    The STE mechanism (standard in quantization-aware training):
        w_fake_quant = w_shadow + (fake_quantize(w_shadow) - w_shadow).detach()
    The forward VALUE equals fake_quantize(w_shadow) exactly (rounded to
    the INT4 grid) — so the model computes as if it were still genuinely
    INT4-constrained. But gradients flow through the .detach()'d
    correction term as zero, landing entirely on w_shadow — so ordinary
    backprop and Adam update the real underlying weight, no custom
    gradient code needed beyond this one identity.

    fake_quantize uses the SAME per-block scale convention as
    _float_to_int4_packed (the scheme _Int4Linear's packed storage is
    built from) — computed ONCE from GPTQ's own Hessian-corrected weight
    at construction time and held FIXED for the lifetime of this object.
    Only the shadow weight moves during training; the quantization grid
    itself does not, so every training step fake-quantizes against a
    consistent, comparable grid rather than a constantly shifting one.

    MEMORY COST while this object exists: one full float32 shadow copy of
    the weight matrix — genuinely double the weight memory of the
    equivalent _Int4Linear, for as long as this object exists. This is
    intentional and temporary: collapse_qat_layers() (called exactly once,
    immediately after the final KD step finishes, regardless of whether
    that step was kept or reverted by its own accuracy gate) converts
    every _Int4LinearQAT back into a clean, frozen _Int4Linear built from
    the shadow's final values, and the shadow is discarded — the memory
    is a cost of that one fine-tuning step, not a permanent cost of using
    GPTQ.

    Only created by apply_gptq_quantization when qat=True is requested —
    run_compression_pipeline only asks for this when USE_KD_FINETUNE=True
    (i.e. a fine-tune that could actually use this trainability is
    genuinely about to run). Otherwise apply_gptq_quantization produces
    the ordinary frozen _Int4Linear, with none of this overhead, exactly
    as before. bits=8 GPTQ never needed this at all: that path already
    leaves a plain, fully-trainable nn.Linear behind (it mutates
    weight.data on a real nn.Linear in place, or — after the clustering-
    compatibility fix — constructs a fresh one), so QAT/STE is scoped to
    bits=4 only, where the frozen-buffer problem actually exists.
    """

    def __init__(
        self,
        shadow_weight: torch.Tensor,   # [out, in] float — GPTQ's own corrected weight
        scale:         torch.Tensor,   # [out, n_blocks] float — per-block scale
        bias:          Optional[torch.Tensor],
        in_features:   int,
        out_features:  int,
        block_size:    int = 128,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.block_size   = block_size
        self.shadow_weight = nn.Parameter(shadow_weight.clone().float())
        # Buffer, not Parameter — the quantization grid is fixed during
        # training; only shadow_weight is meant to move.
        self.register_buffer('scale', scale.clone().float())
        # Real, trainable parameter — GPTQ never touches bias to begin
        # with, so there's no "recovery" story needed for it, it was
        # always free. MUST be wrapped in nn.Parameter explicitly: unlike
        # _Int8Linear (which assigns the LIVE bias Parameter straight
        # through, `self.bias = linear.bias`, and so keeps its subclass
        # and gets auto-registered), the `bias` argument here has already
        # been through `.detach().clone()` at the call site — both
        # operations return a plain torch.Tensor, not an nn.Parameter, so
        # plain attribute assignment silently failed to register it in
        # self._parameters. An unregistered bias is invisible to
        # nn.Module.to(dtype): shadow_weight/scale correctly follow a
        # whole-model .to(torch.float16) cast, but this bias did not,
        # producing a stale float32 tensor sitting next to a float16
        # weight — invisible in ordinary use (forward() below re-casts on
        # every call) but fatal the moment something reads .weight/.bias
        # directly without going through forward(), which is exactly what
        # nn.MultiheadAttention.forward() does for its out_proj sub-layer.
        # (This is the exact defect behind "RuntimeError: self and mat2
        # must have the same dtype, but got Float and Half" on GPTQ+fp16
        # runs with USE_KD_FINETUNE=True.)
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def _fake_quantize(self, w: torch.Tensor) -> torch.Tensor:
        """
        Round w to the fixed per-block INT4 grid and immediately
        dequantize — the forward VALUE this produces is numerically
        identical to what _Int4Linear's packed storage would represent,
        without actually bit-packing anything (no need to; this object is
        temporary by design).
        """
        import torch.nn.functional as _F
        out_f, in_f = w.shape
        n_blocks = self.scale.shape[1]
        pad = n_blocks * self.block_size - in_f
        w_pad = _F.pad(w, (0, pad)) if pad > 0 else w
        w_blocks = w_pad.view(out_f, n_blocks, self.block_size)
        s = self.scale.unsqueeze(-1).clamp(min=1e-8)
        w_int = (w_blocks / s).round().clamp(-8, 7)
        w_dq  = (w_int * s).view(out_f, n_blocks * self.block_size)
        return w_dq[:, :in_f]

    def _ste_weight(self) -> torch.Tensor:
        """
        Straight-through estimator: forward value is the fake-quantized
        (INT4-grid-snapped) weight; gradient flows straight through to
        shadow_weight untouched, since (w_fq - shadow_weight) is detached.
        """
        w_fq = self._fake_quantize(self.shadow_weight)
        return self.shadow_weight + (w_fq - self.shadow_weight).detach()

    @property
    def weight(self) -> torch.Tensor:
        """Same reasoning as _Int4Linear.weight — nn.MultiheadAttention compatibility."""
        return self._ste_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as _F
        w = self._ste_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return _F.linear(x, w, b)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, bits=4 (QAT/STE, trainable), '
                f'block_size={self.block_size}')


def collapse_qat_layers(model: nn.Module) -> nn.Module:
    """
    Convert every _Int4LinearQAT in `model` back into a clean, frozen
    _Int4Linear, using the shadow weight's FINAL (fine-tuned) values —
    dropping the full-precision shadow copy and re-packing fresh.

    Called exactly once, immediately after the final post-quantization KD
    fine-tune finishes (run_compression_pipeline) — regardless of whether
    that fine-tune's own accuracy gate kept or reverted its result; either
    way, whatever model comes out the other side may still contain
    _Int4LinearQAT layers carrying a full fp32 shadow weight, and those
    need to become compact and frozen again before the final report,
    ONNX export, or anything else treats this as "the compressed model."
    This is what actually realises the memory savings _Int4LinearQAT's
    docstring describes as "temporary": before this call, every QAT layer
    is carrying a full fp32 shadow weight alongside its scale; after this
    call, only the compact packed INT4 form remains, exactly like a model
    that went through ordinary (non-QAT) GPTQ.

    Safe to call unconditionally even when QAT was never used (e.g.
    USE_KD_FINETUNE=False, so apply_gptq_quantization never produced any
    _Int4LinearQAT in the first place) — the walk simply finds nothing to
    collapse and returns the model unchanged.

    Args:
        model: Model to collapse IN-PLACE (every _Int4LinearQAT swapped
               for a fresh _Int4Linear in its parent module). By this
               point in the pipeline every earlier stage has already
               deep-copied as needed — there is no "original" this
               function needs to preserve, so an in-place swap is safe
               and avoids one more full-model deep copy.

    Returns:
        The same model object, for chaining.
    """
    converted = 0

    def _walk(module: nn.Module) -> None:
        nonlocal converted
        for name, child in list(module.named_children()):
            if isinstance(child, _Int4LinearQAT):
                with torch.no_grad():
                    final_weight = child._fake_quantize(child.shadow_weight).clone()
                packed, scales = _float_to_int4_packed(final_weight, child.block_size)
                frozen = _Int4Linear(
                    packed=packed, scales=scales,
                    bias=child.bias.detach().clone() if child.bias is not None else None,
                    in_features=child.in_features, out_features=child.out_features,
                    block_size=child.block_size,
                ).to(final_weight.device)
                setattr(module, name, frozen)
                converted += 1
            else:
                _walk(child)

    _walk(model)
    if converted:
        print(f"  [QAT Collapse] Converted {converted} _Int4LinearQAT layer(s) "
              f"back to frozen _Int4Linear — shadow weights dropped, fine-tuned "
              f"values re-packed into compact INT4 storage.")
    return model


def _gptq_quantize_linear(
    weight: torch.Tensor,
    activations: torch.Tensor,
    bits: int = 4,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """
    Apply GPTQ quantization to a single Linear layer's weight matrix.

    Algorithm (Frantar et al., 2022):
      1. Compute H = 2 * X^T X from input activations X.
      2. Add damping: H += percdamp * mean(diag(H)) * I for numerical stability.
      3. Compute H_inv via Cholesky factorization.
      4. Process weights column-by-column in blocks:
         a. Quantize columns in current block.
         b. Compute quantization error for this block.
         c. Propagate error to remaining columns via H_inv.

    Args:
        weight:      Weight matrix [out_features, in_features].
        activations: Input activations [N, in_features].
        bits:        Quantization bits (4 or 8).
        block_size:  Columns processed per block (128 balances speed vs accuracy).
        percdamp:    Hessian damping factor (prevents numerical issues).

    Returns:
        Quantized weight tensor [out_features, in_features], dtype=float32
        (reconstructed from quantized values, ready for use in forward pass).
    """
    W      = weight.float().clone()
    d_in   = W.shape[1]
    n_vals = 2 ** bits                  # e.g. 16 for INT4, 256 for INT8

    # ── Hessian from activations ─────────────────────────────────────────────
    X = activations.float().to(W.device)
    H = (2.0 / X.shape[0]) * (X.T @ X)   # [d_in, d_in]

    # Damping for numerical stability (prevents near-singular Hessian)
    damp  = percdamp * H.diag().mean()
    H.diagonal().add_(damp)

    # Hessian inverse via Cholesky
    try:
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        # Fallback: diagonal approximation if Cholesky fails
        H_inv = torch.diag(1.0 / (H.diagonal().clamp(min=1e-8)))

    # ── Column-by-column quantization with error propagation ─────────────────
    W_quant = W.clone()

    for col_start in range(0, d_in, block_size):
        col_end   = min(col_start + block_size, d_in)
        W_block   = W[:, col_start:col_end].clone()       # [out, block]
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]  # [block, block]
        quant_errors = torch.zeros_like(W_block)

        for j in range(col_end - col_start):
            col_idx = col_start + j
            w_col   = W_block[:, j].clone()               # [out_features]

            # Quantize this column: scale to [-2^(b-1), 2^(b-1)-1] and round
            w_max   = w_col.abs().max().clamp(min=1e-8)
            scale   = (2.0 * w_max) / n_vals
            w_int   = (w_col / scale).round().clamp(-n_vals / 2, n_vals / 2 - 1)
            w_dequant = w_int * scale

            # Error for this column
            err_col = (w_col - w_dequant) / H_inv_block[j, j].clamp(min=1e-8)
            quant_errors[:, j] = err_col

            # Store quantized value
            W_block[:, j] = w_dequant

            # Propagate error to remaining columns in this block
            if j + 1 < col_end - col_start:
                W_block[:, j + 1:] -= torch.outer(err_col, H_inv_block[j, j + 1:])

        W_quant[:, col_start:col_end] = W_block

        # Propagate error beyond this block to remaining columns
        if col_end < d_in:
            W_quant[:, col_end:] -= quant_errors @ H_inv[col_start:col_end, col_end:]

    return W_quant


def apply_gptq_quantization(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    bits: int = 4,
    num_calibration_batches: int = 16,
    block_size: int = 128,
    min_layer_size: int = 64,
    target_layers: Optional[List[str]] = None,
    qat: bool = False,
) -> nn.Module:
    """
    Apply GPTQ-style Hessian-corrected quantization to all Linear layers.

    GPTQ achieves INT4 accuracy close to FP16 by compensating for
    quantization errors using the layer's Hessian (estimated from activations).
    This is significantly better than standard rounding, especially at 4 bits.

    Standard INT8 quantization:     ~1% accuracy drop at 4x compression
    GPTQ INT4 quantization:         ~0.5% accuracy drop at 8x compression
    Standard INT4 (without GPTQ):   ~5-15% accuracy drop at 8x compression

    The quantized weights are stored as float32 (dequantized) in memory.
    True INT4 storage compression requires a custom kernel (e.g. llama.cpp).
    However, ONNX export and TensorRT can take advantage of the quantized
    values for on-device speedup.

    Args:
        model:                   Input model (NOT modified; deep copy returned).
        dataloader:              Calibration DataLoader for Hessian estimation.
        device:                  'cuda' or 'cpu'.
        bits:                    Quantization bits (4 or 8).
        num_calibration_batches: Batches to collect activations for Hessian.
        block_size:              GPTQ block size (128 is standard).
        min_layer_size:          Skip layers with in/out < this (too small to benefit).
        target_layers:           Explicit layer names to quantize.
                                 None = all eligible Linear layers.
        qat:                     If True (and bits=4), produce trainable
                                 _Int4LinearQAT layers instead of frozen
                                 _Int4Linear ones — a real fp32 shadow
                                 weight receives gradients via a straight-
                                 through estimator, so a subsequent
                                 fine-tune can actually update GPTQ's
                                 weights (see _Int4LinearQAT's docstring).
                                 Only meaningful when a fine-tune is
                                 actually about to run afterward — collapse
                                 back to frozen _Int4Linear via
                                 collapse_qat_layers() once it finishes.
                                 Has no effect when bits=8 (that path
                                 already leaves a plain, fully-trainable
                                 nn.Linear behind).

    Returns:
        New nn.Module with GPTQ-quantized Linear layer weights.
        Weights are float32 (dequantized) — compatible with all inference frameworks.
        Model has ._gptq_report attached: {'layers_quantized', 'bits'}.
    """
    print(f"\n[GPTQ] Applying {bits}-bit Hessian-corrected quantization...")
    print(f"  Calibration batches: {num_calibration_batches}  Block size: {block_size}")

    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()

    # Find eligible Linear layers — includes _ClusteredLinear, since the
    # fixed pipeline order runs Clustering before GPTQ. Clustering's role
    # for these layers is pre-conditioning (a smoother, fine-tuned starting
    # point); GPTQ still does its own Hessian-corrected quantization and
    # replaces the whole layer with a fresh _Int4Linear regardless of what
    # it started as — see _ClusteredWeightBase's docstring for why this
    # composes cleanly instead of the two techniques competing for the
    # same layers.
    if target_layers is None:
        target_layers = [
            name for name, m in model_copy.named_modules()
            if isinstance(m, (nn.Linear, _ClusteredLinear))
            and not list(m.children())
            and m.in_features >= min_layer_size
            and m.out_features >= min_layer_size
        ]

    # Snapshot full model size BEFORE quantization (in bytes, then convert to MB)
    # Uses 32-bit float size for each param — GPTQ writes back as float16/float32
    # so RAM doesn't shrink, but we report logical INT{bits} size for each layer.
    size_before_mb = sum(
        p.numel() * p.element_size() for p in model_copy.parameters()
    ) / (1024 ** 2)

    quantized   = 0
    layer_stats = []   # [{name, shape, params, fp32_kb, logical_kb, ratio}]

    for layer_name in target_layers:
        print(f"  Quantizing {layer_name}...", end=" ", flush=True)

        # Collect input activations to this layer
        # (BUGFIX: this used to call _collect_input_activations TWICE in a
        # row with identical arguments — a copy-paste duplication that
        # silently wasted a full num_calibration_batches forward-pass sweep
        # on every single layer, for no purpose. One call is correct.)
        X = _collect_input_activations(
            model_copy, layer_name, dataloader, device, num_calibration_batches
        )

        # Navigate to the layer first so we can fall back gracefully
        parts  = layer_name.split('.')
        parent = model_copy
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer = getattr(parent, parts[-1])

        _is_clustered_input = isinstance(layer, _ClusteredLinear)
        if not isinstance(layer, (nn.Linear, _ClusteredLinear)):
            print("skipped (not Linear)")
            continue

        if X is None or X.shape[0] < 2:
            if bits == 4:
                # PyTorch's MultiheadAttention calls out_proj via the functional API
                # (F.multi_head_attention_forward), bypassing forward hooks entirely.
                # Fall back to standard symmetric INT4 rounding (no Hessian correction).
                # Accuracy cost vs GPTQ is ~1-2% for large layers; storage compression
                # is identical — the layer is still packed to uint8 _Int4Linear
                # (or, in QAT mode, its trainable _Int4LinearQAT counterpart).
                try:
                    _w_start = layer.weight.data.float()
                    packed_w, scales = _float_to_int4_packed(_w_start, block_size)
                    if qat:
                        new_layer = _Int4LinearQAT(
                            shadow_weight = _w_start,
                            scale         = scales,
                            bias          = layer.bias.detach().clone() if layer.bias is not None else None,
                            in_features   = layer.in_features,
                            out_features  = layer.out_features,
                            block_size    = block_size,
                        ).to(device)
                    else:
                        new_layer = _Int4Linear(
                            packed       = packed_w,
                            scales       = scales,
                            bias         = layer.bias.detach() if layer.bias is not None else None,
                            in_features  = layer.in_features,
                            out_features = layer.out_features,
                            block_size   = block_size,
                        ).to(device)
                    setattr(parent, parts[-1], new_layer)
                    quantized += 1
                    params     = layer.weight.numel()
                    fp32_kb    = params * 4 / 1024
                    logical_kb = params * bits / 8 / 1024
                    ratio      = fp32_kb / logical_kb if logical_kb > 0 else 0.0
                    layer_stats.append({
                        'name':       layer_name,
                        'shape':      f"{layer.in_features}×{layer.out_features}",
                        'params':     params,
                        'fp32_kb':    fp32_kb,
                        'logical_kb': logical_kb,
                        'ratio':      ratio,
                    })
                    print(f"done [std INT4 — no activations for Hessian]  "
                          f"({layer.in_features}×{layer.out_features}  "
                          f"FP32:{fp32_kb:.1f} KB → INT4:{logical_kb:.1f} KB  {ratio:.1f}×)")
                except Exception as e:
                    print(f"skipped (fallback failed: {e})")
            else:
                print("skipped (no activations)")
            continue

        try:
            params     = layer.weight.numel()
            fp32_kb    = params * 4 / 1024
            logical_kb = params * bits / 8 / 1024
            ratio      = fp32_kb / logical_kb if logical_kb > 0 else 0.0

            W_quant = _gptq_quantize_linear(
                layer.weight.data, X.to(device),
                bits=bits, block_size=block_size,
            )

            if bits == 4:
                # Pack into real INT4 storage: 2 values per byte = 8× vs float32.
                # Replace the nn.Linear with _Int4Linear so size is actually
                # reduced — or, in QAT mode, with the trainable
                # _Int4LinearQAT counterpart, seeded from this exact same
                # GPTQ-corrected weight and scale, so a subsequent fine-tune
                # starts from precisely what non-QAT GPTQ would have frozen.
                packed_w, scales = _float_to_int4_packed(W_quant, block_size)
                if qat:
                    new_layer = _Int4LinearQAT(
                        shadow_weight = W_quant,
                        scale         = scales,
                        bias          = layer.bias.detach().clone() if layer.bias is not None else None,
                        in_features   = layer.in_features,
                        out_features  = layer.out_features,
                        block_size    = block_size,
                    ).to(device)
                else:
                    new_layer = _Int4Linear(
                        packed      = packed_w,
                        scales      = scales,
                        bias        = layer.bias.detach() if layer.bias is not None else None,
                        in_features  = layer.in_features,
                        out_features = layer.out_features,
                        block_size   = block_size,
                    ).to(device)
                setattr(parent, parts[-1], new_layer)
            else:
                # bits=8: write dequantised (Hessian-corrected) weights back
                # in original dtype. For a plain nn.Linear this mutates
                # weight.data in place. For a _ClusteredLinear input,
                # .weight is a READ-ONLY property (centroids[assignment],
                # reconstructed fresh on every access) — assigning .data to
                # what that property returns has no lasting effect, since
                # the next access just recomputes from the (untouched)
                # centroids again. Build a real replacement nn.Linear
                # instead, exactly mirroring what this branch already does
                # conceptually for a plain nn.Linear (dequantised float
                # weights, no real bit-packing at bits=8 either way).
                if _is_clustered_input:
                    _repl = nn.Linear(layer.in_features, layer.out_features,
                                       bias=layer.bias is not None)
                    _repl.weight.data = W_quant.to(_repl.weight.dtype)
                    if layer.bias is not None:
                        _repl.bias.data = layer.bias.detach().clone()
                    _repl = _repl.to(device)
                    setattr(parent, parts[-1], _repl)
                else:
                    layer.weight.data = W_quant.to(layer.weight.dtype)
            quantized += 1
            layer_stats.append({
                'name':       layer_name,
                'shape':      f"{layer.in_features}×{layer.out_features}",
                'params':     params,
                'fp32_kb':    fp32_kb,
                'logical_kb': logical_kb,
                'ratio':      ratio,
            })
            print(f"done  ({layer.in_features}×{layer.out_features}  "
                  f"FP32:{fp32_kb:.1f} KB → INT{bits}:{logical_kb:.1f} KB  {ratio:.1f}× smaller)")
        except Exception as e:
            print(f"failed ({e})")

    total_fp32_kb    = sum(s['fp32_kb']    for s in layer_stats)
    total_logical_kb = sum(s['logical_kb'] for s in layer_stats)
    total_ratio      = total_fp32_kb / total_logical_kb if total_logical_kb > 0 else 0.0

    print(f"\n[GPTQ] Done.  Layers quantized: {quantized}/{len(target_layers)}")
    if layer_stats:
        print(f"  {'Layer':<35} {'Shape':>12}  {'FP32':>9}  {('INT' + str(bits)):>9}  {'Ratio':>7}")
        print(f"  {'-'*35} {'-'*12}  {'-'*9}  {'-'*9}  {'-'*7}")
        for s in layer_stats:
            print(f"  {s['name']:<35} {s['shape']:>12}  "
                  f"{s['fp32_kb']:>7.1f} KB  {s['logical_kb']:>7.1f} KB  {s['ratio']:>6.1f}×")
        print(f"  {'-'*35} {'-'*12}  {'-'*9}  {'-'*9}  {'-'*7}")
        print(f"  {'TOTAL (quantized layers)':<35} {'':>12}  "
              f"{total_fp32_kb:>7.1f} KB  {total_logical_kb:>7.1f} KB  {total_ratio:>6.1f}×")
        print(f"  Note: PyTorch stores weights as FP16/FP32 in RAM — INT{bits} size is")
        print(f"        the logical compression target achieved with a packed INT{bits} kernel.\n")

    model_copy._gptq_report = {
        'layers_quantized':      quantized,
        'total_eligible_layers': len(target_layers),
        'bits':                  bits,
        'layer_stats':           layer_stats,
        'size_before_mb':        size_before_mb,
        'total_ratio':           total_ratio,
        'qat':                   qat and bits == 4,
    }
    return model_copy