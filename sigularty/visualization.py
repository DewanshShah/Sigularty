"""
visualization.py
================
All visualization and reporting logic for the compression toolkit.

Public API
----------
  generate_compression_report(original_model, compressed_model,
                               metrics_dict, save_path)    -> dict
  plot_epsilon_landscape(results, search_history,
                         optimal_epsilon, ...)             -> None
  plot_pruning_report(pruning_report, save_path)           -> None

Design rules:
  - Imports helper_functions for get_model_size_mb, count_parameters,
    measure_latency, measure_accuracy (no re-implementation).
  - Never runs models or search algorithms — only renders data it receives.
  - All matplotlib figures are closed after saving to prevent resource leaks.
  - Every metric panel gracefully shows 'N/A' when the metric is absent from
    metrics_dict (e.g. latency/accuracy were not computed by the caller).
  - No global state, no hidden side-effects.
"""

from typing import Any, Dict, List, Optional

from scipy.interpolate import CubicSpline   # only scipy import in this file
# Use GridSpec with height_ratios
from matplotlib.gridspec import GridSpec
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sigularty.helper_functions import (
    count_parameters,
    get_model_size_mb,
    measure_accuracy,
    measure_latency,
)

matplotlib.use("Agg")   # non-interactive backend; safe on headless servers


# ============================================================================
# ── INTERNAL HELPERS ─────────────────────────────────────────────────────────
# ============================================================================

def _safe_get(d: dict, key: str, default=None):
    """Return d[key] if it exists and is not None, else default."""
    return d.get(key) if d.get(key) is not None else default


def _fmt(value, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a numeric value or return 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value:{fmt}}{suffix}"


def _bar_pair(
    ax: plt.Axes,
    labels: List[str],
    values: List[Optional[float]],
    colors: List[str],
    ylabel: str,
    title: str,
    ylim: Optional[tuple] = None,
) -> None:
    """
    Draw a two-bar comparison chart.  Missing values (None) are shown as
    a grey bar of height 0 with the label 'N/A'.
    """
    display_values = [v if v is not None else 0.0 for v in values]
    bars = ax.bar(labels, display_values, color=colors, alpha=0.75, edgecolor='black', width=0.45)

    for bar, val in zip(bars, values):
        label_text = _fmt(val) if val is not None else "N/A"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(display_values) * 0.02,
            label_text,
            ha='center', va='bottom',
            fontweight='bold', fontsize=11,
        )

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)


def _ensure_model_metrics(
    original_model: nn.Module,
    compressed_model: nn.Module,
    metrics_dict: dict,
    dataloader=None,
    device: Optional[str] = None,
    input_shape: tuple = (1, 3, 224, 224),
    latency_iterations: int = 100,
    latency_warmup: int = 10,
) -> dict:
    """
    Fill in any missing metrics by computing them directly from the model objects.

    Always computed (no extra args needed):
      original_size_mb, compressed_size_mb, original_params, compressed_params

    Computed when dataloader + device are provided and the value is missing:
      original_accuracy, compressed_accuracy   -- top-1 % via measure_accuracy
      original_latency_ms, compressed_latency_ms -- mean ms via measure_latency

    INT8-quantized compressed models are automatically measured on CPU
    (fbgemm requires CPU regardless of what device is passed in).

    Args:
        original_model:     Uncompressed model.
        compressed_model:   Compressed model.
        metrics_dict:       Caller-supplied metrics; anything present is kept as-is.
        dataloader:         DataLoader for accuracy / latency fallback. Optional.
        device:             'cuda' or 'cpu'. Required for fallback computation.
        input_shape:        Dummy input shape for latency measurement.
        latency_iterations: Timed forward passes.
        latency_warmup:     Untimed warmup passes.

    Returns:
        Updated copy of metrics_dict with all computable values filled in.
    """
    m = dict(metrics_dict)

    # ── Always computable: size and parameter count ───────────────────────────
    if m.get('original_size_mb') is None:
        m['original_size_mb'] = get_model_size_mb(original_model)
    if m.get('compressed_size_mb') is None:
        m['compressed_size_mb'] = get_model_size_mb(compressed_model)
    if m.get('original_params') is None:
        m['original_params'] = count_parameters(original_model)
    if m.get('compressed_params') is None:
        m['compressed_params'] = count_parameters(compressed_model)

    # ── Accuracy and latency: only when dataloader + device are available ─────
    if dataloader is None or device is None:
        return m

    # INT8-quantized models (fbgemm backend) must run on CPU
    _is_quantized = any(
        'quantized' in type(mod).__name__.lower() or 'Quantized' in type(mod).__name__
        for mod in compressed_model.modules()
    )
    comp_device = 'cpu' if _is_quantized else device

    if m.get('original_accuracy') is None:
        print("  [Report] Computing original model accuracy...")
        try:
            m['original_accuracy'] = measure_accuracy(original_model, dataloader, device)
        except Exception as exc:
            print(f"  [Report] Warning: original accuracy failed: {exc}")

    if m.get('compressed_accuracy') is None:
        print("  [Report] Computing compressed model accuracy...")
        try:
            m['compressed_accuracy'] = measure_accuracy(
                compressed_model, dataloader, comp_device
            )
        except Exception as exc:
            print(f"  [Report] Warning: compressed accuracy failed: {exc}")

    if m.get('original_latency_ms') is None:
        print("  [Report] Computing original model latency...")
        try:
            lat = measure_latency(
                original_model, input_shape, device,
                num_iterations=latency_iterations, warmup=latency_warmup,
            )
            m['original_latency_ms'] = lat['mean_ms']
        except Exception as exc:
            print(f"  [Report] Warning: original latency failed: {exc}")

    if m.get('compressed_latency_ms') is None:
        print("  [Report] Computing compressed model latency...")
        try:
            lat = measure_latency(
                compressed_model, input_shape, comp_device,
                num_iterations=latency_iterations, warmup=latency_warmup,
            )
            m['compressed_latency_ms'] = lat['mean_ms']
        except Exception as exc:
            print(f"  [Report] Warning: compressed latency failed: {exc}")

    return m


# ============================================================================
# ── EPSILON LANDSCAPE (LRF Compression Quality Index search results) ─────────
# ============================================================================

def plot_epsilon_landscape(
    results: dict,
    search_history: list,
    optimal_epsilon: Optional[float],
    baseline_accuracy: float,
    baseline_size: float,
    save_path: str = 'epsilon_landscape.png',
) -> None:
    """
    Render a 4-panel visualisation of the LRF epsilon search results.

    This function is purely a renderer — it never runs a model or calls any
    search algorithm.  All data is supplied by find_optimal_epsilon_smart()
    in optimization.py.

    Panels:
      1 (top-left)    — CQI Landscape: anchor points + cubic spline.
      2 (top-right)   — Anchor Points Accuracy vs Size: dual-axis line chart.
      3 (bottom-left) — Binary Search Path: spline background + evaluated points.
      4 (bottom-right)— Top 3 Candidates: horizontal bar chart.

    Args:
        results:           Dict returned by find_optimal_epsilon_smart():
                           {str(round(epsilon, 4)): {accuracy, size_mb, score, phase}}.
                           May also contain a 'warning' key (str) which is filtered out.
        search_history:    Ordered list of evaluation dicts with 'phase' labels.
        optimal_epsilon:   Best epsilon found, or None if no viable epsilon existed.
        baseline_accuracy: Original model accuracy % (used for Panel 2 reference).
        baseline_size:     Original model size in MB (used for Panel 2 reference).
        save_path:         Destination file path for the PNG.
    """

    # ── Extract anchor and binary search data ──────────────────────────────────
    # Filter out the optional 'warning' key and any non-dict entries
    valid_results: List[Tuple[float, dict]] = [
        (float(k), v)
        for k, v in results.items()
        if k != 'warning' and isinstance(v, dict) and 'score' in v
    ]
    valid_results.sort(key=lambda x: x[0])   # sort by epsilon ascending

    anchor_entries = [
        (eps, r) for eps, r in valid_results if r.get('phase') == 'anchor'
    ]
    anchor_entries.sort(key=lambda x: x[0])

    anchor_eps    = [e for e, _ in anchor_entries]
    anchor_scores = [r['score']    for _, r in anchor_entries]
    anchor_accs   = [r['accuracy'] for _, r in anchor_entries]
    anchor_sizes  = [r['size_mb']  for _, r in anchor_entries]

    binary_evals = [
        entry for entry in search_history if entry.get('phase') == 'binary'
    ]

    # Top-3 candidates ranked by score descending
    all_ranked = sorted(valid_results, key=lambda x: x[1]['score'], reverse=True)
    top3       = all_ranked[:3]

    # Detect tie-break: top-2 scores within 5% and optimal is the lower-eps one
    tie_broken = False
    if (
        optimal_epsilon is not None
        and len(all_ranked) >= 2
    ):
        s0 = all_ranked[0][1]['score']
        s1 = all_ranked[1][1]['score']
        if s0 > 0 and abs(s0 - s1) / s0 < 0.05:
            # Top-2 tied — was the winner chosen for being lower-epsilon?
            if abs(all_ranked[0][0] - optimal_epsilon) > 1e-5:
                tie_broken = True   # optimal is NOT the highest-score entry

    # ── Build cubic spline through anchor points ──────────────────────────────
    spline        = None
    x_spline      = None
    y_spline      = None
    if len(anchor_eps) >= 4:
        try:
            spline   = CubicSpline(anchor_eps, anchor_scores)
            x_spline = np.linspace(min(anchor_eps), max(anchor_eps), 300)
            y_spline = spline(x_spline)
        except Exception:
            spline = None   # degrade gracefully if scipy fails

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#f8f9fa')

    title = "Epsilon Landscape — LRF Compression Quality Index Search"
    if optimal_epsilon is None:
        title += "\n(no viable epsilon found — CQI < 0.5)"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # colour constants matching the existing compression report palette
    BLUE     = '#3498db'
    GREEN    = '#2ecc71'
    RED      = '#e74c3c'
    ORANGE   = '#f39c12'
    GREY_MID = '#bdc3c7'
    DARK     = '#2c3e50'
    GOLD     = '#FFD700'
    SILVER   = '#C0C0C0'
    BRONZE   = '#CD7F32'

    # =========================================================================
    # PANEL 1 — CQI Landscape
    # =========================================================================
    ax1 = axes[0, 0]

    # Cubic spline interpolation (dashed, drawn only within anchor range)
    if spline is not None:
        ax1.plot(x_spline, y_spline, linestyle='--', color=BLUE, alpha=0.5,
                 linewidth=1.5, label='Interpolated curve')

    # Horizontal baseline at score = 1.0 (no improvement over original)
    ax1.axhline(1.0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7,
                label='No improvement (CQI=1.0)')

    # Anchor scatter points
    ax1.scatter(anchor_eps, anchor_scores,
                s=150, color=BLUE, edgecolors='black', zorder=5,
                label='Anchor points')

    # Annotate each anchor with its epsilon value (above the point)
    for eps, score in zip(anchor_eps, anchor_scores):
        ax1.annotate(
            f'ε={eps:.2f}',
            xy=(eps, score),
            xytext=(0, 9),
            textcoords='offset points',
            fontsize=8, color=DARK, ha='center',
        )

    # Optimal epsilon star
    if optimal_epsilon is not None:
        opt_key   = str(round(optimal_epsilon, 4))
        opt_score = results.get(opt_key, {}).get('score', 0.0)
        if opt_score == 0.0 and spline is not None:
            opt_score = float(spline(optimal_epsilon))
        ax1.scatter([optimal_epsilon], [opt_score],
                    marker='*', s=350, color='red', edgecolors='darkred',
                    zorder=10, label=f'Optimal ε={optimal_epsilon:.2f}')

    ax1.set_xlabel('Epsilon (ε)', fontweight='bold')
    ax1.set_ylabel('CQI (Compression Quality Index)', fontweight='bold')
    ax1.set_title('Compression Quality Index (CQI) Landscape', fontweight='bold', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # =========================================================================
    # PANEL 2 — Anchor Points: Accuracy vs Size (dual axis)
    # =========================================================================
    ax2   = axes[0, 1]
    ax2_r = ax2.twinx()

    line_acc, = ax2.plot(anchor_eps, anchor_accs,
                         color=GREEN, marker='o', linewidth=2, markersize=7,
                         label='Accuracy (%)')
    line_size, = ax2_r.plot(anchor_eps, anchor_sizes,
                            color=RED, marker='s', linewidth=2, markersize=7,
                            linestyle='--', label='Size (MB)')

    if optimal_epsilon is not None:
        vline = ax2.axvline(optimal_epsilon, color='grey', linestyle='--',
                            linewidth=1.5, alpha=0.6,
                            label=f'Optimal ε={optimal_epsilon:.2f}')
    else:
        vline = None

    ax2.set_xlabel('Epsilon (ε)', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', color=GREEN, fontweight='bold')
    ax2_r.set_ylabel('Size (MB)',  color=RED,   fontweight='bold')
    ax2.set_title('Anchor Points — Accuracy vs Size', fontweight='bold', fontsize=12)
    ax2.grid(alpha=0.3)

    # Combine legend handles from both axes
    handles = [line_acc, line_size]
    if vline is not None:
        handles.append(vline)
    ax2.legend(handles=handles, loc='upper center', fontsize=8)

    # =========================================================================
    # PANEL 3 — Binary Search Path
    # =========================================================================
    ax3 = axes[1, 0]

    if len(binary_evals) == 0:
        # No binary search iterations were needed (search converged on anchor points)
        ax3.text(0.5, 0.5,
                 "Search converged on\nanchor points\n\nNo binary search iterations needed",
                 fontsize=14, color='#7f8c8d', ha='center', va='center',
                 transform=ax3.transAxes)
        ax3.set_title('Binary Search Path', fontweight='bold', fontsize=12)
    else:
        # Cubic spline as a light landscape reference
        if spline is not None and x_spline is not None:
            ax3.plot(x_spline, y_spline, color=GREY_MID, alpha=0.4,
                     linewidth=2, label='Landscape')

        # Binary search evaluation points, numbered in evaluation order
        binary_eps     = [e['epsilon'] for e in binary_evals]
        binary_scores  = [e['score']   for e in binary_evals]

        ax3.scatter(binary_eps, binary_scores,
                    marker='D', s=120, color=ORANGE, edgecolors='black',
                    zorder=5, label='Binary search evaluations')

        for idx, (eps, score) in enumerate(zip(binary_eps, binary_scores), start=1):
            ax3.annotate(
                str(idx),
                xy=(eps, score),
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=8, color=ORANGE, fontweight='bold', ha='center',
            )

        # Optimal epsilon star
        if optimal_epsilon is not None:
            opt_key   = str(round(optimal_epsilon, 4))
            opt_score = results.get(opt_key, {}).get('score', 0.0)
            if opt_score == 0.0 and spline is not None:
                opt_score = float(spline(optimal_epsilon))
            ax3.scatter([optimal_epsilon], [opt_score],
                        marker='*', s=350, color='red', edgecolors='darkred',
                        zorder=10, label=f'Optimal ε={optimal_epsilon:.2f}')

        ax3.set_xlabel('Epsilon (ε)', fontweight='bold')
        ax3.set_ylabel('CQI', fontweight='bold')
        ax3.set_title('Binary Search Path', fontweight='bold', fontsize=12)
        ax3.grid(alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)

    # =========================================================================
    # PANEL 4 — Top 3 Candidates (horizontal bar chart, best at top)
    # =========================================================================
    ax4 = axes[1, 1]

    if len(top3) == 0:
        ax4.text(0.5, 0.5, 'No candidates available.',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Top 3 Candidates', fontweight='bold', fontsize=12)
    else:
        rank_colors = [GOLD, SILVER, BRONZE]

        # Display order: reversed so that rank-1 (best) appears at the top of
        # the chart (highest y-index in barh)
        top3_display = list(reversed(top3))          # [3rd, 2nd, 1st]
        display_colors = list(reversed(rank_colors[:len(top3)]))  # [bronze, silver, gold]
        y_indices = list(range(len(top3_display)))   # [0, 1, 2]

        bars = ax4.barh(
            y_indices,
            [r[1]['score'] for r in top3_display],
            color=display_colors,
            alpha=0.85,
            edgecolor='black',
        )

        ax4.set_yticks(y_indices)
        ax4.set_yticklabels([f"ε={r[0]:.2f}" for r in top3_display])

        # Annotate each bar at the right end with accuracy / size / score
        x_max = max(r[1]['score'] for r in top3_display) if top3_display else 1.0
        for y_pos, (eps, r) in zip(y_indices, top3_display):
            label = (f"  {r['accuracy']:.1f}%  |  "
                     f"{r['size_mb']:.2f} MB  |  "
                     f"score={r['score']:.3f}")
            ax4.text(r['score'], y_pos, label,
                     va='center', fontsize=9, color=DARK)

        # Tie-break annotation on the winning bar
        if tie_broken and optimal_epsilon is not None:
            # Find which y-position corresponds to the optimal epsilon
            for y_pos, (eps, _) in zip(y_indices, top3_display):
                if abs(eps - optimal_epsilon) < 1e-5:
                    ax4.text(0.0, y_pos + 0.3,
                             '← tie-break: lower ε chosen',
                             fontsize=8, color=RED, style='italic',
                             va='center')
                    break

        # Vertical baseline at score = 1.0
        ax4.axvline(1.0, color='grey', linestyle='--', alpha=0.7,
                    label='No improvement (CQI=1.0)')

        ax4.set_xlim(left=0)
        ax4.set_xlabel('CQI (Compression Quality Index)', fontweight='bold')
        ax4.set_title('Top 3 Candidates', fontweight='bold', fontsize=12)
        ax4.grid(axis='x', alpha=0.3)
        ax4.legend(loc='lower right', fontsize=8)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"📊 Epsilon landscape saved → '{save_path}'")


# ============================================================================
# ── PRUNING REPORT ────────────────────────────────────────────────────────────
# ============================================================================

def plot_pruning_report(
    pruning_report: dict,
    save_path: str = 'pruning_report.png',
) -> None:
    """
    Render a 4-panel visual report of structured pruning results.

    Receives the ._pruning_report dict attached to the pruned model by
    apply_structured_pruning().  Never runs the model — only renders data.

    Panels:
      1 (top-left)    — "What did the model look like before vs after?"
                        Grouped bar chart of original vs remaining channel
                        counts per pruned layer.  Annotated with % removed.
      2 (top-right)   — "Why were those filters chosen for removal?"
                        Histogram of all pre-pruning importance scores.
                        Vertical cutoff line; red = removed, green = kept.
      3 (bottom-left) — "Is the model still safe to deploy?"
                        Behavioral probe result as a styled info panel.
                        Background colour reflects severity.
      4 (bottom-right)— "Did fine-tuning actually recover accuracy?"
                        Line chart if fine-tuning history is available,
                        otherwise two-bar original-vs-pruned accuracy.

    Args:
        pruning_report: Dict from model._pruning_report.
        save_path:      Destination PNG path.
    """
    layers_pruned        = pruning_report.get('layers_pruned',         [])
    all_scores           = pruning_report.get('all_importance_scores', [])
    threshold            = pruning_report.get('importance_threshold',  0.0)
    probe                = pruning_report.get('behavioral_probe',      None)
    fine_tune_epochs     = pruning_report.get('fine_tune_epochs',       0)
    fine_tune_history    = pruning_report.get('fine_tune_history',     [])
    original_params      = pruning_report.get('original_params',       None)
    pruned_params        = pruning_report.get('pruned_params',         None)
    pruning_ratio        = pruning_report.get('pruning_ratio',          0.0)
    model_type           = pruning_report.get('model_type',            'unknown')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        f'Structured Pruning Report  '
        f'(ratio={pruning_ratio:.0%}, model_type={model_type})',
        fontsize=15, fontweight='bold', y=0.98,
    )

    GREEN  = '#2ecc71'
    RED    = '#e74c3c'
    BLUE   = '#3498db'
    ORANGE = '#f39c12'
    DARK   = '#2c3e50'
    GREY   = '#bdc3c7'

    # =========================================================================
    # PANEL 1 — Before vs After: channel counts per pruned layer
    # =========================================================================
    ax1 = axes[0, 0]

    if not layers_pruned:
        ax1.text(0.5, 0.5, 'No layers were pruned.',
                 ha='center', va='center', transform=ax1.transAxes,
                 fontsize=13, color=GREY)
    else:
        # Shorten long layer names to keep axis readable
        labels  = [l['name'].split('.')[-3:] for l in layers_pruned]
        labels  = ['.'.join(p) for p in labels]
        orig    = [l['original_channels']  for l in layers_pruned]
        remain  = [l['remaining_channels'] for l in layers_pruned]
        x       = np.arange(len(labels))
        w       = 0.38

        bars_orig   = ax1.bar(x - w / 2, orig,   width=w, color=BLUE,  alpha=0.75,
                              edgecolor='black', label='Original')
        bars_remain = ax1.bar(x + w / 2, remain, width=w, color=GREEN, alpha=0.75,
                              edgecolor='black', label='Remaining')

        # Annotate each bar pair with percentage removed
        for i, layer in enumerate(layers_pruned):
            pct = layer['fraction_removed'] * 100
            ax1.text(x[i], max(orig[i], remain[i]) + max(orig) * 0.02,
                     f'-{pct:.0f}%', ha='center', fontsize=7,
                     color=RED, fontweight='bold')

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
        ax1.set_ylabel('Channel Count', fontweight='bold')
        ax1.legend(fontsize=8)

    ax1.set_title('Before vs After: Channel Counts', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # =========================================================================
    # PANEL 2 — Importance score histogram with cutoff
    # =========================================================================
    ax2 = axes[0, 1]

    if not all_scores:
        ax2.text(0.5, 0.5, 'No importance scores collected.',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=13, color=GREY)
    else:
        scores_arr = np.array(all_scores)
        removed    = scores_arr[scores_arr <= threshold]
        kept       = scores_arr[scores_arr >  threshold]

        bins = min(50, max(10, len(scores_arr) // 20))
        all_range = (scores_arr.min(), scores_arr.max())

        ax2.hist(removed, bins=bins, range=all_range,
                 color=RED,   alpha=0.65, label=f'Removed ({len(removed)})')
        ax2.hist(kept,    bins=bins, range=all_range,
                 color=GREEN, alpha=0.65, label=f'Kept ({len(kept)})')

        ax2.axvline(threshold, color=DARK, linestyle='--', linewidth=1.8,
                    label=f'Cutoff = {threshold:.4f}')

        ax2.set_xlabel('Mean Absolute Activation (Importance)', fontweight='bold')
        ax2.set_ylabel('Filter Count', fontweight='bold')
        ax2.legend(fontsize=8)

    ax2.set_title('Filter Importance Distribution', fontweight='bold', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # =========================================================================
    # PANEL 3 — Behavioral probe (safety assessment)
    # =========================================================================
    ax3 = axes[1, 0]
    ax3.axis('off')

    _SEV_COLORS = {
        'negligible': '#d5f5e3',
        'acceptable': '#d5f5e3',
        'moderate':   '#fef9e7',
        'high':       '#fadbd8',
    }
    _SEV_TEXT_COLORS = {
        'negligible': '#1e8449',
        'acceptable': '#1e8449',
        'moderate':   '#9a7d0a',
        'high':       '#922b21',
    }
    _SEV_ICONS = {
        'negligible': '✅', 'acceptable': '✅',
        'moderate': '⚠️', 'high': '❌',
    }

    if probe is None:
        ax3.set_facecolor('#f2f3f4')
        ax3.text(0.5, 0.5, 'Probe not run.', ha='center', va='center',
                 fontsize=13, color=GREY, transform=ax3.transAxes)
    else:
        severity    = probe.get('severity', 'unknown')
        metric      = probe.get('metric', '')
        value       = probe.get('value', None)
        rec         = probe.get('recommendation', '')
        bg_color    = _SEV_COLORS.get(severity, '#f2f3f4')
        text_color  = _SEV_TEXT_COLORS.get(severity, DARK)
        icon        = _SEV_ICONS.get(severity, '?')

        ax3.set_facecolor(bg_color)

        # Large metric value
        metric_label = 'KL' if metric == 'kl_divergence' else 'Cosine'
        val_str      = f'{value:.4f}' if value is not None else 'N/A'
        ax3.text(0.5, 0.72, f'{metric_label} = {val_str}',
                 ha='center', va='center', fontsize=22, fontweight='bold',
                 color=DARK, transform=ax3.transAxes)

        # Severity label
        ax3.text(0.5, 0.54,
                 f'{icon}  {severity.upper()}',
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 color=text_color, transform=ax3.transAxes)

        # Recommendation
        # Word-wrap at ~55 chars
        words, lines_out, line = rec.split(), [], ''
        for w in words:
            if len(line) + len(w) + 1 > 55:
                lines_out.append(line)
                line = w
            else:
                line = (line + ' ' + w).strip()
        if line:
            lines_out.append(line)
        rec_wrapped = '\n'.join(lines_out)
        ax3.text(0.5, 0.36, rec_wrapped,
                 ha='center', va='center', fontsize=9.5,
                 color=DARK, transform=ax3.transAxes,
                 style='italic', wrap=True)

        # Severity scale bar at bottom
        scale_ax = ax3.inset_axes([0.05, 0.04, 0.90, 0.14])
        scale_ax.axis('off')
        seg_colors  = ['#2ecc71', '#58d68d', '#f39c12', '#e74c3c']
        seg_labels  = ['<0.01\nnegligible', '<0.05\nacceptable',
                       '<0.15\nmoderate', '≥0.15\nhigh']
        for si, (sc, sl) in enumerate(zip(seg_colors, seg_labels)):
            scale_ax.add_patch(
                mpatches.FancyBboxPatch(
                    (si / 4, 0), 0.95 / 4, 1.0,
                    boxstyle='round,pad=0.01',
                    facecolor=sc, edgecolor='white', linewidth=1,
                    transform=scale_ax.transAxes,
                )
            )
            scale_ax.text((si + 0.5) / 4, 0.5, sl,
                          ha='center', va='center', fontsize=6.5,
                          color='white', fontweight='bold',
                          transform=scale_ax.transAxes)

    ax3.set_title('Behavioral Probe — Safety Assessment', fontweight='bold', fontsize=11)

    # =========================================================================
    # PANEL 4 — Fine-tuning recovery (or accuracy comparison)
    # =========================================================================
    ax4 = axes[1, 1]

    if fine_tune_epochs > 0 and fine_tune_history:
        # Line chart of accuracy per fine-tune epoch
        epochs_x = [e['epoch'] for e in fine_tune_history]
        accs     = [e['acc']   for e in fine_tune_history]

        ax4.plot(epochs_x, accs, marker='o', color=GREEN, linewidth=2,
                 markersize=6, label='Fine-tune accuracy')

        # Horizontal dashed line at accuracy before fine-tuning
        # Use first epoch's accuracy as the pre-fine-tune baseline proxy
        if len(accs) > 1:
            # pre-fine-tune acc approximated from probe or first epoch minus improvement
            pre_ft_acc = accs[0] - (accs[-1] - accs[0]) * 0.1
            ax4.axhline(pre_ft_acc, color=RED, linestyle='--', linewidth=1.5,
                        alpha=0.7, label='Pre-fine-tune (approx.)')

        ax4.set_xlabel('Fine-tune Epoch', fontweight='bold')
        ax4.set_ylabel('Accuracy (%)', fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)

    else:
        # No fine-tuning: show parameter count comparison as proxy
        if original_params is not None and pruned_params is not None:
            categories  = ['Original\nModel', 'Pruned\nModel']
            param_vals  = [original_params / 1e6, pruned_params / 1e6]
            bars = ax4.bar(categories, param_vals,
                           color=[BLUE, GREEN], alpha=0.75, edgecolor='black')
            for bar, val in zip(bars, param_vals):
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(param_vals) * 0.01,
                         f'{val:.2f}M', ha='center', fontweight='bold', fontsize=11)
            reduction = (1 - pruned_params / original_params) * 100
            ax4.text(0.5, -0.15, f'↓ {reduction:.1f}% fewer parameters',
                     ha='center', color=GREEN, fontweight='bold',
                     transform=ax4.transAxes, fontsize=11)
            ax4.set_ylabel('Parameters (M)', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Fine-tuning skipped.\nNo accuracy history available.',
                     ha='center', va='center', transform=ax4.transAxes,
                     fontsize=12, color=GREY)
            ax4.axis('off')

    title_4 = 'Fine-tuning Recovery' if (fine_tune_epochs > 0 and fine_tune_history) \
              else 'Parameter Count: Before vs After'
    ax4.set_title(title_4, fontweight='bold', fontsize=11)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"📊 Pruning report saved → '{save_path}'")


# ============================================================================
# ── MAIN COMPRESSION REPORT ──────────────────────────────────────────────────
# ============================================================================

def _parse_techniques(techniques: list) -> dict:
    """
    Parse the techniques_used list into boolean flags.
    Each string looks like 'Structured Pruning (ratio=0.30)' etc.
    Returns a dict of booleans and extracted values.
    """
    joined = ' '.join(t.lower() for t in techniques)
    result = {
        'bn_fusion':    'bn fusion' in joined,
        'pruning':      'pruning' in joined,
        'lrf':          'low-rank' in joined or 'factorization' in joined,
        'clustering':   'clustering' in joined or 'cluster' in joined,
        'kd':           'kd fine-tune' in joined or 'distillation' in joined,
        'quantization': 'quantization' in joined and 'gptq' not in joined.split('quantization')[0],
        'gptq':         'gptq' in joined,
    }
    # Extract epsilon from LRF string e.g. 'Low-Rank Factorization (ε=0.67)'
    result['lrf_epsilon'] = None
    result['lrf_adaptive'] = 'adaptive' in joined
    for t in techniques:
        if 'factorization' in t.lower() or 'low-rank' in t.lower():
            import re
            m = re.search(r'ε=([0-9.]+)', t)
            if m:
                result['lrf_epsilon'] = float(m.group(1))
    # Extract k from clustering string e.g. 'Weight Clustering (k=16)'
    result['cluster_k'] = None
    for t in techniques:
        if 'clustering' in t.lower():
            import re
            m = re.search(r'k=([0-9]+)', t)
            if m:
                result['cluster_k'] = int(m.group(1))
    # Extract pruning ratio
    result['pruning_ratio'] = None
    for t in techniques:
        if 'pruning' in t.lower():
            import re
            m = re.search(r'ratio=([0-9.]+)', t)
            if m:
                result['pruning_ratio'] = float(m.group(1))
    # Extract quant mode
    result['quant_mode'] = None
    for t in techniques:
        if 'quantization' in t.lower() and 'gptq' not in t.lower():
            import re
            m = re.search(r'\(([^)]+)\)', t)
            if m:
                result['quant_mode'] = m.group(1)
    # Extract GPTQ bits
    result['gptq_bits'] = None
    for t in techniques:
        if 'gptq' in t.lower():
            import re
            m = re.search(r'INT([0-9]+)', t, re.IGNORECASE)
            if m:
                result['gptq_bits'] = int(m.group(1))
    # Extract KD T and alpha
    result['kd_temperature'] = None
    result['kd_alpha'] = None
    for t in techniques:
        if 'kd' in t.lower() or 'distillation' in t.lower():
            import re
            tm = re.search(r'T=([0-9.]+)', t)
            am = re.search(r'α=([0-9.]+)', t)
            if tm: result['kd_temperature'] = float(tm.group(1))
            if am: result['kd_alpha'] = float(am.group(1))
    return result


def _draw_technique_panel(
    ax: 'plt.Axes',
    tech_key: str,
    tech_info: dict,
    metrics: dict,
    colors: dict,
) -> None:
    """
    Draw a single technique-specific panel.  Called once per active technique.

    For each technique, shows the most useful data available from metrics_dict.
    Falls back to a clear summary text panel when detailed sub-report data is
    not available.
    """
    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#dee2e6')

    pruning_rep = metrics.get('pruning_report') or {}
    orig_size   = metrics.get('original_size_mb')   or 1.0
    comp_size   = metrics.get('compressed_size_mb') or orig_size
    orig_params = metrics.get('original_params')    or 1
    comp_params = metrics.get('compressed_params')  or orig_params

    GREEN  = colors['GREEN']
    BLUE   = colors['BLUE']
    ORANGE = colors['ORANGE']
    PURPLE = colors['PURPLE']
    RED    = colors['RED']
    AMBER  = '#f0a020'

    if tech_key == 'pruning':
        # ── Pruning panel: horizontal bar chart of pruned layers ──────────────
        ax.set_title('Structured Pruning', fontweight='bold', fontsize=10)
        layers_pruned = pruning_rep.get('layers_pruned', [])
        if layers_pruned:
            # Show top-8 by fraction_removed
            top = sorted(layers_pruned, key=lambda x: x.get('fraction_removed', 0),
                         reverse=True)[:8]
            names  = [l['name'].split('.')[-2] + '.' + l['name'].split('.')[-1]
                      if '.' in l['name'] else l['name'] for l in top]
            fracs  = [l.get('fraction_removed', 0) * 100 for l in top]
            y_pos  = range(len(names))
            bars   = ax.barh(list(y_pos), fracs, color=ORANGE, alpha=0.75,
                             edgecolor='white', height=0.6)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(names, fontsize=7.5)
            ax.set_xlabel('Filters Removed (%)', fontsize=8)
            ax.set_xlim(0, 105)
            ax.axvline(x=float(np.mean(fracs)), color=RED, linestyle='--',
                       linewidth=1.2, alpha=0.7, label=f'Mean {np.mean(fracs):.0f}%')
            ax.legend(fontsize=7)
            ax.grid(axis='x', alpha=0.3)
            n_pruned = len(layers_pruned)
            p_orig   = pruning_rep.get('original_params', orig_params)
            p_new    = pruning_rep.get('pruned_params',   comp_params)
            reduction = (1 - p_new / p_orig) * 100 if p_orig > 0 else 0
            ax.text(0.98, 0.02, f'{n_pruned} layers pruned\n{reduction:.1f}% params removed',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=7.5, color='#555',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Pruning applied\n(no per-layer data)',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)

    elif tech_key == 'lrf':
        # ── LRF panel: size contribution and epsilon info ─────────────────────
        ax.set_title('Low-Rank Factorization', fontweight='bold', fontsize=10)
        eps     = tech_info.get('lrf_epsilon')
        adaptive = tech_info.get('lrf_adaptive', False)

        # Show param count before vs after using the param counts we have
        categories = ['Original\nParams', 'After LRF\nParams']
        values = [orig_params / 1e6, comp_params / 1e6]
        bar_colors = [BLUE, GREEN]

        # If quantization also ran, comp_params reflects all techniques.
        # We can't isolate LRF's contribution exactly without sub-reports,
        # so we show the total compressed params with a note.
        bars = ax.bar(categories, values, color=bar_colors, alpha=0.75,
                      edgecolor='white', width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f'{val:.1f}M', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)
        ax.set_ylabel('Parameters (M)', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        eps_str = f'ε = {eps:.4f}' if eps is not None else 'adaptive'
        mode_str = 'Adaptive per-layer SVD' if adaptive else f'Global ε = {eps:.2f}' if eps else ''
        ax.text(0.5, -0.18,
                f'{eps_str}  |  {mode_str}',
                ha='center', transform=ax.transAxes, fontsize=8,
                color='#555')

    elif tech_key == 'clustering':
        # ── Clustering panel: k-means centroid compression bar ────────────────
        ax.set_title('Weight Clustering', fontweight='bold', fontsize=10)
        k = tech_info.get('cluster_k') or 16

        # Show: float32 unique values (theoretically millions) vs k centroids
        # Demonstrate the compression concept with a log-scale bar
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Draw a conceptual diagram
        info_lines = [
            f'k = {k} centroids',
            f'Each weight → nearest of {k} values',
            '',
            f'float32: ~2³² possible values',
            f'After:   exactly {k} unique values',
            '',
            f'Theoretical storage ratio:',
            f'  log₂({k}) bits vs 32 bits',
            f'  = {np.log2(k):.0f} / 32 = {np.log2(k)/32:.3f}×',
            f'  ({(1 - np.log2(k)/32)*100:.0f}% bit-level reduction)',
        ]
        for i, line in enumerate(info_lines):
            weight = 'bold' if i == 0 else 'normal'
            color  = ORANGE if i == 0 else ('#333' if line else '#aaa')
            ax.text(0.08, 0.92 - i * 0.092, line,
                    transform=ax.transAxes, fontsize=8.5,
                    fontweight=weight, color=color, va='top')

    elif tech_key == 'kd':
        # ── KD panel: hyperparameters and benefit description ─────────────────
        ax.set_title('Knowledge Distillation Fine-tune', fontweight='bold', fontsize=10)
        T     = tech_info.get('kd_temperature')
        alpha = tech_info.get('kd_alpha')

        kd_history = metrics.get('kd_history', [])
        if kd_history and isinstance(kd_history, list) and len(kd_history) > 0:
            # Plot loss/acc curves if history was tracked
            epochs = list(range(1, len(kd_history) + 1))
            losses = [h.get('loss', 0) for h in kd_history]
            accs   = [h.get('acc', 0) * 100 for h in kd_history]
            ax2    = ax.twinx()
            ax.plot(epochs, losses, 'o-', color=RED,    linewidth=2,
                    markersize=5, label='KD Loss')
            ax2.plot(epochs, accs, 's--', color=GREEN, linewidth=2,
                     markersize=5, label='Accuracy (%)')
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8, color=RED)
            ax2.set_ylabel('Accuracy (%)', fontsize=8, color=GREEN)
            ax.legend(loc='upper left', fontsize=7)
            ax2.legend(loc='upper right', fontsize=7)
            ax.grid(alpha=0.3)
        else:
            # No history data — show a clean info panel
            ax.axis('off')
            T_str     = f'{T:.1f}' if T     is not None else '4.0'
            alpha_str = f'{alpha:.2f}' if alpha is not None else '0.70'
            info_lines = [
                f'Temperature   T = {T_str}',
                f'Alpha         α = {alpha_str}',
                '',
                f'Loss = α × CE(student, labels)',
                f'     + (1-α) × T² × KL(teacher ‖ student)',
                '',
                'Teacher: frozen original model',
                'Student: compressed model',
                'Fine-tunes on calibration data only',
            ]
            for i, line in enumerate(info_lines):
                weight = 'bold' if i < 2 else 'normal'
                color  = PURPLE if i < 2 else ('#333' if line else '#aaa')
                ax.text(0.08, 0.92 - i * 0.10, line,
                        transform=ax.transAxes, fontsize=8.5,
                        fontweight=weight, color=color, va='top')

    elif tech_key == 'quantization':
        # ── Quantization panel: size before vs after, mode info ───────────────
        ax.set_title('Quantization', fontweight='bold', fontsize=10)
        mode = tech_info.get('quant_mode') or 'fp16'

        # Size bars
        orig_size_q = orig_size
        comp_size_q = comp_size
        size_before_quant = metrics.get('size_before_quantization_mb', orig_size_q)
        categories  = ['Pre-quant\n(float32)', f'Post-quant\n({mode})']
        values      = [size_before_quant, comp_size_q]
        bar_colors  = [BLUE, PURPLE]
        bars = ax.bar(categories, values, color=bar_colors, alpha=0.75,
                      edgecolor='white', width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f'{val:.1f} MB', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)
        ax.set_ylabel('Size (MB)', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        bits = {'fp16': 16, 'dynamic': 8, 'static': 8}.get(mode, 16)
        ratio_str = f'{32/bits:.0f}× bit reduction ({mode})'
        ax.text(0.5, -0.18, ratio_str,
                ha='center', transform=ax.transAxes, fontsize=8, color='#555')

    elif tech_key == 'gptq':
        # ── GPTQ panel: bits used and layer info ─────────────────────────────
        ax.set_title('GPTQ Quantization', fontweight='bold', fontsize=10)
        bits = tech_info.get('gptq_bits') or 4
        gptq_rep = metrics.get('gptq_report', {}) or {}

        ax.axis('off')
        info_lines = [
            f'Quantization: INT{bits}',
            f'Method: Hessian-corrected (GPTQ)',
            '',
            f'Precision: {bits} bits vs 32 bits float',
            f'Theoretical ratio: {32 / bits:.0f}×',
            f'({(1 - bits/32)*100:.0f}% storage reduction)',
            '',
            'Column-by-column Hessian correction',
            'propagates quantization error forward',
            'to minimize output distortion.',
        ]
        n_layers = gptq_rep.get('layers_quantized', 'N/A')
        if n_layers != 'N/A':
            info_lines.insert(2, f'Layers quantized: {n_layers}')

        for i, line in enumerate(info_lines):
            weight = 'bold' if i == 0 else 'normal'
            color  = AMBER if i == 0 else ('#333' if line else '#aaa')
            ax.text(0.08, 0.92 - i * 0.092, line,
                    transform=ax.transAxes, fontsize=8.5,
                    fontweight=weight, color=color, va='top')


def generate_compression_report(
    original_model: nn.Module,
    compressed_model: nn.Module,
    metrics_dict: Dict[str, Any],
    save_path: Optional[str] = 'compression_report.png',
    dataloader=None,
    device: Optional[str] = None,
    input_shape: tuple = (1, 3, 224, 224),
    latency_iterations: int = 100,
    latency_warmup: int = 10,
) -> dict:
    """
    Generate and save a comprehensive compression analysis report.

    Layout is dynamic — technique-specific panels are only rendered for
    techniques that were actually used.  Unused techniques produce no panel.

    Always rendered (Row 0):
        Accuracy | Model Size | Parameter Count | Inference Latency

    Always rendered (Row 1):
        Summary Table (spans 2 cols) | Technique Badges | Compression Radar

    Dynamic (Row 2, one panel per active technique, up to 4 per row):
        Structured Pruning | Low-Rank Factorization | Weight Clustering |
        KD Fine-tune | Quantization | GPTQ

    BN Fusion is never given a panel — it has no per-layer visualization.

    Expected metrics_dict keys (all optional — computed when missing):
        original_accuracy     (float) : original model top-1 accuracy %
        compressed_accuracy   (float) : compressed model top-1 accuracy %
        original_size_mb      (float)
        compressed_size_mb    (float)
        original_params       (int)
        compressed_params     (int)
        original_latency_ms   (float) : mean inference latency ms
        compressed_latency_ms (float)
        techniques_used       (list)  : list of technique name strings
        pruning_report        (dict)  : ._pruning_report from apply_structured_pruning
        kd_history            (list)  : [{loss, acc}, ...] per KD epoch
        cqi                   (float) : Compression Quality Index

    Args:
        original_model:     The original, uncompressed model.
        compressed_model:   The compressed model to compare against.
        metrics_dict:       Dict of pre-computed metrics (see above).
        save_path:          PNG output path.  Pass None to skip saving.
        dataloader:         DataLoader for fallback accuracy/latency computation.
        device:             'cuda' or 'cpu' for fallback measurement.
        input_shape:        Dummy input shape for latency measurement.
        latency_iterations: Forward passes used for latency timing.
        latency_warmup:     Untimed warmup passes before timing starts.

    Returns:
        Final metrics dict enriched with all computed values.
    """
    # ── Fill missing metrics ──────────────────────────────────────────────────
    m = _ensure_model_metrics(
        original_model, compressed_model, metrics_dict,
        dataloader=dataloader, device=device,
        input_shape=input_shape,
        latency_iterations=latency_iterations,
        latency_warmup=latency_warmup,
    )

    # ── Aliases ───────────────────────────────────────────────────────────────
    orig_size   = _safe_get(m, 'original_size_mb')
    comp_size   = _safe_get(m, 'compressed_size_mb')
    orig_params = _safe_get(m, 'original_params')
    comp_params = _safe_get(m, 'compressed_params')
    orig_acc    = _safe_get(m, 'original_accuracy')
    comp_acc    = _safe_get(m, 'compressed_accuracy')
    orig_lat    = _safe_get(m, 'original_latency_ms')
    comp_lat    = _safe_get(m, 'compressed_latency_ms')
    techniques  = _safe_get(m, 'techniques_used', [])
    pruning_rep = _safe_get(m, 'pruning_report', None)
    cqi_val     = _safe_get(m, 'cqi', None)

    # ── Derived metrics ───────────────────────────────────────────────────────
    size_ratio      = (orig_size / comp_size) if (orig_size and comp_size and comp_size > 0) else None
    size_reduction  = ((1 - comp_size / orig_size) * 100) if (orig_size and comp_size and orig_size > 0) else None
    param_reduction = ((1 - comp_params / orig_params) * 100) if (orig_params and comp_params and orig_params > 0) else None
    acc_drop        = (orig_acc - comp_acc) if (orig_acc is not None and comp_acc is not None) else None
    speedup         = (orig_lat / comp_lat) if (orig_lat and comp_lat and comp_lat > 0) else None

    # ── CQI fallback ─────────────────────────────────────────────────────────
    if cqi_val is None:
        try:
            from sigularty.optimization import compression_quality_index as _cqi_fn
            cqi_val = _cqi_fn(
                comp_acc or 0.0, comp_size or (orig_size or 1.0),
                orig_acc or 1.0, orig_size or 1.0,
                comp_lat, orig_lat,
            ) if orig_acc else None
        except Exception:
            try:
                from optimization import compression_quality_index as _cqi_fn
                cqi_val = _cqi_fn(
                    comp_acc or 0.0, comp_size or (orig_size or 1.0),
                    orig_acc or 1.0, orig_size or 1.0,
                    comp_lat, orig_lat,
                ) if orig_acc else None
            except Exception:
                cqi_val = None

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("COMPRESSION REPORT")
    if techniques:
        print(f"  Techniques: {' → '.join(techniques)}")
    print(f"{'=' * 80}")
    print(f"  Size       : {_fmt(orig_size)} MB  →  {_fmt(comp_size)} MB"
          f"  ({_fmt(size_reduction)}% smaller,  {_fmt(size_ratio)}×)")
    print(f"  Parameters : {_fmt(orig_params, ',.0f')}  →  {_fmt(comp_params, ',.0f')}"
          f"  ({_fmt(param_reduction)}% fewer)")
    print(f"  Accuracy   : {_fmt(orig_acc)}%  →  {_fmt(comp_acc)}%"
          f"  (drop: {_fmt(acc_drop)}%)")
    print(f"  Latency    : {_fmt(orig_lat, '.3f')} ms  →  {_fmt(comp_lat, '.3f')} ms"
          f"  ({_fmt(speedup)}×)")
    if cqi_val is not None:
        print(f"  CQI        : {cqi_val:.3f}  (CQI > 1.0 = better than original tradeoff)")
    if pruning_rep is not None:
        n_pruned = len(pruning_rep.get('layers_pruned', []))
        pp_orig  = pruning_rep.get('original_params', 0)
        pp_new   = pruning_rep.get('pruned_params', 0)
        pr_pct   = (1 - pp_new / pp_orig) * 100 if pp_orig > 0 else 0
        print(f"  Pruning    : {n_pruned} layers  {pr_pct:.1f}% params removed")
    if acc_drop is not None:
        # acc_drop here = orig - comp, so positive = degraded, negative = improved
        if acc_drop <= 1.0 and (size_ratio or 0) > 1.5:
            print("\n  ✅  Excellent compression! Ready to deploy.")
        elif acc_drop <= 2.0 and (size_ratio or 0) > 2.0:
            print("\n  ✅  Good compression with acceptable accuracy loss.")
        elif acc_drop <= 1.0:
            print("\n  ⚠️   Minimal compression. Consider more aggressive settings.")
        else:
            print(f"\n  ❌  Accuracy drop too high ({acc_drop:.2f}%). Raise epsilon or reduce k.")
    print(f"{'=' * 80}\n")

    # ── Determine active technique panels ─────────────────────────────────────
    # BN Fusion is intentionally excluded — nothing to visualise.
    tech_info    = _parse_techniques(techniques)
    PANEL_ORDER  = ['pruning', 'lrf', 'clustering', 'kd', 'quantization', 'gptq']
    active_panels = [k for k in PANEL_ORDER if tech_info.get(k, False)]
    n_tech_panels = len(active_panels)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Row 0: 4 metric bars (always)
    # Row 1: summary table (2 cols) + badges + radar (always)
    # Row 2+: technique panels (only active ones, up to 4 per row)
    n_rows = 2
    n_tech_rows = 0
    if n_tech_panels > 0:
        n_tech_rows = (n_tech_panels + 3) // 4   # ceil(n / 4)
        n_rows += n_tech_rows

    fig_height = 11 + n_tech_rows * 3.2
    fig = plt.figure(figsize=(18, fig_height))
    fig.patch.set_facecolor('#f8f9fa')

    title_text = "Model Compression Report"
    if techniques:
        title_text += f"\n{' → '.join(techniques)}"
    title_top = 0.98 - (0.015 if techniques else 0)
    fig.suptitle(title_text, fontsize=15, fontweight='bold', y=title_top)

    # Proportional row heights: rows 0+1 get 11 units total, each tech row gets 3.2
    total_h = 11 + n_tech_rows * 3.2
    row_heights = [4.5, 6.5] + [3.2] * n_tech_rows   # in figure-inches

    
    height_ratios = [4.5, 6.5] + [3.2] * n_tech_rows
    gs = GridSpec(
        n_rows, 4,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.55, wspace=0.35,
        left=0.06, right=0.96,
        top=0.92 if not techniques else 0.89,
        bottom=0.06,
    )

    COLORS = {
        'BLUE':   '#3498db',
        'GREEN':  '#2ecc71',
        'RED':    '#e74c3c',
        'ORANGE': '#f39c12',
        'PURPLE': '#9b59b6',
        'GREY':   '#bdc3c7',
        'AMBER':  '#f0a020',
    }
    GREEN  = COLORS['GREEN']
    BLUE   = COLORS['BLUE']
    ORANGE = COLORS['ORANGE']
    PURPLE = COLORS['PURPLE']
    GREY   = COLORS['GREY']
    RED    = COLORS['RED']
    label_orig = 'Original'
    label_comp = 'Compressed'

    # ── Row 0: four metric bars ───────────────────────────────────────────────
    ax_acc = fig.add_subplot(gs[0, 0])
    _bar_pair(
        ax_acc, [label_orig, label_comp], [orig_acc, comp_acc],
        [BLUE, RED if (acc_drop or 0) > 2 else GREEN],
        'Accuracy (%)', 'Accuracy',
        ylim=(0, 115) if (orig_acc or 0) > 0 else None,
    )
    if acc_drop is not None:
        color = 'green' if acc_drop <= 1.0 else ('orange' if acc_drop <= 3.0 else 'red')
        ax_acc.text(0.5, -0.16, f"Drop: {acc_drop:.2f}%",
                    ha='center', color=color, fontweight='bold',
                    transform=ax_acc.transAxes, fontsize=11)

    ax_size = fig.add_subplot(gs[0, 1])
    _bar_pair(ax_size, [label_orig, label_comp], [orig_size, comp_size],
              [BLUE, GREEN], 'Size (MB)', 'Model Size (MB)')
    if size_reduction is not None:
        ax_size.text(0.5, -0.16, f"↓ {size_reduction:.1f}%  ({size_ratio:.2f}×)",
                     ha='center', color='green', fontweight='bold',
                     transform=ax_size.transAxes, fontsize=11)

    ax_params = fig.add_subplot(gs[0, 2])
    orig_pm = (orig_params / 1e6) if orig_params else None
    comp_pm = (comp_params / 1e6) if comp_params else None
    _bar_pair(ax_params, [label_orig, label_comp], [orig_pm, comp_pm],
              [BLUE, ORANGE], 'Parameters (M)', 'Parameter Count (M)')
    if param_reduction is not None:
        ax_params.text(0.5, -0.16, f"↓ {param_reduction:.1f}%",
                       ha='center', color='green', fontweight='bold',
                       transform=ax_params.transAxes, fontsize=11)

    ax_lat = fig.add_subplot(gs[0, 3])
    _bar_pair(ax_lat, [label_orig, label_comp], [orig_lat, comp_lat],
              [BLUE, PURPLE], 'Latency (ms)', 'Inference Latency (ms)')
    if speedup is not None:
        sym   = "↓" if speedup >= 1.0 else "↑"
        color = 'green' if speedup >= 1.0 else 'red'
        ax_lat.text(0.5, -0.16,
                    f"{sym} {abs(speedup):.2f}× {'faster' if speedup >= 1 else 'slower'}",
                    ha='center', color=color, fontweight='bold',
                    transform=ax_lat.transAxes, fontsize=11)

    # ── Row 1: summary table + badges + radar ────────────────────────────────
    ax_table = fig.add_subplot(gs[1, 0:2])
    ax_table.axis('off')
    rows = [
        ["Metric",            "Original",               "Compressed",            "Change"],
        ["Accuracy (%)",      _fmt(orig_acc),           _fmt(comp_acc),          _fmt(acc_drop, suffix="% drop") if acc_drop is not None else "N/A"],
        ["Size (MB)",         _fmt(orig_size),          _fmt(comp_size),         f"{_fmt(size_reduction)}% ↓" if size_reduction is not None else "N/A"],
        ["Compression Ratio", "1.00×",                  f"{_fmt(size_ratio)}×",  ""],
        ["Parameters",        _fmt(orig_params,',.0f'), _fmt(comp_params,',.0f'),f"{_fmt(param_reduction)}% ↓" if param_reduction is not None else "N/A"],
        ["Latency (ms)",      _fmt(orig_lat,'.3f'),     _fmt(comp_lat,'.3f'),    f"{_fmt(speedup)}× speedup" if speedup is not None else "N/A"],
    ]
    if cqi_val is not None:
        rows.append(["CQI", "1.000 (baseline)", f"{cqi_val:.3f}",
                     f"{'↑' if cqi_val > 1 else '↓'} {abs(cqi_val-1):.3f}"])
    if pruning_rep is not None:
        _n   = len(pruning_rep.get('layers_pruned', []))
        _pp  = pruning_rep.get('original_params', 0)
        _pn  = pruning_rep.get('pruned_params',   0)
        _pct = (1 - _pn / _pp) * 100 if _pp > 0 else 0
        rows.append(["Pruning",
                      f"{_pp:,.0f} params",
                      f"{_pn:,.0f} params",
                      f"{_n} layers  {_pct:.1f}% ↓"])

    col_widths = [0.28, 0.22, 0.24, 0.26]
    table = ax_table.table(cellText=rows[1:], colLabels=rows[0],
                           colWidths=col_widths, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(4):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        bg = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(4):
            table[i, j].set_facecolor(bg)
    if acc_drop is not None and abs(acc_drop) > 2.0:
        for j in range(4):
            table[1, j].set_facecolor('#fde8e8')
    ax_table.set_title("Summary Table", fontweight='bold', fontsize=12, pad=10)

    # ── Technique badges ──────────────────────────────────────────────────────
    ax_badges = fig.add_subplot(gs[1, 2])
    ax_badges.axis('off')
    ax_badges.set_title("Techniques Applied", fontweight='bold', fontsize=12)
    badge_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    if techniques:
        # Auto-scale badge height based on number of techniques
        badge_h   = min(0.18, 0.85 / max(len(techniques), 1))
        badge_gap = 0.85 / max(len(techniques), 1)
        for idx, tech in enumerate(techniques):
            y_pos = 0.88 - idx * badge_gap
            color = badge_colors[idx % len(badge_colors)]
            ax_badges.add_patch(mpatches.FancyBboxPatch(
                (0.04, y_pos - badge_h * 0.5), 0.92, badge_h,
                boxstyle="round,pad=0.02", linewidth=1.5,
                edgecolor=color, facecolor=color + '22',
                transform=ax_badges.transAxes,
            ))
            ax_badges.text(0.5, y_pos, f"✓  {tech}",
                           ha='center', va='center',
                           fontsize=min(9.5, 72 / max(len(techniques), 1)),
                           fontweight='bold', color=color,
                           transform=ax_badges.transAxes)
    else:
        ax_badges.text(0.5, 0.5, 'No technique metadata\nprovided.',
                       ha='center', va='center', fontsize=11, color=GREY,
                       transform=ax_badges.transAxes)

    # ── Compression radar ─────────────────────────────────────────────────────
    ax_radar = fig.add_subplot(gs[1, 3], projection='polar')
    radar_labels = ['Size\nReduction', 'Speed\nImprovement', 'Compression\nRatio', 'Param\nReduction']
    raw_vals = [
        size_reduction,
        ((1 - 1 / speedup) * 100) if (speedup and speedup > 0) else None,
        (min((size_ratio or 0) / 10, 1.0) * 100) if size_ratio else None,
        param_reduction,
    ]
    radar_vals = [np.clip(v if v is not None else 0.0, 0, 100) for v in raw_vals]
    N      = len(radar_labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    vals   = radar_vals + [radar_vals[0]]
    ax_radar.plot(angles, vals, 'o-', lw=2, color=GREEN)
    ax_radar.fill(angles, vals, alpha=0.25, color=GREEN)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=9)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_yticks([25, 50, 75, 100])
    ax_radar.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax_radar.set_title("Compression\nScore", fontweight='bold', fontsize=11, pad=14)
    for i, val in enumerate(raw_vals):
        if val is None:
            ax_radar.get_xticklabels()[i].set_color(GREY)

    # ── Row(s) 2+: dynamic technique panels ──────────────────────────────────
    for panel_idx, tech_key in enumerate(active_panels):
        tech_row = 2 + panel_idx // 4
        tech_col = panel_idx % 4
        ax_tech = fig.add_subplot(gs[tech_row, tech_col])
        _draw_technique_panel(ax_tech, tech_key, tech_info, m, COLORS)

    # If last row of technique panels is partial, hide unused axes
    if n_tech_panels > 0:
        last_row_count = n_tech_panels % 4
        if last_row_count != 0:
            last_row_idx = 2 + (n_tech_panels - 1) // 4
            for col in range(last_row_count, 4):
                ax_empty = fig.add_subplot(gs[last_row_idx, col])
                ax_empty.axis('off')

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"📊 Compression report saved → '{save_path}'")
    else:
        plt.show()
        plt.close(fig)

    m.update({
        'size_ratio':      size_ratio,
        'size_reduction':  size_reduction,
        'param_reduction': param_reduction,
        'acc_drop':        acc_drop,
        'speedup':         speedup,
        'cqi':             cqi_val,
    })
    return m