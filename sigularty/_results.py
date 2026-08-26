"""
Result objects returned by compress() and analyze().
These are the only data structures customers ever see.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CompressionResult:
    """
    Returned by compress().  All attributes are read-only after construction.

    Attributes
    ----------
    model : nn.Module
        The compressed model, on the same device it was received on.
        Drop-in replacement for the original.
    compression_ratio : float
        size_original / size_compressed.  E.g. 3.8 means 3.8× smaller.
    accuracy_retention : float
        Percentage of the original model's accuracy retained.
        E.g. 96.4 means the compressed model achieves 96.4% of the
        original's top-1 accuracy.  None if no dataloader was provided
        for evaluation.
    size_original_mb : float
        Original model disk size in MB (parameters + buffers).
    size_compressed_mb : float
        Compressed model disk size in MB.
    latency_speedup : float
        original_latency / compressed_latency.  > 1.0 = faster.
        None if latency was not measured.
    techniques_applied : list[str]
        Human-readable list of techniques that ran, in pipeline order.
    report_path : str or None
        Absolute path to the saved PNG compression report.
        None if save_report=False or report generation failed.
    pruning_report : dict or None
        Behavioral probe results from structured pruning (KL divergence,
        severity, layers pruned).  None if pruning was not used.
    """
    model: Any                              # nn.Module — typed as Any to avoid torch import
    compression_ratio: float
    accuracy_retention: Optional[float]
    size_original_mb: float
    size_compressed_mb: float
    latency_speedup: Optional[float]
    techniques_applied: List[str]
    report_path: Optional[str]
    pruning_report: Optional[Dict[str, Any]]

    def __repr__(self) -> str:
        acc = f"{self.accuracy_retention:.1f}%" if self.accuracy_retention is not None else "N/A"
        spd = f"{self.latency_speedup:.2f}x" if self.latency_speedup is not None else "N/A"
        return (
            f"CompressionResult(\n"
            f"  compression_ratio    = {self.compression_ratio:.2f}x\n"
            f"  accuracy_retention   = {acc}\n"
            f"  size                 = {self.size_original_mb:.1f} MB → {self.size_compressed_mb:.1f} MB\n"
            f"  latency_speedup      = {spd}\n"
            f"  techniques_applied   = {self.techniques_applied}\n"
            f"  report_path          = {self.report_path!r}\n"
            f")"
        )


@dataclass
class LayerSignal:
    """
    Per-layer analysis signal returned by analyze().

    Attributes
    ----------
    name : str
        Dotted module path, e.g. 'features.3.conv'.
    layer_type : str
        PyTorch class name, e.g. 'Conv2d', 'Linear'.
    params : int
        Total parameter count for this layer.
    lrf_recommended : bool
        True if the layer has a low-rank structure that LRF can exploit.
    lrf_epsilon : float or None
        Analytically optimal epsilon for this layer (from SVD energy).
        None if not applicable.
    prunable : bool
        True if the layer is a grouped-1 Conv2d that Torch-Pruning can prune.
    """
    name: str
    layer_type: str
    params: int
    lrf_recommended: bool
    lrf_epsilon: Optional[float]
    prunable: bool


@dataclass
class AnalysisResult:
    """
    Returned by analyze().

    Attributes
    ----------
    recommended_techniques : list[str]
        Ordered list of technique names recommended for this model,
        from highest to lowest expected benefit.
    per_layer_signals : list[LayerSignal]
        One entry per eligible layer, sorted by parameter count descending.
    model_size_mb : float
        Original model size in MB.
    total_params : int
        Total trainable parameter count.
    lrf_eligible_layers : int
        Number of layers eligible for Low-Rank Factorization.
    prunable_layers : int
        Number of Conv2d layers (groups=1) Torch-Pruning can handle.
    notes : list[str]
        Human-readable notes and warnings about the model's compressibility.
    """
    recommended_techniques: List[str]
    per_layer_signals: List[LayerSignal]
    model_size_mb: float
    total_params: int
    lrf_eligible_layers: int
    prunable_layers: int
    notes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"AnalysisResult(\n"
            f"  model_size_mb          = {self.model_size_mb:.1f} MB\n"
            f"  total_params           = {self.total_params:,}\n"
            f"  lrf_eligible_layers    = {self.lrf_eligible_layers}\n"
            f"  prunable_layers        = {self.prunable_layers}\n"
            f"  recommended_techniques = {self.recommended_techniques}\n"
            f"  notes                  = {self.notes}\n"
            f")"
        )