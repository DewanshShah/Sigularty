"""
sigularty
==================
Neural network compression toolkit.

Public interface:
    compress(model, dataloader, **kwargs)       -> CompressionResult
    finetune(model, train_loader, **kwargs)      -> nn.Module
    find_best_lr(model, dataloader, **kwargs)    -> float
    analyze(model, dataloader=None, **kwargs)    -> AnalysisResult
    load_from_registry(model_name, **kwargs)     -> RegistryResult
    plot_compression_report(result, model, dl)   -> str (PNG path)
    plot_pruning_report(result)                  -> str or None
    plot_epsilon_landscape(...)                  -> str (PNG path)

Everything else (pipeline internals, search algorithms, visualisation) lives
in this package as plain Python modules and isn't re-exported at this top
level — see compression.py, optimization.py, helper_functions.py,
model_registry.py, and visualization.py directly for lower-level access.

NOTE (v0.0.1): load_model_and_data() has been REMOVED from the public API.
It duplicated load_from_registry() (same job — load a registry model plus
its paired dataset in one call) but returned a plain 4-tuple instead of a
named RegistryResult dataclass. load_from_registry() is the sole supported
entry point for this going forward. If you were calling
load_model_and_data(name), the direct replacement is:

    r = load_from_registry(name)
    model, train_loader, test_loader, num_classes = (
        r.model, r.train_loader, r.test_loader, r.num_classes
    )

The underlying registry loader in model_registry.py (load_model_and_data)
is unaffected by this change — only the sigularty._api wrapper of the same
name was removed from the public surface.
"""

from sigularty._api import (
    compress,
    analyze,
    finetune,
    find_best_lr,
    load_from_registry,
    plot_compression_report,
    plot_pruning_report,
    plot_epsilon_landscape,
    CompressionResult,
    AnalysisResult,
    LayerSignal,
    RegistryResult,
)

__all__ = [
    "compress",
    "analyze",
    "finetune",
    "find_best_lr",
    "load_from_registry",
    "plot_compression_report",
    "plot_pruning_report",
    "plot_epsilon_landscape",
    "CompressionResult",
    "AnalysisResult",
    "LayerSignal",
    "RegistryResult",
]

__version__ = "1.0.1"