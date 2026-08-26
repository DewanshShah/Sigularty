"""
Auto hyperparameter search.
Called by compress() when auto_search=True.
Wraps optimization.py without exposing it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def run_auto_search(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    use_lrf: bool,
    use_pruning: bool,
    lrf_epsilon: float,
    pruning_ratio: float,
    pruning_max_ratio: float,
    pruning_model_type: str,
    pruning_num_classes: int,
    pruning_cal_batches: int,
    lrf_min_layer_size: int,
    input_shape: tuple,
    num_trials: int,
    baseline_accuracy: Optional[float],
    baseline_size: float,
    accuracy_drop_threshold: float,
    **kwargs,
) -> Tuple[float, float, int]:
    """
    Run hyperparameter search and return (lrf_epsilon, pruning_ratio, pruning_steps).
    """
    import importlib
    _opt = importlib.import_module("sigularty.optimization")
    _hf  = importlib.import_module("sigularty.helper_functions")

    # Measure baseline if not already done
    if baseline_accuracy is None:
        baseline_accuracy = _hf.measure_accuracy(model, dataloader, device)

    try:
        from sigularty.helper_functions import measure_latency
        _lat = measure_latency(model, input_shape, device, num_iterations=10, warmup=3)
        baseline_latency = _lat["mean_ms"]
    except Exception:
        baseline_latency = 1.0

    best_lrf_eps     = lrf_epsilon
    best_prune_ratio = pruning_ratio
    best_prune_steps = 1
    half             = max(5, num_trials // 2)

    if use_lrf:
        print("\n[Auto Search] Searching for optimal LRF epsilon...")
        try:
            optimal_eps, _, _ = _opt.find_optimal_epsilon_smart(
                model=model,
                test_loader=dataloader,
                device=device,
                baseline_accuracy=baseline_accuracy,
                baseline_size=baseline_size,
                baseline_latency_ms=baseline_latency,
                num_trials=half,
                cache_path="compressiontoolkit_eps_cache.json",
                min_layer_size=lrf_min_layer_size,
                min_rank=kwargs.get('lrf_min_rank', 1),
                skip_large_kernels=kwargs.get('lrf_skip_large_kernels', False),
                accuracy_drop_threshold=accuracy_drop_threshold,
            )
            if optimal_eps is not None:
                best_lrf_eps = optimal_eps
                print(f"[Auto Search] Best LRF epsilon: {best_lrf_eps:.4f}")
            else:
                print("[Auto Search] LRF search found no improvement — keeping default epsilon.")
        except Exception as e:
            print(f"[Auto Search] LRF search failed ({e}) — keeping default epsilon.")

    if use_pruning:
        print("\n[Auto Search] Searching for optimal pruning ratio...")
        try:
            optimal_params, _, _ = _opt.find_optimal_pruning_params(
                model=model,
                dataloader=dataloader,
                test_loader=dataloader,
                device=device,
                baseline_accuracy=baseline_accuracy,
                baseline_size=baseline_size,
                baseline_latency_ms=baseline_latency,
                num_trials=half,
                cache_path="compressiontoolkit_prune_cache.json",
                model_type=pruning_model_type,
                num_classes=pruning_num_classes,
                num_calibration_batches=pruning_cal_batches,
                max_pruning_ratio=pruning_max_ratio,
                accuracy_drop_threshold=accuracy_drop_threshold,
            )
            if optimal_params is not None:
                best_prune_ratio = optimal_params.get("pruning_ratio", pruning_ratio)
                best_prune_steps = optimal_params.get("iterative_steps", 1)
                print(f"[Auto Search] Best pruning ratio: {best_prune_ratio:.4f}, "
                      f"steps: {best_prune_steps}")
            else:
                print("[Auto Search] Pruning search found no viable config — keeping defaults.")
        except Exception as e:
            print(f"[Auto Search] Pruning search failed ({e}) — keeping defaults.")

    return best_lrf_eps, best_prune_ratio, best_prune_steps