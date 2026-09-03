"""
sigularty/_api.py
==========================
Public Python wrapper around sigularty's internal modules.

This file is a thin orchestration layer — it does not implement compression
algorithms itself. The actual logic lives in the sibling modules:
compression.py, optimization.py, helper_functions.py, model_registry.py,
visualization.py.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


# ============================================================================
# ── RESULT TYPES ─────────────────────────────────────────────────────────────
# ============================================================================

@dataclass
class LayerSignal:
    """Per-layer analysis produced by analyze()."""
    name:        str
    layer_type:  str    # 'Conv2d' or 'Linear'
    lrf_epsilon: float  # analytically optimal epsilon from SVD energy
    prunable:    bool   # True for non-grouped Conv2d only
    size_kb:     float  # weight tensor size in kilobytes


@dataclass
class AnalysisResult:
    """
    Returned by analyze().  Describes the model and recommends techniques.
    """
    size_mb:                float
    num_parameters:         int
    architecture_type:      str          # 'cnn' | 'transformer' | 'hybrid' | 'unknown'
    recommended_techniques: List[str]    # e.g. ['use_lrf', 'use_clustering', ...]
    per_layer_signals:      List[LayerSignal]
    accuracy:               Optional[float] = None  # top-1% if dataloader given

    def __repr__(self) -> str:
        recs = ", ".join(self.recommended_techniques)
        acc  = f"{self.accuracy:.2f}%" if self.accuracy is not None else "not measured"
        return (
            f"AnalysisResult("
            f"size={self.size_mb:.1f} MB, "
            f"params={self.num_parameters:,}, "
            f"arch={self.architecture_type}, "
            f"accuracy={acc}, "
            f"recommended=[{recs}])"
        )


@dataclass
class CompressionResult:
    """
    Returned by compress().

    Attributes
    ----------
    model               Compressed nn.Module — drop-in replacement for original.
    compression_ratio   orig_size / comp_size.  3.8 = 3.8× smaller.
    accuracy_retention  (compressed_acc / original_acc) × 100.
                        96.4 means the compressed model achieves 96.4% of
                        the original model's accuracy, not an absolute number.
                        Both accuracy values are measured on test_loader —
                        a genuine held-out set — never on the data the model
                        is fine-tuned on.
    size_original_mb    Original model size in MB.
    size_compressed_mb  Compressed model size in MB.
    original_accuracy   Original model top-1 accuracy % on test_loader.
    compressed_accuracy Compressed model top-1 accuracy % on test_loader.
    original_latency_ms Mean inference latency (ms) of original model.
    compressed_latency_ms Mean inference latency (ms) of compressed model.
    latency_speedup     original_lat / compressed_lat. >1.0 = faster.
    cqi                 Compression Quality Index (accuracy × size × latency).
    techniques_applied  List of technique names in pipeline order. Only
                        techniques that actually survived their accuracy
                        gate are listed — a reverted technique is not
                        "applied".
    report_path         Absolute path to saved PNG report, or None.
    pruning_report      Dict with per-layer pruning details, or None (also
                        None if pruning ran but was reverted by the gate).
    """
    model:                nn.Module
    compression_ratio:    float
    accuracy_retention:   float
    size_original_mb:     float
    size_compressed_mb:   float
    original_accuracy:    float
    compressed_accuracy:  float
    original_latency_ms:  float
    compressed_latency_ms: float
    latency_speedup:      float
    cqi:                  float
    techniques_applied:   List[str]
    report_path:          Optional[str]
    pruning_report:       Optional[dict]

    def __repr__(self) -> str:
        return (
            f"CompressionResult("
            f"ratio={self.compression_ratio:.2f}x, "
            f"accuracy_retention={self.accuracy_retention:.1f}%, "
            f"size={self.size_original_mb:.1f}→{self.size_compressed_mb:.1f} MB, "
            f"speedup={self.latency_speedup:.2f}x, "
            f"cqi={self.cqi:.3f})"
        )


# ============================================================================
# ── PUBLIC API ────────────────────────────────────────────────────────────────
# ============================================================================


# ============================================================================
# ── REGISTRY SHORTCUT: load_from_registry ────────────────────────────────────
# ============================================================================

@dataclass
class RegistryResult:
    """
    Returned by load_from_registry().

    Attributes
    ----------
    model          : Pretrained nn.Module, already on `device`, ready to compress.
    train_loader   : DataLoader for the model's associated training dataset.
    test_loader    : DataLoader for the model's associated test dataset.
    num_classes    : Number of output classes for this model/dataset pair.
    model_name     : The registry key that was used (e.g. 'resnet50').
    dataset_name   : The dataset key (e.g. 'cifar100', 'flowers102', 'sst2').
    """
    model:        nn.Module
    train_loader: object
    test_loader:  object
    num_classes:  int
    model_name:   str
    dataset_name: str

    def __repr__(self) -> str:
        return (
            f"RegistryResult("
            f"model={self.model_name}, "
            f"dataset={self.dataset_name}, "
            f"num_classes={self.num_classes})"
        )


def load_from_registry(
    model_name:    str,
    *,
    device:        Optional[str] = None,
    train_sample:  Optional[int] = None,
    test_sample:   Optional[int] = 500,
    batch_size:    int           = 32,
    model_path:    Optional[str] = None,
    force_retrain: bool          = False,
    pretrain_epochs: int         = 10,
    pretrain_lr:     float       = 1e-4,
) -> RegistryResult:
    """
    Load a model AND its paired dataset from the registry in one call.

    This is the sole supported entry point for registry-based loading — see
    this file's module docstring.

    Handles everything automatically:
      - Downloads the dataset paired with the model (CIFAR-10, CIFAR-100,
        Flowers-102, or SST-2) and builds DataLoaders.
      - Instantiates the model with ImageNet / HuggingFace pretrained weights.
      - Loads from checkpoint if one exists at model_path, otherwise fine-tunes
        from pretrained and saves the best checkpoint.

    After calling this, you can pass the result directly into compress():

        r = load_from_registry('resnet50', device='cuda')
        result = compress(r.model, r.train_loader, test_loader=r.test_loader,
                          num_classes=r.num_classes)

    Parameters
    ----------
    model_name      : Registry key. See supported models below.
    device          : 'cuda' or 'cpu'. Auto-detected when None.
    train_sample    : Max training samples. None = full dataset.
    test_sample     : Max test samples. None = full test set. Default 500.
    batch_size      : DataLoader batch size.
    model_path      : Path to save/load .pth checkpoint.
                      Defaults to 'models/<model_name>.pth'.
    force_retrain   : Re-train even if a checkpoint already exists.
    pretrain_epochs : Fine-tuning epochs when training from scratch.
    pretrain_lr     : Learning rate when training from scratch.

    Supported models
    ----------------
    CNNs (CIFAR-10):    custom_cnn
    CNNs (CIFAR-100):   resnet18, resnet50, resnext50_32x4d, wide_resnet50_2,
                        vgg16, densenet121, convnext_tiny, regnet_y_400mf,
                        shufflenet_v2_x1_0, squeezenet1_1, mobilenet_v3_large,
                        vit_b_16, swin_t
    CNNs (Flowers-102): efficientnet_b0, inception_v3
    NLP (SST-2):        bert_base, distilbert, roberta_base, albert_base,
                        distilgpt2

    Returns
    -------
    RegistryResult with .model, .train_loader, .test_loader, .num_classes,
    .model_name, .dataset_name
    """
    from sigularty.helper_functions import (
        setup_data_for_model,
        load_or_train_from_registry,
    )
    from sigularty.model_registry import get_model_meta

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    meta         = get_model_meta(model_name)
    num_classes  = meta['num_classes']
    dataset_name = meta['dataset']

    # Use registry-recommended lr/epochs if caller didn't override them
    rec = meta.get('recommended', {})
    if pretrain_epochs == 10 and 'pretrain_epochs' in rec:
        pretrain_epochs = rec['pretrain_epochs']
    if pretrain_lr == 1e-4 and 'pretrain_lr' in rec:
        pretrain_lr = rec['pretrain_lr']

    # Default checkpoint path
    if model_path is None:
        model_path = f'models/{model_name}.pth'

    print(f"\n[load_from_registry] {model_name} + {dataset_name}")
    print(f"  device={device}  num_classes={num_classes}  checkpoint={model_path}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    train_loader, test_loader = setup_data_for_model(
        model_name=model_name,
        train_sample=train_sample,
        test_sample=test_sample,
        batch_size=batch_size,
        device=device,
    )

    # ── 2. Load or train model ────────────────────────────────────────────────
    model = load_or_train_from_registry(
        model_name=model_name,
        train_loader=train_loader,
        test_loader=test_loader,
        model_path=model_path,
        epochs=pretrain_epochs,
        lr=pretrain_lr,
        device=device,
        force_retrain=force_retrain,
    )

    return RegistryResult(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        model_name=model_name,
        dataset_name=dataset_name,
    )

def compress(
    model:      nn.Module,
    dataloader: DataLoader,
    *,
    # ── Evaluation data ───────────────────────────────────────────────────────
    # A genuine held-out set. Every accuracy measurement in this function —
    # baseline, both hyperparameter searches, every technique's gate, the
    # final report — reads accuracy from test_loader, never from dataloader.
    # If you don't supply one, compress() derives one automatically by
    # splitting dataloader's own dataset 70/30 (train/held-out, fixed seed)
    # and prints a warning when it does — see _auto_split_dataloader below.
    test_loader:             Optional[DataLoader] = None,
    # ── Device ────────────────────────────────────────────────────────────────
    device:                  Optional[str]   = None,
    # ── Model metadata ────────────────────────────────────────────────────────
    model_type:              str             = 'unknown',
    num_classes:             int             = 10,
    # ── Pre-training (fine-tune before compressing) ───────────────────────────
    # Use these when the model has not been trained on your target dataset yet
    # (e.g. freshly loaded from load_from_registry() with no checkpoint).
    pretrain_epochs:         int             = 0,      # 0 = skip pre-training
    pretrain_lr:             float           = 1e-3,
    pretrain_test_loader:    Optional[DataLoader] = None,  # if None uses test_loader
    # ── Hyperparameter searches (optional — adds evaluations before pipeline) ─
    find_optimal_epsilon:    bool            = False,
    find_optimal_pruning:    bool            = False,
    epsilon_search_trials:   int             = 15,
    pruning_search_trials:   int             = 16,
    pruning_search_ft_epochs: int            = 1,
    pruning_search_ft_lr:    float           = 1e-4,
    accuracy_drop_threshold: float           = 5.0,
    # Direct pp value, not a multiplier. After epoch 1 of Pruning's/LRF's own
    # recovery fine-tune (both in the real pipeline and their searches), abort
    # the remaining epochs if the drop vs. the original baseline already
    # exceeds this. None (default) = disabled — every fine-tune runs to
    # completion. See _kd_recovery_fine_tune's docstring in compression.py
    # for the full mechanism.
    early_abort_threshold:   Optional[float] = None,
    # Explicit cache-path overrides for the two hyperparameter searches. None
    # (default) = auto-derive a per-model path under .sigularty_cache/ (see
    # _derive_cache_key below) so compressing two different models from the
    # same working directory never silently shares (and corrupts) each
    # other's cached trial results.
    epsilon_cache_path:      Optional[str]   = None,
    pruning_cache_path:      Optional[str]   = None,
    # ── Technique enable flags ────────────────────────────────────────────────
    use_pruning:             bool            = False,
    use_lrf:                 bool            = True,
    use_clustering:          bool            = True,
    use_quantization:        bool            = True,
    use_kd_finetune:         bool            = True,
    use_gptq:                bool            = False,
    # ── Pruning hyperparameters ───────────────────────────────────────────────
    pruning_ratio:           float           = 0.3,
    pruning_max_ratio:       float           = 0.95,
    # Ceiling specifically for auto-detected residual/skip-connection-coupled
    # groups (the layers whose channel count IS the residual stream for an
    # entire network stage). None (default) = fall back to pruning_max_ratio
    # above, matching main.py's PRUNING_RESIDUAL_MAX_RATIO convention. See
    # _find_residual_coupled_groups() in compression.py and README.md's
    # "Architecture-Agnostic Residual Group Detection" section.
    pruning_residual_max_ratio: Optional[float] = None,
    pruning_model_type:      str             = 'classifier',
    pruning_fine_tune_epochs: int            = 3,
    pruning_fine_tune_lr:    float           = 1e-4,
    pruning_cal_batches:     int             = 50,
    pruning_iterative_steps: int             = 1,
    pruning_isomorphic:      bool            = False,
    pruning_round_to:        Optional[int]   = None,
    # ── LRF hyperparameters ───────────────────────────────────────────────────
    lrf_epsilon:             float           = 0.5,
    lrf_adaptive:            bool            = False,
    lrf_energy_threshold:    float           = 0.99,
    lrf_min_layer_size:      int             = 64,
    lrf_min_rank:            int             = 2,
    lrf_skip_large_kernels:  bool            = False,
    # ── Clustering hyperparameters ────────────────────────────────────────────
    num_clusters:            int             = 16,
    cluster_fine_tune_epochs: int            = 5,
    cluster_fine_tune_lr:    float           = 1e-5,
    # ── KD fine-tuning hyperparameters ────────────────────────────────────────
    kd_epochs:               int             = 3,
    kd_lr:                   float           = 1e-5,
    kd_temperature:          float           = 4.0,
    kd_alpha:                float           = 0.7,
    kd_max_batches:          int             = 50,
    # ── Quantization hyperparameters ─────────────────────────────────────────
    quant_mode:              str             = 'fp16',
    quant_cal_batches:       int             = 100,
    # ── GPTQ hyperparameters ─────────────────────────────────────────────────
    gptq_bits:               int             = 4,
    gptq_cal_batches:        int             = 16,
    gptq_block_size:         int             = 128,
    # ── CQI scoring weights ───────────────────────────────────────────────────
    cqi_w_accuracy:          float           = 1.0,
    cqi_w_size:              float           = 1.0,
    cqi_w_latency:           float           = 1.0,
    cqi_w_kl:                float           = 1.0,
    # ── Report ────────────────────────────────────────────────────────────────
    save_report:             bool            = True,
    report_path:             str             = 'compression_report.png',
) -> CompressionResult:
    """
    Compress a PyTorch model.

    Runs BN Fusion → (optional Pruning) → (optional LRF) → (optional Clustering)
    → (optional GPTQ) → (optional Quantization) → (optional KD Fine-tuning)
    in a fixed order. The original model is never modified.

    Parameters
    ----------
    model       : Any nn.Module.  Deep-copied internally.
    dataloader  : DataLoader yielding (inputs, labels).  Used for calibration
                  and fine-tuning throughout the pipeline. Never evaluated
                  against directly — see test_loader.

    Returns
    -------
    CompressionResult
        Contains the compressed model and all measurement results.

    Notes
    -----
    Evaluation happens on test_loader throughout — the baseline measurement,
    both hyperparameter searches, every technique's accuracy gate, and the
    final report. dataloader is reserved for calibration and fine-tuning. If
    test_loader is not supplied, compress() derives one automatically by
    splitting dataloader's dataset 70/30 (train/held-out) with a fixed seed,
    so the model is never fine-tuned and then graded on the exact same
    samples. See _auto_split_dataloader's docstring for the mechanics.

    Latency is always measured in float32 on the target device so GPU-vs-GPU
    comparisons are fair.  Accuracy and size use the final compressed model,
    which may be INT8/fp16.

    Accuracy-drop gating is always active: Structured Pruning, Low-Rank
    Factorization, and Weight Clustering, and GPTQ, standard Quantization,
    and the final KD recovery fine-tune are each individually measured
    before/after and reverted to their pre-technique state if that ONE
    technique's own marginal accuracy drop exceeds accuracy_drop_threshold. A
    reverted technique will not appear in the returned
    CompressionResult.techniques_applied. This matches
    run_compression_pipeline()'s gating, including:
      - QAT-aware GPTQ (apply_gptq_quantization(..., qat=use_kd_finetune))
        so the final KD step can recover accuracy from GPTQ-quantized
        weights via a straight-through estimator, rather than finding them
        frozen;
      - collapse_qat_layers() afterward, shrinking those layers back to
        compact frozen INT4 storage once KD finishes;
      - one bounded, LR-adjusted retry if KD's cumulative recovery (vs. the
        absolute original baseline) still falls short of
        accuracy_drop_threshold after KD's own gate already passed.
    See apply_compression_pipeline() in compression.py,
    run_compression_pipeline() in helper_functions.py, and "Global
    Accuracy-Drop Threshold & Per-Technique Gating" in README.md for the
    full mechanism.

    The KD fine-tune (use_kd_finetune) runs after GPTQ and standard
    quantization, never before, so it recovers accuracy lost to every prior
    step at once.

    Search caches (epsilon_cache_path / pruning_cache_path): each search's
    trial-by-trial results are cached to disk purely by hyperparameter value
    (see find_optimal_epsilon_smart / find_optimal_pruning_params in
    optimization.py), with no reference to which model or data split
    produced them. compress() derives a per-model cache filename under
    .sigularty_cache/ by default so two different models never silently
    share (and corrupt) each other's cached numbers. Pass an explicit path
    yourself if you need a stronger guarantee (e.g. distinct caches per
    dataset too, not just per model/num_classes).
    """
    # ── Deferred imports ────────────────────────────────
    from sigularty.compression import (
        apply_compression_pipeline,
        apply_quantization,
        apply_gptq_quantization,
        collapse_qat_layers,
        fine_tune_with_distillation,
    )
    from sigularty.helper_functions import (
        get_model_size_mb,
        measure_accuracy,
        measure_latency,
        count_parameters,
        gate_technique_accuracy,
        report_technique_impact,
    )
    from sigularty.optimization import (
        compression_quality_index,
        find_optimal_epsilon_smart,
        find_optimal_pruning_params,
    )
    from sigularty.visualization import generate_compression_report

    # ── Device resolution ─────────────────────────────────────────────────────
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # fp16 quantization requires a GPU — fall back gracefully
    if quant_mode == 'fp16' and device == 'cpu':
        print("  ⚠️  fp16 requested but device=cpu — switching to dynamic INT8.")
        quant_mode = 'dynamic'

    # ── Evaluation data — resolve test_loader before anything touches the
    #    data, including the optional pre-training step below. If pretrain
    #    trained on data that later gets called "held out", every downstream
    #    measurement against that data would be contaminated the same way
    #    the whole rest of this fix exists to prevent. ─────────────────────────
    if test_loader is None:
        dataloader, test_loader = _auto_split_dataloader(dataloader)

    # ── Optional: pre-train on target dataset before compressing ────────────
    # This is needed when a freshly loaded model has a randomly initialised
    # head (ImageNet weights + untrained classifier = ~1% accuracy).
    if pretrain_epochs > 0:
        print(f"\n[compress] Pre-training for {pretrain_epochs} epoch(s) "
              f"(lr={pretrain_lr}) before compressing...")
        from sigularty.helper_functions import train_model as _train_model
        _test_loader = pretrain_test_loader if pretrain_test_loader is not None else test_loader
        model = _train_model(
            model=model,
            train_loader=dataloader,
            test_loader=_test_loader,
            num_classes=num_classes,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            device=device,
            save_path=None,
        )
        model.eval()
        print(f"[compress] Pre-training complete.")

    # ── Baseline measurements ─────────────────────────────────────────────────
    print("\n[compress] Measuring baseline (size, accuracy, latency)...")
    model.eval()
    model.to(device)

    baseline_size     = get_model_size_mb(model)
    baseline_acc      = measure_accuracy(model, test_loader, device)
    # Infers both shape and dtype from the first batch, since latency
    # measurement needs a dummy input of the right kind (int64 token IDs
    # for NLP models, float32 pixels for vision) — shape/dtype don't depend
    # on which subset of samples this comes from, so dataloader (train side)
    # is just as good a source for this as test_loader would be.
    input_shape, input_dtype = _infer_input_shape_and_dtype(model, dataloader, device)
    baseline_latency  = measure_latency(
        model, input_shape, device, num_iterations=50, warmup=5,
        input_dtype=input_dtype,
    )['mean_ms']

    print(f"  Baseline: {baseline_acc:.2f}%  {baseline_size:.2f} MB  "
          f"{baseline_latency:.3f} ms")

    # ── Optional: epsilon search ──────────────────────────────────────────────
    if find_optimal_epsilon and use_lrf:
        _eps_cache_path = epsilon_cache_path or (
            f".sigularty_cache/epsilon_{_derive_cache_key(model, num_classes)}.json"
        )
        print(f"\n[compress] LRF epsilon search ({epsilon_search_trials} trials)...")
        print(f"  Cache: '{_eps_cache_path}'")
        best_eps, _, _ = find_optimal_epsilon_smart(
            model=model,
            test_loader=test_loader,
            device=device,
            baseline_accuracy=baseline_acc,
            baseline_size=baseline_size,
            baseline_latency_ms=baseline_latency,
            accuracy_drop_threshold=accuracy_drop_threshold,
            num_trials=epsilon_search_trials,
            cache_path=_eps_cache_path,
            input_shape=input_shape,
            input_dtype=input_dtype,
            early_abort_threshold=early_abort_threshold,
            w_accuracy=cqi_w_accuracy,
            w_size=cqi_w_size,
            w_latency=cqi_w_latency,
        )
        if best_eps is not None:
            print(f"  → Optimal ε = {best_eps:.4f}")
            lrf_epsilon = best_eps

    # ── Optional: pruning search ──────────────────────────────────────────────
    if find_optimal_pruning and use_pruning:
        _prune_cache_path = pruning_cache_path or (
            f".sigularty_cache/pruning_{_derive_cache_key(model, num_classes)}.json"
        )
        print(f"\n[compress] Pruning search ({pruning_search_trials} trials)...")
        print(f"  Cache: '{_prune_cache_path}'")
        best_cfg, _, _ = find_optimal_pruning_params(
            model=model,
            dataloader=dataloader,
            test_loader=test_loader,
            device=device,
            baseline_accuracy=baseline_acc,
            baseline_size=baseline_size,
            baseline_latency_ms=baseline_latency,
            model_type=pruning_model_type,
            num_classes=num_classes,
            max_pruning_ratio=pruning_max_ratio,
            residual_max_ratio=pruning_residual_max_ratio,
            accuracy_drop_threshold=accuracy_drop_threshold,
            num_trials=pruning_search_trials,
            cache_path=_prune_cache_path,
            input_shape=input_shape,
            input_dtype=input_dtype,
            search_ft_epochs=pruning_search_ft_epochs,
            search_ft_lr=pruning_search_ft_lr,
            early_abort_threshold=early_abort_threshold,
            w_accuracy=cqi_w_accuracy,
            w_size=cqi_w_size,
            w_latency=cqi_w_latency,
            w_kl=cqi_w_kl,
        )
        if best_cfg is not None:
            pruning_ratio           = best_cfg.get('pruning_ratio',     pruning_ratio)
            pruning_max_ratio       = best_cfg.get('pruning_max_ratio', pruning_max_ratio)
            pruning_iterative_steps = best_cfg.get('iterative_steps',   pruning_iterative_steps)
            print(f"  → ratio={pruning_ratio:.3f}  "
                  f"max_ratio={pruning_max_ratio:.2f}  "
                  f"steps={pruning_iterative_steps}")

    # ── Structural compression (float32) ───────────────────────────────────────
    # BN Fusion, Pruning, LRF, Clustering. Quantization is deferred:
    # quantized tensors (qint8) have no CUDA kernel, which would make GPU
    # latency comparisons unfair. KD fine-tuning is also deferred so it runs
    # exactly once, after quantization/GPTQ, and can recover accuracy lost
    # to all of them at once rather than just to the structural changes.
    print("\n[compress] Structural compression...")
    compressed_pre_quant = apply_compression_pipeline(
        model=model,
        dataloader=dataloader,
        device=device,
        use_bn_fusion=True,
        use_pruning=use_pruning,
        use_low_rank=use_lrf,
        use_clustering=use_clustering,
        use_kd_finetune=False,
        use_quantization=False,
        use_gptq=False,
        pruning_ratio=pruning_ratio,
        pruning_model_type=pruning_model_type,
        pruning_num_classes=num_classes,
        pruning_fine_tune_epochs=pruning_fine_tune_epochs,
        pruning_fine_tune_lr=pruning_fine_tune_lr,
        pruning_cal_batches=pruning_cal_batches,
        pruning_iterative_steps=pruning_iterative_steps,
        pruning_round_to=pruning_round_to,
        pruning_isomorphic=pruning_isomorphic,
        pruning_max_ratio=pruning_max_ratio,
        pruning_residual_max_ratio=pruning_residual_max_ratio,
        lrf_epsilon=lrf_epsilon,
        lrf_min_layer_size=lrf_min_layer_size,
        lrf_min_rank=lrf_min_rank,
        lrf_skip_large_kernels=lrf_skip_large_kernels,
        lrf_adaptive=lrf_adaptive,
        lrf_energy_threshold=lrf_energy_threshold,
        cluster_num_clusters=num_clusters,
        cluster_fine_tune_epochs=cluster_fine_tune_epochs,
        cluster_fine_tune_lr=cluster_fine_tune_lr,
        cluster_num_classes=num_classes,
        cluster_layers=None,
        kd_teacher=None,
        kd_epochs=kd_epochs,
        kd_lr=kd_lr,
        kd_temperature=kd_temperature,
        kd_alpha=kd_alpha,
        kd_num_classes=num_classes,
        # Per-technique accuracy gating: with test_loader supplied, every
        # mutating technique here (Structured Pruning, LRF, Weight
        # Clustering) is measured before/after and reverted if its own
        # marginal drop exceeds accuracy_drop_threshold. See README.md's
        # "Global Accuracy-Drop Threshold & Per-Technique Gating" section.
        test_loader=test_loader,
        accuracy_drop_threshold=accuracy_drop_threshold,
        early_abort_threshold=early_abort_threshold,
        original_size_mb=baseline_size,
        original_latency_ms=baseline_latency,
        input_shape=input_shape,
        input_dtype=input_dtype,
    )

    # Latency measured on float32 model — GPU vs GPU, apples-to-apples
    comp_latency = measure_latency(
        compressed_pre_quant, input_shape, device, num_iterations=50, warmup=5,
        input_dtype=input_dtype,
    )['mean_ms']

    # current_accuracy is threaded through the entire rest of this function —
    # every gate below reads/updates it, mirroring run_compression_pipeline()'s
    # own current_accuracy variable in helper_functions.py. The structural
    # compression step above already measured this as ._gating_accuracy (since
    # test_loader was passed in); reusing it avoids a redundant
    # measure_accuracy() call for an identical number.
    current_accuracy = getattr(compressed_pre_quant, '_gating_accuracy', baseline_acc)

    # ── GPTQ → Quantization → final KD recovery (all gated) ───────────────────
    # GPTQ and standard Quantization are each individually measured before/
    # after and reverted on their own marginal accuracy drop; the final KD
    # recovery fine-tune runs last, after both, so it can recover accuracy
    # lost to quantization/GPTQ too — not just to the earlier structural
    # changes.
    compressed_model = compressed_pre_quant

    # Tracks whether the FINAL compressed_model ended up as a dynamic-INT8
    # (CPU-resident) model. A real tracked flag rather than re-deriving from
    # quant_mode, because gating can revert dynamic quant back to a
    # GPU-resident fp32 model — every later device-dependent decision reads
    # this flag instead of re-checking quant_mode.
    _final_quant_is_dynamic = False
    _use_gptq   = use_gptq     # local, mutable copy — becomes False if reverted
    _quant_kept = False
    _kd_kept    = False

    # ── GPTQ (gated) ───────────────────────────────────────────────────────────
    if _use_gptq:
        print(f"\n[compress] GPTQ INT{gptq_bits}...")
        _pre_gptq_model    = compressed_model
        _pre_gptq_accuracy = current_accuracy
        _gptq_result = apply_gptq_quantization(
            copy.deepcopy(compressed_model),
            dataloader=dataloader,
            device=device,
            bits=gptq_bits,
            num_calibration_batches=gptq_cal_batches,
            block_size=gptq_block_size,
            # Only produce trainable (QAT/STE) GPTQ layers when a KD
            # fine-tune is actually going to run afterward — that's the
            # only thing that would ever use shadow_weight's trainability.
            # Otherwise this is the plain frozen _Int4Linear, zero added
            # overhead. See _Int4LinearQAT's docstring in compression.py.
            qat=use_kd_finetune,
        )
        compressed_model, current_accuracy, _gptq_kept = gate_technique_accuracy(
            f"GPTQ INT{gptq_bits}", _pre_gptq_model, _gptq_result, test_loader,
            device, accuracy_drop_threshold, pre_accuracy=_pre_gptq_accuracy,
        )
        _gptq_report_data = getattr(_gptq_result, '_gptq_report', {}) or {}
        _gptq_n_quantized  = _gptq_report_data.get('layers_quantized', 0)
        _gptq_n_eligible   = _gptq_report_data.get('total_eligible_layers', 0)
        _gptq_post_acc = (
            current_accuracy if _gptq_kept
            else measure_accuracy(_gptq_result, test_loader, device)
        )
        report_technique_impact(
            f"GPTQ INT{gptq_bits}",
            pre_model=_pre_gptq_model, post_model=_gptq_result,
            pre_accuracy=_pre_gptq_accuracy, post_accuracy=_gptq_post_acc,
            was_kept=_gptq_kept,
            pre_device=device, post_device=device,
            original_accuracy=baseline_acc, original_size_mb=baseline_size,
            original_latency_ms=baseline_latency,
            input_shape=input_shape, input_dtype=input_dtype,
            structural_summary=(
                f"{_gptq_n_quantized}/{_gptq_n_eligible} eligible Linear "
                f"layer(s) quantized to INT{gptq_bits}"
            ),
            structural_zero=(_gptq_n_quantized == 0),
        )
        if not _gptq_kept:
            # GPTQ was reverted — treat it as if it had never been requested
            # for every downstream decision (dynamic-skip check, KD's qat
            # wiring already happened at construction time and is harmless
            # either way since collapse_qat_layers is unconditional below).
            _use_gptq = False

    # ── Standard Quantization (gated) ─────────────────────────────────────────
    # Dynamic INT8 is skipped when GPTQ already ran on Linear layers —
    # running both would discard GPTQ's Hessian correction (dynamic INT8
    # would just re-round GPTQ's own INT4-corrected weights). fp16 is NOT
    # skipped when GPTQ is absent or reverted here — fp16-only remains a
    # fully supported, independent configuration for compress().
    _skip_dynamic = _use_gptq and quant_mode == 'dynamic'

    if use_quantization and not _skip_dynamic:
        print(f"\n[compress] Quantization ({quant_mode})...")
        _pre_quant_model    = compressed_model
        _pre_quant_accuracy = current_accuracy
        quant_input = copy.deepcopy(compressed_model)
        if quant_mode == 'dynamic':
            quant_input = quant_input.cpu()
        _quant_result = apply_quantization(
            quant_input,
            dataloader=dataloader,
            mode=quant_mode,
            num_calibration_batches=quant_cal_batches,
        )
        _quant_eval_device = 'cpu' if quant_mode == 'dynamic' else device
        compressed_model, current_accuracy, _quant_kept = gate_technique_accuracy(
            f"Quantization ({quant_mode})", _pre_quant_model, _quant_result,
            test_loader, _quant_eval_device, accuracy_drop_threshold,
            pre_accuracy=_pre_quant_accuracy,
        )
        _quant_post_acc = (
            current_accuracy if _quant_kept
            else measure_accuracy(_quant_result, test_loader, _quant_eval_device)
        )
        report_technique_impact(
            f"Quantization ({quant_mode})",
            pre_model=_pre_quant_model, post_model=_quant_result,
            pre_accuracy=_pre_quant_accuracy, post_accuracy=_quant_post_acc,
            was_kept=_quant_kept,
            pre_device=device, post_device=_quant_eval_device,
            original_accuracy=baseline_acc, original_size_mb=baseline_size,
            original_latency_ms=baseline_latency,
            input_shape=input_shape, input_dtype=input_dtype,
            structural_summary=f"all parameters cast to {quant_mode}",
            structural_zero=False,
        )
        if _quant_kept:
            _final_quant_is_dynamic = (quant_mode == 'dynamic')
        # else: reverted — compressed_model is back to the pre-quant model,
        # which never moved off `device`, so _final_quant_is_dynamic
        # correctly stays False.
    elif use_quantization and _skip_dynamic:
        print(f"\n[compress] Skipping dynamic INT8 — GPTQ already ran on "
              f"Linear layers. Use quant_mode='fp16' to also compress conv "
              f"weights (or combine GPTQ+fp16 in one run).")

    # ── Final KD recovery fine-tune (gated + cumulative-recovery retry) ──────
    # Runs last — after GPTQ and standard Quantization — so it recovers
    # accuracy lost to every prior step at once.
    if use_kd_finetune:
        print(f"\n[compress] KD Fine-tune ({kd_epochs} epoch(s), "
              f"post-quantization)...")
        _kd_device = 'cpu' if _final_quant_is_dynamic else device
        _pre_kd_model    = compressed_model
        _pre_kd_accuracy = current_accuracy
        try:
            _kd_result = fine_tune_with_distillation(
                student=compressed_model,
                teacher=model,
                dataloader=dataloader,
                num_classes=num_classes,
                epochs=kd_epochs,
                lr=kd_lr,
                device=_kd_device,
                temperature=kd_temperature,
                alpha=kd_alpha,
                max_batches=kd_max_batches,
                test_loader=test_loader,
            )
        except Exception as exc:
            print(f"  ⚠️  KD fine-tune failed: {exc}")
            _kd_result = None

        if _kd_result is not None:
            compressed_model, current_accuracy, _kd_kept = gate_technique_accuracy(
                "KD Fine-tune", _pre_kd_model, _kd_result, test_loader, _kd_device,
                accuracy_drop_threshold, pre_accuracy=_pre_kd_accuracy,
            )
            _pre_kd_size_mb = get_model_size_mb(_pre_kd_model)
            _kd_post_acc = (
                current_accuracy if _kd_kept
                else measure_accuracy(_kd_result, test_loader, _kd_device)
            )
            report_technique_impact(
                "KD Fine-tune (final)",
                pre_model=_pre_kd_model, post_model=_kd_result,
                pre_accuracy=_pre_kd_accuracy, post_accuracy=_kd_post_acc,
                was_kept=_kd_kept,
                pre_device=_kd_device, post_device=_kd_device,
                original_accuracy=baseline_acc, original_size_mb=baseline_size,
                original_latency_ms=baseline_latency,
                input_shape=input_shape, input_dtype=input_dtype,
                pre_size_mb=_pre_kd_size_mb,
                structural_summary=(
                    "n/a — recovers accuracy lost to every prior technique, "
                    "no structural change of its own"
                ),
                structural_zero=False,
            )

            if _kd_kept:
                # ── Cumulative recovery-quality check + one bounded retry ────
                # If the CUMULATIVE drop vs. the absolute original baseline
                # still exceeds threshold even after KD's own marginal gate
                # passed, inspect the per-epoch test-accuracy trend for
                # exactly one bounded, LR-adjusted retry.
                cumulative_drop = baseline_acc - current_accuracy
                if cumulative_drop > accuracy_drop_threshold:
                    _history = getattr(_kd_result, '_kd_history', [])
                    _test_accs = [h['test_acc'] for h in _history if h.get('test_acc') is not None]
                    _retry_lr: Optional[float] = None
                    _retry_reason: Optional[str] = None

                    if len(_test_accs) >= 2:
                        _deltas = [_test_accs[i] - _test_accs[i - 1] for i in range(1, len(_test_accs))]
                        _avg_delta = sum(_deltas) / len(_deltas)
                        _all_positive = all(d > 0 for d in _deltas)
                        if abs(_avg_delta) < 0.5:
                            _retry_lr = kd_lr * 0.5
                            _retry_reason = (
                                f"stagnant (avg epoch-to-epoch change "
                                f"{_avg_delta:+.2f}pp, |Δ|<0.5pp)"
                            )
                        elif _all_positive and _avg_delta < 2.0:
                            _retry_lr = kd_lr * 1.5
                            _retry_reason = (
                                f"small constant increase (avg {_avg_delta:+.2f}pp/"
                                f"epoch, <2pp)"
                            )

                    if _retry_lr is not None:
                        print(
                            f"\n  [KD Recovery] Cumulative accuracy drop "
                            f"{cumulative_drop:.2f}pp still exceeds the "
                            f"{accuracy_drop_threshold:.1f}pp threshold after KD. "
                            f"Trend is {_retry_reason} — retrying ONCE with lr "
                            f"{kd_lr:.2e} → {_retry_lr:.2e}, restarting fresh "
                            f"from the pre-KD checkpoint. This result is "
                            f"accepted unconditionally regardless of outcome — "
                            f"no further retries.\n"
                        )
                        try:
                            _retry_result = fine_tune_with_distillation(
                                student=_pre_kd_model,
                                teacher=model,
                                dataloader=dataloader,
                                num_classes=num_classes,
                                epochs=kd_epochs,
                                lr=_retry_lr,
                                device=_kd_device,
                                temperature=kd_temperature,
                                alpha=kd_alpha,
                                max_batches=kd_max_batches,
                                test_loader=test_loader,
                            )
                            _pre_retry_accuracy = current_accuracy
                            del compressed_model
                            if _kd_device == 'cuda':
                                torch.cuda.empty_cache()
                            compressed_model = _retry_result
                            current_accuracy = measure_accuracy(
                                compressed_model, test_loader, _kd_device,
                            )
                            print(
                                f"  [KD Recovery] Retry result: "
                                f"{current_accuracy:.2f}% (was "
                                f"{_pre_retry_accuracy:.2f}% before retry). "
                                f"Accepting this result regardless of outcome — "
                                f"if it is still not acceptable, try again with "
                                f"different compression settings."
                            )
                        except Exception as exc:
                            print(f"  ⚠️  KD recovery retry failed ({exc}) — "
                                  f"keeping the pre-retry result.")
                    else:
                        print(
                            f"\n  [KD Recovery] Cumulative accuracy drop "
                            f"{cumulative_drop:.2f}pp still exceeds the "
                            f"{accuracy_drop_threshold:.1f}pp threshold, but the "
                            f"per-epoch trend doesn't match the stagnant/"
                            f"small-increase retry conditions — accepting the "
                            f"result as-is. Consider less aggressive "
                            f"compression settings.\n"
                        )

    # ── Collapse any QAT-mode GPTQ layers back to frozen, compact form ───────
    # Safe to call unconditionally: if QAT was never requested (no fine-tune
    # was going to run, or GPTQ wasn't used at all), this just walks the
    # model, finds zero _Int4LinearQAT instances, and returns unchanged.
    compressed_model = collapse_qat_layers(compressed_model)
    if _use_gptq and use_kd_finetune:
        # Collapse reconstructs the exact same weight the QAT layer's own
        # forward() was already computing, so this should not change
        # accuracy — re-measuring anyway rather than trusting that by
        # assumption, since this is the one point where the model's actual
        # computation graph just changed shape. _kd_device is guaranteed set
        # by this point: it's assigned unconditionally near the top of the
        # `if use_kd_finetune:` block above, exactly the condition gating
        # entry into this branch too.
        current_accuracy = measure_accuracy(compressed_model, test_loader, _kd_device)

    # ── Device for evaluating the final compressed model ──────────────────────
    # fp16 stays on the main compute device; dynamic INT8 forces CPU (fbgemm).
    # Uses _final_quant_is_dynamic (a tracked flag, see above) rather than
    # re-deriving from quant_mode, because quantization gating can have
    # reverted dynamic quant back to a GPU-resident fp32 model.
    acc_device = 'cpu' if _final_quant_is_dynamic else device

    # Re-apply fp16 if quantization mode is fp16 — fine_tune_with_distillation
    # internally casts the student to float32 for stable training, so without
    # this the reported size would equal the float32 baseline even though
    # quant_mode='fp16' was requested. Only relevant when fp16 quantization
    # actually survived its gate.
    if use_quantization and quant_mode == 'fp16' and _quant_kept:
        compressed_model = compressed_model.to(torch.float16)
        current_accuracy = measure_accuracy(compressed_model, test_loader, acc_device)
        print(f"\n  [fp16 recast] Re-measured accuracy after final fp16 cast: "
              f"{current_accuracy:.2f}%")

    # ── Evaluate final compressed model ───────────────────────────────────────
    # current_accuracy already reflects the correctly-measured, final state
    # from whichever gate/re-measurement above touched it last — reusing it
    # here avoids a redundant measure_accuracy() call for an identical number.
    print(f"\n[compress] Evaluating compressed model (on {acc_device})...")
    comp_acc  = current_accuracy
    comp_size = get_model_size_mb(compressed_model)

    # ── What actually survived gating ─────────────────────────────────────────
    # A technique that got reverted by its own accuracy gate must not be
    # reported as "applied" — matches run_compression_pipeline's identical
    # convention in helper_functions.py.
    _gating_report = getattr(compressed_pre_quant, '_gating_report', {})
    pruning_rep = getattr(compressed_pre_quant, '_pruning_report', None)
    _pruning_incompatible = bool((pruning_rep or {}).get('torch_pruning_incompatible', False))

    # ── Build techniques_applied ──────────────────────────────────────────────
    # Order matches actual execution order: BN Fusion → Pruning → LRF →
    # Clustering → GPTQ → Quantization → final KD.
    techniques: List[str] = ['BN Fusion']
    if use_pruning and not _pruning_incompatible and _gating_report.get('Structured Pruning', True):
        techniques.append(f'Structured Pruning (ratio={pruning_ratio:.2f})')
    if use_lrf and _gating_report.get('Low-Rank Factorization', True):
        tag = f'ε={lrf_epsilon:.2f}' if not lrf_adaptive else 'adaptive'
        techniques.append(f'Low-Rank Factorization ({tag})')
    if use_clustering and _gating_report.get('Weight Clustering', True):
        techniques.append(f'Weight Clustering (k={num_clusters})')
    if _use_gptq:
        techniques.append(f'GPTQ INT{gptq_bits}')
    if use_quantization and not _skip_dynamic and _quant_kept:
        techniques.append(f'Quantization ({quant_mode})')
    if use_kd_finetune and _kd_kept:
        techniques.append(f'KD Fine-tune (T={kd_temperature:.1f}, α={kd_alpha:.2f})')

    # ── CQI ───────────────────────────────────────────────────────────────────
    cqi = compression_quality_index(
        accuracy=comp_acc,
        size_mb=comp_size,
        baseline_accuracy=baseline_acc,
        baseline_size=baseline_size,
        latency_ms=comp_latency,
        baseline_latency_ms=baseline_latency,
        w_accuracy=cqi_w_accuracy,
        w_size=cqi_w_size,
        w_latency=cqi_w_latency,
    )

    # ── Compression report PNG ────────────────────────────────────────────────
    out_report_path: Optional[str] = None
    if save_report:
        try:
            metrics_dict = {
                'original_accuracy':     baseline_acc,
                'compressed_accuracy':   comp_acc,
                'original_size_mb':      baseline_size,
                'compressed_size_mb':    comp_size,
                'original_params':       count_parameters(model),
                'compressed_params':     count_parameters(compressed_model),
                'original_latency_ms':   baseline_latency,
                'compressed_latency_ms': comp_latency,
                'techniques_used':       techniques,
                'cqi':                   cqi,
                'pruning_report':        pruning_rep,
            }
            generate_compression_report(
                original_model=model,
                compressed_model=compressed_model,
                metrics_dict=metrics_dict,
                save_path=report_path,
                dataloader=test_loader,
                device=acc_device,
            )
            out_report_path = os.path.abspath(report_path)
        except Exception as exc:
            print(f"  ⚠️  Report generation failed: {exc}")

    # ── Derived scalars ───────────────────────────────────────────────────────
    lat_speedup   = (baseline_latency / comp_latency) if comp_latency > 0 else 1.0
    acc_retention = (comp_acc / baseline_acc * 100)   if baseline_acc  > 0 else 0.0
    comp_ratio    = (baseline_size / comp_size)        if comp_size     > 0 else 1.0

    print(f"\n[compress] Done.")
    print(f"  Ratio       : {comp_ratio:.2f}×")
    print(f"  Accuracy    : {baseline_acc:.2f}% → {comp_acc:.2f}%  "
          f"(retention {acc_retention:.1f}%)")
    print(f"  Size        : {baseline_size:.2f} MB → {comp_size:.2f} MB")
    print(f"  Latency     : {baseline_latency:.3f} ms → {comp_latency:.3f} ms  "
          f"({lat_speedup:.2f}× speedup)")
    print(f"  CQI         : {cqi:.3f}")

    return CompressionResult(
        model=compressed_model,
        compression_ratio=comp_ratio,
        accuracy_retention=acc_retention,
        size_original_mb=baseline_size,
        size_compressed_mb=comp_size,
        original_accuracy=baseline_acc,
        compressed_accuracy=comp_acc,
        original_latency_ms=baseline_latency,
        compressed_latency_ms=comp_latency,
        latency_speedup=lat_speedup,
        cqi=cqi,
        techniques_applied=techniques,
        report_path=out_report_path,
        pruning_report=pruning_rep,
    )



# ============================================================================
# ── FINETUNE ─────────────────────────────────────────────────────────────────
# ============================================================================


# ============================================================================
# ── LEARNING RATE FINDER ─────────────────────────────────────────────────────
# ============================================================================

def find_best_lr(
    model,
    dataloader,
    *,
    device     = None,
    num_classes: int   = 10,
    start_lr: float    = 1e-7,
    end_lr: float      = 10.0,
    num_steps: int     = 100,
    smooth_window: int = 5,
    diverge_threshold: float = 4.0,
) -> float:
    """
    Find the optimal learning rate before fine-tuning or pre-training.

    Runs the LR range test (Smith 2017): exponentially increases lr from
    start_lr to end_lr over num_steps batches and returns the lr at the
    point of steepest loss descent.

    Use this BEFORE calling finetune() or compress() with pretrain_epochs > 0
    so you don't waste a run on the wrong learning rate.

    Rule of thumb:
        SGD  → use returned value directly
        Adam → use returned value ÷ 10

    Parameters
    ----------
    model       : Any nn.Module.
    dataloader  : DataLoader yielding (inputs, labels).
    device      : 'cuda' or 'cpu'. Auto-detected when None.
    num_classes : Output class count (unused in the range test itself).
    start_lr    : Lowest lr to test (default 1e-7).
    end_lr      : Highest lr to test (default 10.0).
    num_steps   : How many batches to sweep over.
    smooth_window: Loss smoothing window to reduce noise.
    diverge_threshold: Stop early if loss explodes beyond this multiple of best.

    Returns
    -------
    float — suggested learning rate.

    Saves
    -----
    'lr_finder.png' — loss vs lr plot with the suggested lr marked.

    Example
    -------
    >>> from sigularty import find_best_lr, finetune, compress
    >>>
    >>> best_lr = find_best_lr(model, train_loader, device='cuda')
    >>> # Suggested LR: 1.23e-03
    >>> # For SGD    : 1.23e-03
    >>> # For Adam   : 1.23e-04  (÷10 rule)
    >>>
    >>> model = finetune(model, train_loader, lr=best_lr / 10, epochs=10)
    >>> result = compress(model, train_loader)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from sigularty.helper_functions import find_best_lr as _fbr
    return _fbr(
        model, dataloader,
        device=device,
        num_classes=num_classes,
        start_lr=start_lr,
        end_lr=end_lr,
        num_steps=num_steps,
        smooth_window=smooth_window,
        diverge_threshold=diverge_threshold,
    )

def finetune(
    model:       nn.Module,
    train_loader: DataLoader,
    *,
    test_loader:  Optional[DataLoader] = None,
    num_classes:  int                  = 10,
    epochs:       int                  = 10,
    lr:           float                = 1e-3,
    max_batches:  int                  = 0,    # max batches per epoch (0 = full loader)
    device:       Optional[str]        = None,
    save_path:    Optional[str]        = None,
) -> nn.Module:
    """
    Fine-tune a model on a target dataset before compressing it.

    Call this when load_from_registry() returns a model with ~1% accuracy
    (randomly initialised classification head on top of pretrained features).
    After fine-tuning, pass the returned model to compress().

    Parameters
    ----------
    model        : Any nn.Module. Modified in-place AND returned.
    train_loader : DataLoader for training.
    test_loader  : DataLoader for validation. If None, uses train_loader —
                   pass a genuine held-out loader if you want this
                   function's own per-epoch test_acc to mean anything.
    num_classes  : Output class count (for accuracy metric).
    epochs       : Number of training epochs. 5-15 recommended for transfer learning.
    lr           : Learning rate. 1e-3 is a safe default for Adam with a new head.
    max_batches  : Max batches per epoch. 0 = full loader. Set 50-100 for fast iteration.
    device       : 'cuda' or 'cpu'. Auto-detected when None.
    save_path    : Optional path to save best checkpoint (.pth). None = don't save.

    Returns
    -------
    nn.Module — the fine-tuned model (same object, returned for chaining).

    Example
    -------
    >>> from sigularty import load_from_registry, finetune, compress
    >>>
    >>> r = load_from_registry('resnet50')
    >>> model = finetune(r.model, r.train_loader, test_loader=r.test_loader,
    ...                  num_classes=r.num_classes, epochs=10)
    >>> result = compress(model, r.train_loader, test_loader=r.test_loader,
    ...                   num_classes=r.num_classes)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _test = test_loader if test_loader is not None else train_loader

    from sigularty.helper_functions import train_model as _train_model

    # If max_batches is set, wrap the loader to stop early each epoch
    _loader = train_loader
    if max_batches > 0:
        from itertools import islice
        class _CappedLoader:
            def __init__(self, loader, n):
                self.loader = loader
                self.n = n
                self.dataset = getattr(loader, 'dataset', None)
            def __iter__(self):
                return islice(iter(self.loader), self.n)
            def __len__(self):
                return min(self.n, len(self.loader))
        _loader = _CappedLoader(train_loader, max_batches)

    return _train_model(
        model=model,
        train_loader=_loader,
        test_loader=_test,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        device=device,
        save_path=save_path,
    )


def analyze(
    model:      nn.Module,
    dataloader: Optional[DataLoader] = None,
    *,
    device: Optional[str] = None,
) -> AnalysisResult:
    """
    Analyse a model and recommend compression techniques.

    Does not modify the model.  The full per-layer SVD analysis is run
    to derive analytically optimal LRF epsilon values per layer.  These are
    the same values the adaptive LRF mode uses during compression.

    Parameters
    ----------
    model       : Any nn.Module.
    dataloader  : Optional. If provided, top-1 accuracy is measured against
                  it — pass a held-out loader if you want that number to
                  reflect generalisation rather than whatever the model has
                  already been trained on.
    device      : 'cuda' or 'cpu'. Auto-detected when None.

    Returns
    -------
    AnalysisResult
        Architecture type, recommended techniques, and per-layer signals.
    """
    from sigularty.helper_functions import (
        get_model_size_mb,
        count_parameters,
        measure_accuracy,
    )

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.eval()
    size_mb    = get_model_size_mb(model)
    num_params = count_parameters(model)

    # ── Per-layer signals ─────────────────────────────────────────────────────
    per_layer: List[LayerSignal] = []
    has_conv       = False
    has_linear     = False
    has_attention  = False

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and not list(mod.children()):
            has_conv = True
            if mod.groups == 1 and mod.in_channels > 64 and mod.out_channels > 64:
                eps      = _compute_layer_lrf_epsilon(mod.weight.data)
                size_kb  = mod.weight.nelement() * mod.weight.element_size() / 1024
                per_layer.append(LayerSignal(
                    name=name, layer_type='Conv2d',
                    lrf_epsilon=eps, prunable=True, size_kb=size_kb,
                ))
        elif isinstance(mod, nn.Linear) and not list(mod.children()):
            has_linear = True
            if mod.in_features > 64 and mod.out_features > 64:
                eps     = _compute_layer_lrf_epsilon(mod.weight.data)
                size_kb = mod.weight.nelement() * mod.weight.element_size() / 1024
                per_layer.append(LayerSignal(
                    name=name, layer_type='Linear',
                    lrf_epsilon=eps, prunable=False, size_kb=size_kb,
                ))
        if hasattr(mod, 'num_heads') or 'attention' in name.lower():
            has_attention = True

    # Sort by size descending so the most impactful layers come first
    per_layer.sort(key=lambda s: s.size_kb, reverse=True)

    # ── Architecture type ─────────────────────────────────────────────────────
    if has_attention and has_conv:
        arch_type = 'hybrid'
    elif has_attention:
        arch_type = 'transformer'
    elif has_conv:
        arch_type = 'cnn'
    else:
        arch_type = 'unknown'

    # ── Technique recommendations ─────────────────────────────────────────────
    # Logic: what gives compression benefit vs what is architecture-compatible.
    recs: List[str] = ['use_clustering', 'use_quantization']   # always safe
    if has_conv or has_linear:
        recs.append('use_lrf')
    if has_conv and not has_attention:
        recs.append('use_pruning')
    if len([t for t in recs if t != 'use_clustering' and t != 'use_quantization']) > 0:
        recs.append('use_kd_finetune')  # recovery after structural compression

    # ── Optional accuracy measurement ─────────────────────────────────────────
    accuracy: Optional[float] = None
    if dataloader is not None:
        try:
            accuracy = measure_accuracy(model, dataloader, device)
        except Exception as exc:
            print(f"  ⚠️  Accuracy measurement failed: {exc}")

    return AnalysisResult(
        size_mb=size_mb,
        num_parameters=num_params,
        architecture_type=arch_type,
        recommended_techniques=recs,
        per_layer_signals=per_layer,
        accuracy=accuracy,
    )


# ============================================================================
# ── PRIVATE HELPERS ──────────────────────────────────────────────────────────
# ============================================================================

def _compute_layer_lrf_epsilon(
    weight: torch.Tensor,
    energy_threshold: float = 0.95,
) -> float:
    """
    Analytically derive the optimal LRF epsilon for a single weight tensor.

    Computes the minimum rank needed to retain `energy_threshold` fraction
    of the total singular value energy (squared), then divides by
    min(in_dim, out_dim) to get the epsilon ratio.

    This is the same logic used by adaptive LRF inside compression.py.
    """
    try:
        w = weight.detach().float().cpu()
        if w.ndim == 4:
            w = w.view(w.size(0), -1)   # [out, in*kh*kw]
        if min(w.shape) < 2:
            return 1.0
        _, S, _ = torch.linalg.svd(w, full_matrices=False)
        energy_cs = (S ** 2).cumsum(0) / (S ** 2).sum()
        # Number of singular values needed to hit threshold
        rank = int((energy_cs < energy_threshold).sum().item()) + 1
        rank = max(1, min(rank, min(w.shape)))
        return round(float(rank / min(w.shape)), 4)
    except Exception:
        return 0.5


def _infer_input_shape_and_dtype(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple:
    """
    Return (input_shape, input_dtype) from the first dataloader batch.

    input_shape:  (1, ...) — a single-example shape.
    input_dtype:  the batch's own dtype — torch.long for NLP token-ID
                  inputs, torch.float32 for vision pixels, etc. Needed so
                  every per-technique impact report's latency measurement
                  (measure_latency's input_dtype param) builds a dummy
                  input of the right kind.

    Falls back to ((1, 3, 224, 224), None) if the batch cannot be read —
    matching measure_latency's own float32 default for the dtype.
    """
    try:
        x, _ = next(iter(dataloader))
        return tuple(x[:1].shape), x.dtype
    except Exception:
        return (1, 3, 224, 224), None


def _auto_split_dataloader(
    dataloader: DataLoader,
    train_frac: float = 0.7,
    seed: int = 42,
    small_test_warning_threshold: int = 100,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split dataloader's underlying dataset into a train subset and a held-out
    test subset, wrapping each in a fresh DataLoader.

    Used by compress() when the caller doesn't supply test_loader. Without
    this, every accuracy measurement in compress() — baseline, both
    hyperparameter searches, every technique's gate, the final report —
    would read accuracy off the exact same data the model is being
    fine-tuned on, which stops being a meaningful signal the moment the
    model can re-memorize that data (trivial for most models on the small
    calibration sets typically used during compression).

    The split uses a fixed seed so the same dataloader always produces the
    same train/test split across repeated compress() calls on the same
    data — otherwise two runs of an identical configuration could disagree
    purely because they happened to draw a different 30%.

    This is a plain random split, not stratified by label — on a dataset
    with many classes and few samples per class, the resulting test split
    can by chance under-represent or entirely miss some classes, making its
    accuracy noisier than the sample count alone suggests. The split still
    proceeds regardless; small_test_warning_threshold only controls when an
    extra warning about that noise gets printed.

    The new loaders inherit batch_size, num_workers, and pin_memory from
    the original dataloader. The train split is shuffled; the test split
    isn't.

    Args:
        dataloader: Original DataLoader. Its .dataset is what gets split;
                    the DataLoader object itself is not reused directly.
        train_frac: Fraction of samples assigned to the train split.
        seed:       Fixed seed for the split, for reproducibility.
        small_test_warning_threshold: If the resulting test split has fewer
                    samples than this, an additional warning is printed
                    calling out that the resulting accuracy read will be
                    noisy. The split still proceeds either way.

    Returns:
        (train_loader, test_loader)

    Raises:
        ValueError: If the dataset doesn't support len() (e.g. an
                    IterableDataset), or has fewer than 2 samples.
    """
    dataset = dataloader.dataset

    if not hasattr(dataset, '__len__'):
        raise ValueError(
            "compress() couldn't auto-split dataloader's dataset because it "
            "doesn't implement __len__ (e.g. an IterableDataset). Pass a "
            "genuine held-out test_loader explicitly instead."
        )

    n = len(dataset)
    if n < 2:
        raise ValueError(
            f"compress() couldn't auto-split dataloader's dataset — it only "
            f"has {n} sample(s), not enough to form a train/test split. Pass "
            f"a genuine held-out test_loader explicitly instead."
        )

    train_size = max(1, min(n - 1, round(train_frac * n)))
    test_size  = n - train_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, test_subset = random_split(
        dataset, [train_size, test_size], generator=generator,
    )

    batch_size  = dataloader.batch_size or 32
    num_workers = dataloader.num_workers

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    print(
        f"\n  ⚠️  [compress] No test_loader supplied — auto-splitting "
        f"dataloader's dataset {train_frac:.0%}/{1 - train_frac:.0%}: "
        f"{train_size} train / {test_size} test (seed={seed}). Every "
        f"accuracy check in this run — baseline, searches, every "
        f"technique's gate, the final report — reads accuracy off this "
        f"held-out {test_size}, never off the data the model fine-tunes "
        f"on. Pass test_loader explicitly to use your own held-out set "
        f"instead."
    )
    if test_size < small_test_warning_threshold:
        print(
            f"  ⚠️  [compress] The held-out split is only {test_size} "
            f"sample(s) — below the {small_test_warning_threshold}-sample "
            f"threshold this warns at. Expect a noisy accuracy read; "
            f"individual technique gate decisions may be unreliable with "
            f"a set this small. Pass a larger test_loader if you can."
        )

    return train_loader, test_loader


def _derive_cache_key(model: nn.Module, num_classes: int) -> str:
    """
    Build a filename-safe, per-model cache-key string for the epsilon and
    pruning search caches.

    find_optimal_epsilon_smart / find_optimal_pruning_params (optimization.py)
    cache every trial result to disk purely by hyperparameter value — e.g.
    str(round(epsilon, 4)), or a "{ratio}|{max_ratio}|{epochs}|{lr}|{steps}"
    compound string. Neither key has any reference to which model produced
    that result, so two different models compressed from the same working
    directory need separate cache files, or one would silently read back
    the other's numbers under a matching hyperparameter value.

    This derives a string from things that are cheap to compute and differ
    across genuinely different models (class name + total parameter count +
    output class count), so the same model reliably reuses its own cache
    across repeated compress() calls (the crash-recovery behaviour documented
    in optimization.py's module docstring), while a different model gets its
    own, separate file under .sigularty_cache/.

    Not a cryptographic or collision-proof hash — just enough to keep
    different models' caches from colliding. Pass epsilon_cache_path /
    pruning_cache_path explicitly to compress() if you need a stronger
    guarantee (e.g. distinct caches per dataset too, not just per model).
    """
    from sigularty.helper_functions import count_parameters
    n_params = count_parameters(model)
    return f"{type(model).__name__}_{n_params}_{num_classes}"


# ============================================================================
# ── VISUALIZATION API ─────────────────────────────────────────────────────────
# All three functions are thin wrappers — they never run models or search
# algorithms.  They only render data already computed by compress() / analyze().
# ============================================================================

def plot_compression_report(
    result:          'CompressionResult',
    original_model:  nn.Module,
    dataloader,
    *,
    save_path:          str           = 'compression_report.png',
    device:             Optional[str] = None,
    show_in_notebook:   bool          = True,
) -> str:
    """
    Render and save the full compression report for a CompressionResult.

    Produces a PNG with:
      Row 0 — Accuracy / Size / Parameters / Latency bar charts
      Row 1 — Summary table + technique badges + compression radar
      Row 2+ — One panel per technique actually used (pruning, LRF,
                clustering, KD, quantization, GPTQ)

    Parameters
    ----------
    result          : CompressionResult returned by compress().
    original_model  : The original uncompressed model (needed for side-by-side
                      comparisons in the report).
    dataloader      : DataLoader used for any missing metrics — pass the
                      same test_loader you used for compress() if you want
                      any fallback measurements to stay consistent with it.
    save_path       : Output PNG path. Default 'compression_report.png'.
    device          : Device for metric fallback computation. Auto-detected.
    show_in_notebook: If True and running in Jupyter/Colab, displays the image
                      inline after saving.

    Returns
    -------
    str — absolute path to the saved PNG.

    Example
    -------
    >>> result = compress(model, dataloader, test_loader=test_loader)
    >>> plot_compression_report(result, model, test_loader)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        from sigularty.visualization import generate_compression_report as _gcr
    except ImportError:
        from visualization import generate_compression_report as _gcr

    metrics = {
        'original_accuracy':     result.original_accuracy,
        'compressed_accuracy':   result.compressed_accuracy,
        'original_size_mb':      result.size_original_mb,
        'compressed_size_mb':    result.size_compressed_mb,
        'original_latency_ms':   result.original_latency_ms,
        'compressed_latency_ms': result.compressed_latency_ms,
        'techniques_used':       result.techniques_applied,
        'cqi':                   result.cqi,
        'pruning_report':        result.pruning_report,
    }

    _gcr(
        original_model=original_model,
        compressed_model=result.model,
        metrics_dict=metrics,
        save_path=save_path,
        dataloader=dataloader,
        device=device,
    )

    abs_path = os.path.abspath(save_path)

    if show_in_notebook:
        try:
            from IPython.display import Image, display
            display(Image(abs_path))
        except Exception:
            pass

    return abs_path


def plot_pruning_report(
    result:    'CompressionResult',
    *,
    save_path: str  = 'pruning_report.png',
    show_in_notebook: bool = True,
) -> Optional[str]:
    """
    Render the per-layer structured pruning report.

    Shows which layers were pruned, how many filters were removed per layer,
    and fine-tuning recovery curves.

    Only produces output if pruning was used during compress() — returns None
    and prints a notice otherwise.

    Parameters
    ----------
    result    : CompressionResult returned by compress().
    save_path : Output PNG path. Default 'pruning_report.png'.
    show_in_notebook: Display inline in Jupyter/Colab if True.

    Returns
    -------
    str or None — absolute path to saved PNG, or None if pruning was not used.

    Example
    -------
    >>> result = compress(model, dataloader, test_loader=test_loader, use_pruning=True)
    >>> plot_pruning_report(result)
    """
    if result.pruning_report is None:
        print("  [plot_pruning_report] Pruning was not used — no report to show.")
        return None

    try:
        from sigularty.visualization import plot_pruning_report as _ppr
    except ImportError:
        from visualization import plot_pruning_report as _ppr

    _ppr(pruning_report=result.pruning_report, save_path=save_path)
    abs_path = os.path.abspath(save_path)

    if show_in_notebook:
        try:
            from IPython.display import Image, display
            display(Image(abs_path))
        except Exception:
            pass

    return abs_path


def plot_epsilon_landscape(
    epsilon_results: dict,
    search_history:  list,
    optimal_epsilon: Optional[float],
    baseline_accuracy: float,
    baseline_size_mb:  float,
    *,
    save_path: str  = 'epsilon_landscape.png',
    show_in_notebook: bool = True,
) -> str:
    """
    Render the LRF epsilon search landscape.

    Shows CQI scores across epsilon values, accuracy vs size tradeoff,
    the binary search convergence path, and the top 3 candidate epsilons.

    This is automatically called by compress() when find_optimal_epsilon=True.
    Call it manually if you ran the search yourself via optimization functions.

    Parameters
    ----------
    epsilon_results   : Dict of {epsilon_str: {accuracy, size_mb, score, phase}}
                        from find_optimal_epsilon_smart().
    search_history    : Ordered list of evaluation dicts.
    optimal_epsilon   : Best epsilon selected, or None.
    baseline_accuracy : Original model accuracy %.
    baseline_size_mb  : Original model size in MB.
    save_path         : Output PNG path.
    show_in_notebook  : Display inline in Jupyter/Colab if True.

    Returns
    -------
    str — absolute path to the saved PNG.

    Example
    -------
    >>> # Typically called automatically. Manual use:
    >>> from sigularty import plot_epsilon_landscape
    >>> plot_epsilon_landscape(results, history, 0.67, 72.1, 90.6)
    """
    try:
        from sigularty.visualization import plot_epsilon_landscape as _pel
    except ImportError:
        from visualization import plot_epsilon_landscape as _pel

    _pel(
        results=epsilon_results,
        search_history=search_history,
        optimal_epsilon=optimal_epsilon,
        baseline_accuracy=baseline_accuracy,
        baseline_size=baseline_size_mb,
        save_path=save_path,
    )
    abs_path = os.path.abspath(save_path)

    if show_in_notebook:
        try:
            from IPython.display import Image, display
            display(Image(abs_path))
        except Exception:
            pass

    return abs_path