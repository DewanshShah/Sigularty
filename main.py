"""
main.py
=======
Entry point for the Model Compression Toolkit.

This file owns ONLY:
  1. Hyperparameter constants — the single source of truth for every setting.
  2. main()                  — wires constants into args and runs the pipeline.

All logic lives in helper_functions.py and compression.py.
This file should never grow beyond constants + main().

CLI usage:
  python main.py
  python main.py --no-clustering --quant-mode dynamic
  python main.py --lrf-epsilon 0.3 --num-clusters 32
  python main.py --no-low-rank --no-quantization --force-retrain
  python main.py --pruning --pruning-ratio 0.2
  python main.py --find-optimal-epsilon
  python main.py --export-onnx --run-onnx
  python main.py --train-samples 0 --test-samples 0   # full dataset
  python main.py --find-optimal-epsilon --epsilon-search-ft-epochs 2 --fine-tune-abort-threshold 25
"""

import torch

from sigularty.helper_functions import parse_args, run_compression_pipeline


# ============================================================================
# HYPERPARAMETERS — edit these, not the functions
# ============================================================================

# ── Active Model ──────────────────────────────────────────────────────────────
# Change this to switch which model the toolkit compresses.
# Run: python main.py --list-models  to print all 20 options with descriptions.
#
# Model                  Architecture              Dataset       Params
# ─────────────────────────────────────────────────────────────────────
# custom_cnn             CNN (Custom)              cifar10        0.0M  ← your model
# efficientnet_b0        CNN (NAS/MBConv)          flowers102     5.3M  ← default
# resnet18               CNN (ResNet)              cifar100      11.2M
# resnet50               CNN (ResNet)              cifar100      25.6M
# resnext50_32x4d        CNN (ResNeXt)             cifar100      25.0M
# wide_resnet50_2        CNN (WideResNet)          cifar100      68.9M
# vgg16                  CNN (VGG)                 cifar100     138.4M
# densenet121            CNN (DenseNet)            cifar100       8.0M
# convnext_tiny          CNN (ConvNeXt)            cifar100      28.6M
# regnet_y_400mf         CNN (RegNet)              cifar100       4.3M
# shufflenet_v2_x1_0     CNN (ShuffleNet)          cifar100       2.3M
# squeezenet1_1          CNN (SqueezeNet)          cifar100       1.2M
# mobilenet_v3_large     CNN (MobileNet)           cifar100       5.5M
# inception_v3           CNN (Inception)           cifar100      27.2M  (299×299)
# vit_b_16               Transformer (ViT)         cifar100      86.6M  *
# swin_t                 Transformer (Swin)        cifar100      28.3M
# bert_base              NLP Transformer (BERT)    sst2         110.0M  **
# distilbert             NLP Transformer (DistilBERT) sst2       66.4M  **
# roberta_base           NLP Transformer (RoBERTa) sst2         125.0M  **
# albert_base            NLP Transformer (ALBERT)  sst2          11.7M  **
# distilgpt2             NLP Transformer (GPT-2)   sst2          81.9M  **
#
# * vit_b_16 (and any other architecture built on nn.MultiheadAttention) has
#   USE_LOW_RANK and FIND_OPTIMAL_EPSILON auto-disabled at runtime regardless
#   of this file's constants — see model_has_multihead_attention() in
#   helper_functions.py and the "LRF + nn.MultiheadAttention" section of
#   README.md for the mechanical reason (LRF wraps Linear layers in
#   nn.Sequential, which breaks nn.MultiheadAttention.forward()'s direct
#   .weight/.bias access on out_proj).
# ** NLP models require: pip install transformers datasets --break-system-packages
ACTIVE_MODEL = 'vit_b_16'

# ── Paths ─────────────────────────────────────────────────────────────────────
PRETRAIN_MODEL_PATH    = f'models/{ACTIVE_MODEL}.pth'
REPORT_SAVE_PATH       = 'compression_report.png'
EPSILON_LANDSCAPE_PATH = 'epsilon_landscape.png'
ONNX_SAVE_PATH         = 'compressed_model.onnx'

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_SAMPLE_SIZE = 5000     # 0 = all available training samples for active model
TEST_SAMPLE_SIZE  = 1000   # 0 = full test set; 500 is fast for iteration
BATCH_SIZE        = 16

# ── Model ─────────────────────────────────────────────────────────────────────
NUM_CLASSES = 10   # Flowers-102 has 102 categories

# ── Pre-training ──────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS = 10
PRETRAIN_LR     = 0.0001

# ── Technique flags ───────────────────────────────────────────────────────────
USE_PRUNING      = True
USE_LOW_RANK     = True
USE_CLUSTERING   = True
USE_QUANTIZATION = True

# ── Structured Pruning ────────────────────────────────────────────────────────
PRUNING_RATIO             = 0.3
PRUNING_MODEL_TYPE        = 'classifier'
PRUNING_FINE_TUNE_EPOCHS  = 5
PRUNING_FINE_TUNE_LR      = 0.00001
PRUNING_CAL_BATCHES       = 50
PRUNING_ITERATIVE_STEPS   = 1
PRUNING_ROUND_TO          = None    # set 8 or 16 for Tensor Core alignment
PRUNING_ISOMORPHIC        = False
PRUNING_MAX_RATIO         = 0.5     # per-group ceiling — now actually enforced.
                                    # Previously hardcoded to 1.0 (no-op) by two
                                    # interacting bugs: a pre-clamp guard that
                                    # silently collapsed the GLOBAL ratio target
                                    # down to this value instead of protecting
                                    # individual groups, and a literal `1.0`
                                    # passed to Torch-Pruning's MetaPruner
                                    # regardless of this constant. Both fixed —
                                    # see apply_structured_pruning's docstring.
PRUNING_RESIDUAL_MAX_RATIO = None  # Ceiling specifically for auto-detected
                                    # residual/skip-connection-coupled groups
                                    # (the layers whose width IS the residual
                                    # stream for an entire network stage —
                                    # collapsing them damages every downstream
                                    # block in that stage, not just one layer's
                                    # worth of capacity; see README's
                                    # "Architecture-Agnostic Residual Group
                                    # Detection" section for the full story).
                                    # None (default) = fall back to
                                    # PRUNING_MAX_RATIO above. Set an explicit
                                    # number here to give residual-critical
                                    # layers a DIFFERENT ceiling than everything
                                    # else. Applies to both the pruning search
                                    # and the final pipeline run.
PRUNING_REPORT_PATH       = 'pruning_report.png'
# NOTE: pruning's recovery fine-tune now ALWAYS uses knowledge distillation
# against the original (pre-pruning) model — see KD_TEMPERATURE / KD_ALPHA
# below, which this technique now reuses (no separate pruning-specific KD
# hyperparameters were added; the existing fine-tune epochs/lr above still
# control the recovery loop's duration and learning rate exactly as before).

# ── Low-Rank Factorization ────────────────────────────────────────────────────
LRF_EPSILON              = 0.5
LRF_MIN_LAYER_SIZE       = 64    # skip layers with dim <= this
LRF_MIN_RANK             = 2     # prevents rank-1 collapse at low epsilon
LRF_SKIP_LARGE_KERNELS   = True  # auto-set per model by registry; override here if needed
#   False → factorize all eligible conv layers (good for EfficientNet/MobileNet/ViT)
#   True  → skip Conv2d with kernel>1×1 (MUST be True for ResNet/VGG/DenseNet/Inception)
#            prevents the extra kernel launch overhead that causes latency regression

# LRF recovery fine-tune — NEW.  0 epochs = no fine-tune (LRF's old behaviour).
# When enabled, ALWAYS knowledge distillation against the original model —
# reuses KD_TEMPERATURE / KD_ALPHA below, same as pruning's and clustering's
# recovery fine-tunes.  Recommended whenever LRF_EPSILON is aggressive enough
# to cause a noticeable accuracy drop on its own.
LRF_FINE_TUNE_EPOCHS = 3
LRF_FINE_TUNE_LR     = 0.00001

# ── Weight Clustering ─────────────────────────────────────────────────────────
CLUSTER_NUM_CLUSTERS     = 16
CLUSTER_FINE_TUNE_EPOCHS = 5
CLUSTER_FINE_TUNE_LR     = 0.00001

# ── Quantization ──────────────────────────────────────────────────────────────
QUANT_MODE                    = "fp16"
QUANT_NUM_CALIBRATION_BATCHES = 100

# ── New Tier 1–3 technique flags ──────────────────────────────────────────────
# Tier 1
USE_BN_FUSION    = True    # Always on. Zero accuracy cost. Folds BN into Conv/Linear.

# Tier 2A — Sensitivity Analysis (guides per-layer pruning, runs before pruning)

# Tier 2B — Knowledge Distillation Fine-tuning (runs after LRF + clustering, and
# is now also the temperature/alpha source for pruning's and LRF's own KD
# recovery fine-tunes — see PRUNING_FINE_TUNE_* and LRF_FINE_TUNE_* above)
USE_KD_FINETUNE  = True   # True = fine-tune compressed model against original
KD_EPOCHS        = 3       # fine-tuning epochs (3 is usually enough)
KD_LR            = 0.00001  # learning rate (keep small — we're recovering, not retraining)
KD_TEMPERATURE   = 4.0     # softmax temperature (2–6); higher = softer teacher distributions
KD_ALPHA         = 0.7     # 0.7 → 70% task loss + 30% distillation loss

# Tier 3A — Adaptive LRF (per-layer epsilon from SVD energy analysis)
LRF_ADAPTIVE         = False   # True = compute epsilon per-layer analytically (better than search)
LRF_ENERGY_THRESHOLD = 0.99    # retain this fraction of each layer's SVD energy (0.95–0.99)

# Tier 3B — GPTQ Quantization (Hessian-corrected, INT4/INT8)
USE_GPTQ         = True   # True = apply GPTQ after standard compression
GPTQ_BITS        = 4       # 4 = INT4 (8× compression), 8 = INT8 (4× compression)
GPTQ_CAL_BATCHES = 16      # batches to estimate Hessian from activations
GPTQ_BLOCK_SIZE  = 128     # GPTQ block size (standard is 128)

# ── Epsilon search ────────────────────────────────────────────────────────────
FIND_OPTIMAL_EPSILON = True

# ── Global accuracy-drop threshold (applies to EVERY compression technique) ──
# This single value is now the SOLE accuracy-drop budget used everywhere:
#   - The epsilon search (LRF) and pruning hyperparameter search both use it
#     to pick the best surviving configuration.
#   - The actual pipeline ALSO uses it as a hard per-technique gate: after
#     every technique runs (Structured Pruning, LRF, Weight Clustering, GPTQ,
#     standard Quantization, and the final KD recovery fine-tune), accuracy
#     is measured and compared against the accuracy immediately before THAT
#     ONE technique ran.  If the marginal drop exceeds this threshold, that
#     technique's result is discarded and the model reverts to its
#     pre-technique state.
#   - Each technique gets its OWN independent budget — a costly earlier
#     technique does NOT eat into a later technique's allowance.  E.g. if
#     pruning alone drops accuracy by 8pp (under a 10pp threshold, so it's
#     kept), LRF is then judged ONLY on its own marginal drop from that
#     8pp-lower starting point, not against the original 10pp budget minus
#     8 already "spent".
#   - BN Fusion (mathematically lossless) and Sensitivity Analysis (read-only,
#     never mutates the model) are never gated — there's nothing to revert.
# A technique that gets reverted is NOT listed in the final report's
# "techniques used" — the report only ever describes what actually survived.
ACCURACY_DROP_THRESHOLD = 10.0
EPSILON_SEARCH_NUM_TRIALS = 15   # total evaluations for LRF epsilon search
                                  # budget: 5 anchors + rest → ternary iterations

# Epsilon search per-trial fine-tuning — NEW, opt-in via this epoch count.
# Historically epsilon search never fine-tuned: every trial applied LRF and
# measured accuracy immediately (fast — no training). 0 epochs preserves
# that exact behaviour. >0 epochs routes every trial through
# apply_low_rank_factorization's own existing fine-tune mechanism (the same
# one the main pipeline's LRF step already uses) BEFORE measuring that
# trial's accuracy.
#   COST WARNING: this multiplies epsilon search's runtime by roughly
#   EPSILON_SEARCH_FT_EPOCHS × EPSILON_SEARCH_NUM_TRIALS real training
#   epochs. On a large model, a search that previously took well under a
#   minute can now take as long as a pruning search with comparable
#   settings. FINE_TUNE_ABORT_THRESHOLD (below) caps the WORST case for
#   clearly-bad epsilons (they abort after epoch 1) — though note it can't
#   help THIS specific knob much, since EPSILON_SEARCH_FT_EPOCHS=1 means
#   there's only one epoch to begin with, nothing left to abort out of.
#   It matters far more for PRUNING_SEARCH_FT_EPOCHS (3 epochs) below.
# Default of 1 (not matched to PRUNING_SEARCH_FT_EPOCHS's 3) is a
# deliberately conservative starting point given the above — raise it once
# you've confirmed the runtime cost is acceptable for your model/trial count.
EPSILON_SEARCH_FT_EPOCHS = 1
EPSILON_SEARCH_FT_LR     = LRF_FINE_TUNE_LR   # reuse LRF's own established fine-tune LR;
                                               # no reason epsilon search needs a different one

# Early-abort threshold (pp) — applies to Structured Pruning's and
# (Adaptive) LRF's recovery fine-tunes, in BOTH the main pipeline and their
# respective hyperparameter searches (they all share the same underlying
# _kd_recovery_fine_tune function in compression.py).
#   After epoch 1 of any such fine-tune, if the accuracy drop versus the
#   ABSOLUTE ORIGINAL model already exceeds this many percentage points,
#   the remaining epochs of THAT fine-tune are skipped — recovering that
#   much ground in what's left is judged implausible, so finishing would
#   only spend compute on a result the accuracy-drop gate above would
#   revert anyway (or, during search, that the search's own final selection
#   would never pick).
#   This is a DIRECT pp value, not a multiplier on ACCURACY_DROP_THRESHOLD —
#   FINE_TUNE_ABORT_THRESHOLD = 30.0 means "abort after epoch 1 if the drop
#   already exceeds 30pp," full stop, independent of whatever
#   ACCURACY_DROP_THRESHOLD happens to be set to. (An earlier version of
#   this toolkit expressed this as a multiplier 'a' applied to
#   ACCURACY_DROP_THRESHOLD — e.g. a=3.0 x threshold=10.0 gave the same
#   30pp ceiling — but that meant changing ACCURACY_DROP_THRESHOLD silently
#   moved the abort ceiling too. A direct value is easier to reason about
#   at a glance and decouples the two knobs.)
#   Example: FINE_TUNE_ABORT_THRESHOLD=30.0 — if a config's epoch-1 drop is
#   already 31pp, recovering back under even the RELAXED
#   (ACCURACY_DROP_THRESHOLD+10pp) selection tier in the remaining epochs is
#   not realistic — finishing wastes compute on a result that was already lost.
# This is a heuristic, not a guarantee — a config could in principle still
# recover sharply in a later epoch; raise this value to take on more of that
# risk in exchange for fewer wasted epochs on genuinely hopeless configs.
# None disables it entirely (every fine-tune always runs to completion, the
# old behaviour) — use a very large number (e.g. 1000) instead of None if you
# want to keep the CLI flag's float type happy while effectively never
# triggering it.
FINE_TUNE_ABORT_THRESHOLD = 40.0
# Raised from 30.0: a real search run showed a config at 34.7pp epoch-1 drop
# get cut off, never getting the chance to recover the way a similarly-
# positioned config (17-23pp epoch-1 drop) did over its remaining 2 epochs.
# 40.0 keeps that kind of borderline case alive while still cutting off the
# clearly-hopeless cluster (55pp+ epoch-1 drops) that showed no realistic
# path to recovery in the same run.

# ── Pruning hyperparameter search ─────────────────────────────────────────────
# Set FIND_OPTIMAL_PRUNING = True to auto-search for best pruning config.
# This runs BEFORE the main pipeline and replaces PRUNING_RATIO, PRUNING_FINE_TUNE_*,
# and PRUNING_ITERATIVE_STEPS with the values found by the search.
FIND_OPTIMAL_PRUNING          = True
PRUNING_SEARCH_NUM_TRIALS     = 15   # total evaluations for pruning search
                                      # budget: ~60% ratio, ~35% grid, 1 iter_steps
PRUNING_SEARCH_FT_EPOCHS      = 3    # fine-tune epochs per trial during search
                                      # (1 is fast; increase to 2-3 for better estimates)
PRUNING_SEARCH_FT_LR          = 1e-4 # fine-tune lr per trial during search

# ── Compression Quality Index (CQI) weights ──────────────────────────────────
# Each weight is an exponent applied to that factor's ratio.
# Default = 1.0 for all: standard multiplicative scoring.
# Increase a weight to make that factor dominate search and reporting.
#
# Examples:
#   CQI_W_LATENCY = 3   → search strongly prefers faster models
#   CQI_W_ACCURACY = 2  → accuracy loss penalised quadratically
#   CQI_W_KL = 2        → search avoids pruning configs that shift output distribution
#
# Score interpretation (all weights = 1):
#   CQI = 1.0   no improvement over original
#   CQI = 2.5   2.5x better on combined weighted accuracy × compression × speedup
#
# These weights are yours to tune for your deployment goals — the toolkit
# never adjusts them automatically.  If a high CQI_W_SIZE causes the search
# to favor a heavily-compressed-but-accuracy-poor configuration, the fix is
# either ACCURACY_DROP_THRESHOLD (a hard gate the search cannot override) or
# rebalancing these weights yourself — not something this file will do for you.
CQI_W_ACCURACY = 1.0   # weight for accuracy factor
CQI_W_SIZE     = 1.5   # weight for model size / compression factor
CQI_W_LATENCY  = 1.0   # weight for latency / speedup factor
CQI_W_KL       = 1.0   # weight for KL divergence penalty (pruning only)

# ── ONNX ──────────────────────────────────────────────────────────────────────
EXPORT_ONNX = False
RUN_ONNX    = False
ONNX_OPSET  = 17

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> dict:
    """
    Wire every constant into args, then run the pipeline.

    Wiring rules:
      Boolean flags (use_*, export_*, find_*):
        Use OR so CLI --no-X can override a True constant, and CLI --X can
        enable something the constant has False.
        Pattern: args.flag = args.flag or CONSTANT

      Numeric / string constants:
        parse_args() defaults are placeholders — the real value is the constant
        here.  We always overwrite with the constant so changing it in this file
        actually takes effect.  CLI flags still win because parse_known_args()
        sets args.X to the CLI value before main() runs; we only overwrite when
        the value equals the argparse placeholder default.

      Path constants (report_path, onnx_path, etc.):
        Only override when the CLI default is still the placeholder string,
        so an explicit CLI path is never clobbered.
    """
    args = parse_args()

    # ── Boolean / flag constants ──────────────────────────────────────────────
    # The OR pattern lets CLI --flag enable a technique even when the constant
    # is False.  When the constant is False AND no CLI flag was passed, the
    # result is False — i.e. the constant correctly disables the technique.
    args.use_bn_fusion        = getattr(args, 'use_bn_fusion', False)    or USE_BN_FUSION
    args.use_pruning          = args.use_pruning      or USE_PRUNING
    args.use_low_rank         = args.use_low_rank     or USE_LOW_RANK
    args.use_clustering       = args.use_clustering   or USE_CLUSTERING
    args.use_kd_finetune      = getattr(args, 'use_kd_finetune', False)  or USE_KD_FINETUNE
    args.use_quantization     = args.use_quantization or USE_QUANTIZATION
    args.use_gptq             = getattr(args, 'use_gptq', False)         or USE_GPTQ
    args.lrf_adaptive         = getattr(args, 'lrf_adaptive', False)     or LRF_ADAPTIVE
    args.export_onnx          = args.export_onnx      or EXPORT_ONNX
    args.accuracy_drop_threshold = ACCURACY_DROP_THRESHOLD
    args.run_onnx             = args.run_onnx         or RUN_ONNX
    args.find_optimal_epsilon    = args.find_optimal_epsilon or FIND_OPTIMAL_EPSILON
    if args.epsilon_search_num_trials == 15:   args.epsilon_search_num_trials  = EPSILON_SEARCH_NUM_TRIALS
    if getattr(args, 'epsilon_search_ft_epochs', 1) == 1:
        args.epsilon_search_ft_epochs = EPSILON_SEARCH_FT_EPOCHS
    if getattr(args, 'epsilon_search_ft_lr', 0.00001) == 0.00001:
        args.epsilon_search_ft_lr = EPSILON_SEARCH_FT_LR
    args.find_optimal_pruning     = getattr(args, 'find_optimal_pruning', False) or FIND_OPTIMAL_PRUNING
    if getattr(args, 'pruning_search_num_trials', 16) == 16:
        args.pruning_search_num_trials = PRUNING_SEARCH_NUM_TRIALS
    # NOTE: these two comparisons must match the CLI flags' own `default=`
    # values (now that --pruning-search-ft-epochs/--pruning-search-ft-lr
    # exist) — they used to fall back to getattr(..., 1) / getattr(..., 1e-4)
    # purely as placeholders because NO CLI flag existed yet, so the
    # constant always won unconditionally. Now that the flags exist with
    # defaults of 3 / 1e-4 (matching the constants below), comparing against
    # the OLD placeholder values (1) would silently stop this override from
    # ever firing the moment someone changes the constant to anything other
    # than 3 — comparing against 3 / 1e-4 is what keeps "constant wins
    # unless the user passes an explicit CLI value" working correctly.
    if getattr(args, 'pruning_search_ft_epochs', 3) == 3:
        args.pruning_search_ft_epochs = PRUNING_SEARCH_FT_EPOCHS
    if getattr(args, 'pruning_search_ft_lr', 1e-4) == 1e-4:
        args.pruning_search_ft_lr = PRUNING_SEARCH_FT_LR
    if getattr(args, 'fine_tune_abort_threshold', 30.0) == 30.0:
        args.fine_tune_abort_threshold = FINE_TUNE_ABORT_THRESHOLD

    # ── Search guards — don't waste time searching for disabled techniques ───
    # If the technique itself is off, its hyperparameter search must also be off
    # regardless of FIND_OPTIMAL_PRUNING / FIND_OPTIMAL_EPSILON constants.
    if not args.use_pruning:
        args.find_optimal_pruning = False
    if not args.use_low_rank:
        args.find_optimal_epsilon = False

    # ── Quantization coupling — FP16 is only meaningful alongside GPTQ ───────
    # GPTQ compresses specific layers to INT4; FP16 then covers the remainder.
    # With GPTQ off, applying FP16 alone gives 2× at the cost of precision with
    # no INT4 benefit — so we disable FP16 automatically when GPTQ is off.
    if not args.use_gptq:
        args.use_quantization = False

    # ── Numeric / string — constant always wins unless CLI flag was explicit ───
    # CLI detection: argparse sets the value to its default when no flag is
    # passed.  We compare against that default and replace with the constant.
    # If the user passed an explicit CLI value it will differ from the default,
    # so the condition is False and the CLI value is preserved.
    if args.lrf_epsilon          == 0.5:      args.lrf_epsilon          = LRF_EPSILON
    if args.lrf_min_layer_size   in (10, 64): args.lrf_min_layer_size   = LRF_MIN_LAYER_SIZE
    if args.lrf_min_rank         == 2:        args.lrf_min_rank         = LRF_MIN_RANK
    # lrf_skip_large_kernels: use constant (registry is only advisory now)
    args.lrf_skip_large_kernels = LRF_SKIP_LARGE_KERNELS
    if getattr(args, 'lrf_fine_tune_epochs', 0) == 0:    args.lrf_fine_tune_epochs = LRF_FINE_TUNE_EPOCHS
    if getattr(args, 'lrf_fine_tune_lr', 0.0001) == 0.0001: args.lrf_fine_tune_lr  = LRF_FINE_TUNE_LR
    if args.num_clusters         == 16:       args.num_clusters         = CLUSTER_NUM_CLUSTERS
    if args.cluster_fine_tune_epochs == 5:    args.cluster_fine_tune_epochs = CLUSTER_FINE_TUNE_EPOCHS
    if args.cluster_fine_tune_lr == 0.0001:   args.cluster_fine_tune_lr = CLUSTER_FINE_TUNE_LR
    if args.quant_mode           == 'dynamic':args.quant_mode           = QUANT_MODE
    if args.quant_cal_batches    == 100:      args.quant_cal_batches    = QUANT_NUM_CALIBRATION_BATCHES
    # New technique numerics
    if getattr(args, 'kd_epochs', 3)            == 3:    args.kd_epochs           = KD_EPOCHS
    if getattr(args, 'kd_lr', 0.0001)           == 0.0001: args.kd_lr             = KD_LR
    if getattr(args, 'kd_temperature', 4.0)     == 4.0:  args.kd_temperature      = KD_TEMPERATURE
    if getattr(args, 'kd_alpha', 0.7)           == 0.7:  args.kd_alpha            = KD_ALPHA
    if getattr(args, 'lrf_energy_threshold', 0.99) == 0.99: args.lrf_energy_threshold = LRF_ENERGY_THRESHOLD
    if getattr(args, 'gptq_bits', 4)            == 4:    args.gptq_bits           = GPTQ_BITS
    if getattr(args, 'gptq_cal_batches', 16)    == 16:   args.gptq_cal_batches    = GPTQ_CAL_BATCHES
    if getattr(args, 'gptq_block_size', 128)    == 128:  args.gptq_block_size     = GPTQ_BLOCK_SIZE
    if args.train_samples        == 1000:     args.train_samples        = TRAIN_SAMPLE_SIZE
    if args.test_samples         == 500:      args.test_samples         = TEST_SAMPLE_SIZE
    if args.batch_size           == 32:       args.batch_size           = BATCH_SIZE
    if args.pretrain_epochs      == 10:       args.pretrain_epochs      = PRETRAIN_EPOCHS
    if args.pretrain_lr          == 0.001:    args.pretrain_lr          = PRETRAIN_LR
    if args.onnx_opset           == 17:       args.onnx_opset           = ONNX_OPSET

    # Pruning numerics
    if args.pruning_ratio              == 0.3:          args.pruning_ratio             = PRUNING_RATIO
    if args.pruning_model_type         == 'classifier': args.pruning_model_type        = PRUNING_MODEL_TYPE
    if args.pruning_fine_tune_epochs   == 3:            args.pruning_fine_tune_epochs  = PRUNING_FINE_TUNE_EPOCHS
    if args.pruning_fine_tune_lr       == 0.0001:       args.pruning_fine_tune_lr      = PRUNING_FINE_TUNE_LR
    if args.pruning_cal_batches        == 50:           args.pruning_cal_batches       = PRUNING_CAL_BATCHES
    if args.pruning_iterative_steps    == 1:            args.pruning_iterative_steps   = PRUNING_ITERATIVE_STEPS
    if args.pruning_round_to           is None:         args.pruning_round_to          = PRUNING_ROUND_TO
    # isomorphic is a store_true flag — OR pattern
    args.pruning_isomorphic = args.pruning_isomorphic or PRUNING_ISOMORPHIC
    if args.pruning_max_ratio == 1.0:   args.pruning_max_ratio = PRUNING_MAX_RATIO
    # No CLI flag for this one — main.py constant only. Direct assignment,
    # no sentinel-comparison dance needed since there's no CLI value that
    # could ever compete with it.
    args.pruning_residual_max_ratio = PRUNING_RESIDUAL_MAX_RATIO

    # ── Active model — drives dataset, input shape, num_classes ─────────────
    args.active_model = getattr(args, 'active_model', None) or ACTIVE_MODEL
    # When using the registry, num_classes is read inside the pipeline from
    # the model metadata.  Keep it as a fallback for legacy EfficientNet flow.
    args.num_classes = NUM_CLASSES

    # ── Registry metadata — override pretrain_epochs and pretrain_lr only ───────
    # main.py hyperparameters take precedence for all OTHER settings.
    # Registry recommendations override ONLY pretrain_epochs and pretrain_lr.
    # This ensures your USE_PRUNING, USE_CLUSTERING, etc. constants always work as expected,
    # while allowing the model registry to guide dataset-specific training hyperparameters.
    # (LRF is a special case: nn.MultiheadAttention architectures get
    # use_low_rank / find_optimal_epsilon force-disabled at RUNTIME inside
    # run_compression_pipeline, regardless of this file's constants — see the
    # ACTIVE_MODEL comment above. That is a correctness safety check, not a
    # registry "recommendation", so it is not handled here.)
    _cli_model_path_default = 'models/efficientnet_b0_flowers102.pth'
    try:
        from sigularty.model_registry import SUPPORTED_MODELS
        if args.active_model in SUPPORTED_MODELS:
            _rec = SUPPORTED_MODELS[args.active_model].get('recommended', {})

            # Print registry recommendations as informational notes
            if _rec:
                print(f"\n  [Registry] Recommended config for '{args.active_model}':")
                if 'use_pruning' in _rec:
                    status = "enabled" if _rec['use_pruning'] else "disabled"
                    print(f"    use_pruning = {status}  (main.py: {args.use_pruning})")
                if 'use_low_rank' in _rec:
                    status = "enabled" if _rec['use_low_rank'] else "disabled"
                    print(f"    use_low_rank = {status}  (main.py: {args.use_low_rank})")
                if 'use_clustering' in _rec:
                    status = "enabled" if _rec['use_clustering'] else "disabled"
                    print(f"    use_clustering = {status}  (main.py: {args.use_clustering})")
                if 'use_quantization' in _rec:
                    status = "enabled" if _rec['use_quantization'] else "disabled"
                    print(f"    use_quantization = {status}  (main.py: {args.use_quantization})")
                if 'lrf_skip_large_kernels' in _rec:
                    print(f"    lrf_skip_large_kernels = {_rec['lrf_skip_large_kernels']}"
                          f"  (main.py: {args.lrf_skip_large_kernels})")
                
                # Registry overrides ONLY pretrain_epochs and pretrain_lr.
                # All technique flags (use_pruning, use_low_rank, etc.) are controlled
                # exclusively by main.py constants — the registry never overrides them.
                if _rec.get('pretrain_epochs'):
                    print(f"    Recommended pretrain_epochs: {_rec['pretrain_epochs']}  (main.py: {args.pretrain_epochs})")
                    args.pretrain_epochs = _rec['pretrain_epochs']
                if _rec.get('pretrain_lr'):
                    print(f"    Recommended pretrain_lr: {_rec['pretrain_lr']}  (main.py: {args.pretrain_lr})")
                    args.pretrain_lr = _rec['pretrain_lr']
                print(f"  [Registry] Using registry values for pretrain_epochs & pretrain_lr; main.py constants for all others.")
    except ImportError:
        pass

    # Auto-generate a model-specific checkpoint path when using the registry.
    if args.active_model != 'efficientnet_b0' and args.model_path == _cli_model_path_default:
        args.model_path = f"models/{args.active_model}.pth"
    elif args.active_model == 'efficientnet_b0' and args.model_path == _cli_model_path_default:
        args.model_path = PRETRAIN_MODEL_PATH

    # ── CQI weights ───────────────────────────────────────────────────────────
    args.cqi_w_accuracy = CQI_W_ACCURACY
    args.cqi_w_size     = CQI_W_SIZE
    args.cqi_w_latency  = CQI_W_LATENCY
    args.cqi_w_kl       = CQI_W_KL

    # ── --list-models: print registry and exit ────────────────────────────────
    if getattr(args, 'list_models', False):
        try:
            from sigularty.model_registry import SUPPORTED_MODELS
            print("\nAvailable models for ACTIVE_MODEL:")
            print(f"  {'Name':30s} {'Architecture':28s} {'Dataset':12s} {'Params':8s}")
            print(f"  {'─' * 84}")
            for name, meta in SUPPORTED_MODELS.items():
                print(f"  {name:30s} {meta['architecture']:28s} {meta['dataset']:12s} "
                      f"{meta['params_m']:>5.1f}M")
            print()
        except ImportError:
            print("model_registry.py not found.")
        import sys; sys.exit(0)

    # ── Path constants — only override when still at CLI placeholder ───────────
    # Note: args.model_path is already set above (auto-generated from active_model
    # or PRETRAIN_MODEL_PATH). Do NOT override it here again.
    if args.report_path  == 'compression_report.png':
        args.report_path = REPORT_SAVE_PATH
    if args.onnx_path    == 'compressed_model.onnx':
        args.onnx_path   = ONNX_SAVE_PATH
    if args.pruning_report_path == 'pruning_report.png':
        args.pruning_report_path = PRUNING_REPORT_PATH
    # epsilon_landscape_path has no CLI flag — always from main.py
    args.epsilon_landscape_path = EPSILON_LANDSCAPE_PATH
    # device: run_compression_pipeline re-detects CUDA at runtime;
    # DEVICE constant here is documentation / fallback only.

    return run_compression_pipeline(args)


if __name__ == "__main__":
    main()