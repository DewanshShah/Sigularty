"""
model_registry.py
=================
Registry of 20 models for the compression toolkit benchmark.

Each entry defines: architecture, task, dataset, input shape, loader function,
and why it is architecturally interesting for compression research.

The compression pipeline is fully model-agnostic — the only model-aware code
is here (loading + head replacement) and in helper_functions.py (dataset setup).

Usage
-----
    from sigularty.model_registry import SUPPORTED_MODELS, get_model_meta, load_model

    # See all models
    for name, meta in SUPPORTED_MODELS.items():
        print(f"{name:30s} {meta['architecture']:25s} {meta['dataset']}")

    # Load a model ready for the compression pipeline
    model = load_model('resnet50', num_classes=100, device='cuda')

    # Get input shape and dtype for latency measurement
    meta = get_model_meta('bert_base')
    # meta['input_shape'] = (1, 128)
    # meta['input_dtype'] = torch.long

Design rules
------------
  - Every loader sets requires_grad=True on all parameters by default.
    The caller controls which layers to freeze during fine-tuning.
  - NLP models are wrapped in NLPClassifierWrapper so the pipeline receives
    (input_ids_tensor, label_tensor) from the DataLoader and calls
    model(input_ids) → logits, identical to vision models.
  - No model is ever mutated outside its loader function.
  - Missing optional dependencies (transformers) raise ImportError with
    a clear install instruction, not a silent failure.
"""

import torch
from torch import nn
from typing import Optional


# ============================================================================
# ── NLP WRAPPER ──────────────────────────────────────────────────────────────
# Wraps HuggingFace AutoModelForSequenceClassification so it accepts a plain
# int64 input_ids tensor and returns float logits — identical interface to
# torchvision classifiers.  This means the compression pipeline (pruning,
# LRF, clustering, quantization, training loop, measure_accuracy) works
# without any NLP-specific code paths.
# ============================================================================

class NLPClassifierWrapper(nn.Module):
    """
    Thin wrapper making HuggingFace sequence classification models look like
    standard torchvision classifiers to the compression pipeline.

    Forward signature: (input_ids: LongTensor [B, L]) → logits [B, num_classes]
    The attention mask is derived internally from input_ids (1 where != pad_id).
    """

    def __init__(self, hf_model: nn.Module, pad_token_id: int = 0) -> None:
        super().__init__()
        self.model        = hf_model
        self.pad_token_id = pad_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_token_id).long()
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def extra_repr(self) -> str:
        return f"pad_token_id={self.pad_token_id}"


# ============================================================================
# ── MODEL LOADERS ────────────────────────────────────────────────────────────
# One function per model family.  Each:
#   1. Loads pretrained weights.
#   2. Replaces the classification head for num_classes.
#   3. Returns the nn.Module (not yet moved to device — caller handles that).
# ============================================================================

def _load_custom_cnn(num_classes: int) -> nn.Module:
    """
    Instantiates the CustomCNN architecture from CNNcompvision.py.
    No pretrained weights — the model is trained from scratch on CIFAR-10.
    The architecture is defined inline so the registry has no external dependency.
    hidden=16 matches the HIDDEN_UNITS constant in CNNcompvision.py.
    """
    hidden = 16

    class _CustomCNN(nn.Module):
        def __init__(self, input_ch: int, hidden: int, output: int) -> None:
            super().__init__()
            self.Conv_block1 = nn.Sequential(
                nn.Conv2d(input_ch, hidden, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, 1, 1),   nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.Conv_block2 = nn.Sequential(
                nn.Conv2d(hidden,   hidden*2, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(hidden*2, hidden*2, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.Conv_block3 = nn.Sequential(
                nn.Conv2d(hidden*2, hidden, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(hidden,   hidden, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(hidden * 4 * 4, output),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.Conv_block3(self.Conv_block2(self.Conv_block1(x))))

    return _CustomCNN(3, hidden, num_classes)


def _load_efficientnet_b0(num_classes: int) -> nn.Module:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    try:
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def _load_resnet18(num_classes: int) -> nn.Module:
    from torchvision.models import resnet18, ResNet18_Weights
    try:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _load_resnet50(num_classes: int) -> nn.Module:
    from torchvision.models import resnet50, ResNet50_Weights
    try:
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _load_resnext50(num_classes: int) -> nn.Module:
    from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
    try:
        m = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = resnext50_32x4d(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _load_wide_resnet50(num_classes: int) -> nn.Module:
    from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
    try:
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = wide_resnet50_2(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _load_vgg16(num_classes: int) -> nn.Module:
    from torchvision.models import vgg16, VGG16_Weights
    try:
        m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = vgg16(weights=None)
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m

def _load_densenet121(num_classes: int) -> nn.Module:
    from torchvision.models import densenet121, DenseNet121_Weights
    try:
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def _load_convnext_tiny(num_classes: int) -> nn.Module:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    try:
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    return m

def _load_regnet_y_400mf(num_classes: int) -> nn.Module:
    from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
    try:
        m = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = regnet_y_400mf(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _load_shufflenet_v2(num_classes: int) -> nn.Module:
    from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
    try:
        m = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = shufflenet_v2_x1_0(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _load_squeezenet(num_classes: int) -> nn.Module:
    from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
    try:
        m = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = squeezenet1_1(weights=None)
    # SqueezeNet uses Conv2d as classifier, not Linear
    m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    m.num_classes    = num_classes
    return m

def _load_mobilenet_v3_large(num_classes: int) -> nn.Module:
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    try:
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

def _load_inception_v3(num_classes: int) -> nn.Module:
    from torchvision.models import inception_v3, Inception_V3_Weights
    # torchvision enforces aux_logits=True when loading pretrained weights.
    # Load with aux_logits=True first, then disable the aux head post-load.
    # This removes the auxiliary output so training uses a single loss term,
    # and the model produces a single tensor output (required by the pipeline).
    try:
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = inception_v3(weights=None, aux_logits=True)
    m.aux_logits = False
    m.AuxLogits  = None   # prevent forward() from computing the aux branch
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _load_vit_b_16(num_classes: int) -> nn.Module:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    try:
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    return m

def _load_swin_t(num_classes: int) -> nn.Module:
    from torchvision.models import swin_t, Swin_T_Weights
    try:
        m = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    except (AttributeError, RuntimeError) as e:
        print(f"  [Registry] Warning: Could not load pretrained weights ({str(e)[:60]}...). Loading model without weights.")
        m = swin_t(weights=None)
    m.head = nn.Linear(m.head.in_features, num_classes)
    return m

def _load_bert_base(num_classes: int) -> nn.Module:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError(
            "NLP models require the transformers library.\n"
            "Install with: pip install transformers datasets --break-system-packages"
        )
    hf = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=num_classes,
    )
    tok = AutoTokenizer.from_pretrained('bert-base-uncased')
    return NLPClassifierWrapper(hf, pad_token_id=tok.pad_token_id)

def _load_distilbert(num_classes: int) -> nn.Module:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers datasets --break-system-packages")
    hf = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_classes,
    )
    tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    return NLPClassifierWrapper(hf, pad_token_id=tok.pad_token_id)

def _load_roberta_base(num_classes: int) -> nn.Module:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers datasets --break-system-packages")
    hf = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=num_classes,
    )
    tok = AutoTokenizer.from_pretrained('roberta-base')
    return NLPClassifierWrapper(hf, pad_token_id=tok.pad_token_id)

def _load_albert_base(num_classes: int) -> nn.Module:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers datasets --break-system-packages")
    hf = AutoModelForSequenceClassification.from_pretrained(
        'albert-base-v2', num_labels=num_classes,
    )
    tok = AutoTokenizer.from_pretrained('albert-base-v2')
    return NLPClassifierWrapper(hf, pad_token_id=tok.pad_token_id)

def _load_distilgpt2(num_classes: int) -> nn.Module:
    try:
        from transformers import GPT2ForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers datasets --break-system-packages")
    # GPT-2 has no PAD token by default — use EOS as PAD
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    tok.pad_token = tok.eos_token
    hf = GPT2ForSequenceClassification.from_pretrained(
        'distilgpt2', num_labels=num_classes,
        pad_token_id=tok.eos_token_id,
    )
    return NLPClassifierWrapper(hf, pad_token_id=tok.eos_token_id)


# ============================================================================
# ── MODEL REGISTRY ───────────────────────────────────────────────────────────
# Each entry is the complete specification for one model.
# Fields:
#   description     Why this model is architecturally interesting for compression
#   architecture    Short label (CNN, Transformer, NAS, etc.)
#   task            'image_classification' or 'text_classification'
#   dataset         Which dataset to use ('flowers102', 'cifar100', 'sst2')
#   num_classes     Output head size
#   input_shape     Shape for latency dummy input (includes batch dim)
#   input_dtype     torch.long for NLP (token IDs), None = float32 default
#   params_m        Approximate parameter count in millions
#   loader          Function ref: (num_classes) → nn.Module
#   notes           Compression-specific warnings or tips
# ============================================================================

_FLOAT = None   # shorthand: float32 dummy input (default for vision)
_LONG  = torch.long   # int64 dummy input for NLP token IDs

SUPPORTED_MODELS: dict = {

    # ── Custom / User-defined models ─────────────────────────────────────────
    'custom_cnn': {
        'description':  'CustomCNN — 3-block conv network trained on CIFAR-10 (32×32). '
                        'Defined in CNNcompvision.py. Hidden units=16, ~26K params.',
        'architecture': 'CNN (Custom)',
        'task':         'image_classification',
        'dataset':      'cifar10',
        'num_classes':  10,
        'input_shape':  (1, 3, 32, 32),   # CIFAR-10 native size — no resize needed
        'input_dtype':  _FLOAT,
        'params_m':     0.026,
        'loader':       _load_custom_cnn,
        'notes':        'Small custom model. No pretrained weights — loads from checkpoint. '
                        'All convs are 3×3 so skip large kernels for LRF. '
                        'Very small parameter count — clustering and quantization give the most benefit.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          False,   # too small — LRF overhead exceeds benefit
            'use_clustering':        True,
            'use_quantization':      True,
            'use_gptq':              False,   # only Linear is 256x10; below GPTQ min_layer_size
            'quant_mode':            'fp16',  # fp16 halves all conv weights; more effective than dynamic INT8 on a conv-heavy model
            'lrf_skip_large_kernels': True,
            'pretrain_epochs':       20,
            'pretrain_lr':           0.01,
        },
    },

    # ── Vision: EfficientNet family ──────────────────────────────────────────
    'efficientnet_b0': {
        'description':  'EfficientNet-B0 — NAS compound scaling, MBConv + SE. Baseline model.',
        'architecture': 'CNN (NAS/MBConv)',
        'task':         'image_classification',
        'dataset':      'flowers102',
        'num_classes':  102,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     5.3,
        'loader':       _load_efficientnet_b0,
        'notes':        'Existing baseline. Depthwise convs skipped by LRF.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': False,  # mostly 1×1 convs already
            'pretrain_epochs':       10,
            'pretrain_lr':           0.001,
        },
    },

    # ── Vision: ResNet family ────────────────────────────────────────────────
    'resnet18': {
        'description':  'ResNet-18 — classic residual network, 11M params. Smallest ResNet.',
        'architecture': 'CNN (ResNet)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     11.2,
        'loader':       _load_resnet18,
        'notes':        'No depthwise convs. LRF should skip 3×3 kernels to avoid latency regression.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,   # skip 3×3; only factorize 1×1 + Linear
            'pretrain_epochs':       20,       # CIFAR-100 needs more training
            'pretrain_lr':           0.01,
        },
    },
    'resnet50': {
        'description':  'ResNet-50 — deeper residual, bottleneck blocks, 25M params.',
        'architecture': 'CNN (ResNet)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     25.6,
        'loader':       _load_resnet50,
        'notes':        'Bottleneck has 1×1→3×3→1×1. Skip 3×3 in LRF for latency. '
                        'Pruning is very effective on the large conv channels.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,   # bottleneck 3×3 causes latency regression
            'pretrain_epochs':         15,
            'pretrain_lr':           0.00001,
        },
    },
    'resnext50_32x4d': {
        'description':  'ResNeXt-50 — grouped convolutions (32 groups × 4 channels). Tests grouped conv behaviour.',
        'architecture': 'CNN (ResNeXt)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     25.0,
        'loader':       _load_resnext50,
        'notes':        'Grouped convs (groups=32) skipped by LRF automatically. '
                        'Skip 3×3 kernels too for latency.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,
            'pretrain_epochs':       25,
            'pretrain_lr':           0.01,
        },
    },
    'wide_resnet50_2': {
        'description':  'Wide ResNet-50-2 — 2× wider than ResNet-50, 69M params. Tests wide layer compression.',
        'architecture': 'CNN (WideResNet)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     68.9,
        'loader':       _load_wide_resnet50,
        'notes':        'Wide 3×3 conv channels → 2× more launch overhead from LRF. '
                        'Skip large kernels. Clustering is very effective here (uniform weights).',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,
            'pretrain_epochs':       25,
            'pretrain_lr':           0.01,
        },
    },

    # ── Vision: VGG ─────────────────────────────────────────────────────────
    'vgg16': {
        'description':  'VGG-16 — very deep plain 3×3 conv, 138M params. No skip connections.',
        'architecture': 'CNN (VGG)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     138.4,
        'loader':       _load_vgg16,
        'notes':        'All convs are 3×3 — LRF must skip large kernels or latency doubles. '
                        'LRF on the three FC layers (4096→4096→num_classes) is extremely effective. '
                        'Clustering slow on 138M params — reduce fine_tune_epochs.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,   # ALL VGG convs are 3×3 — must skip
            'pretrain_epochs':       20,
            'pretrain_lr':           0.001,
        },
    },

    # ── Vision: Dense / efficient architectures ──────────────────────────────
    'densenet121': {
        'description':  'DenseNet-121 — dense connections (every layer connects to all later layers), 8M params.',
        'architecture': 'CNN (DenseNet)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     8.0,
        'loader':       _load_densenet121,
        'notes':        'Dense skip connections make pruning cascade widely — use low ratio. '
                        'Mix of 1×1 and 3×3 convs; skip 3×3 for latency.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,
            'pretrain_epochs':       20,
            'pretrain_lr':           0.001,
        },
    },
    'convnext_tiny': {
        'description':  'ConvNeXt-Tiny — modern CNN: large 7×7 kernels, LayerNorm, inverted bottleneck. 28M params.',
        'architecture': 'CNN (ConvNeXt)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     28.6,
        'loader':       _load_convnext_tiny,
        'notes':        'Uses 7×7 depthwise conv + 1×1 pointwise (like inverted MobileNet). '
                        'Skip large kernels — 7×7 factorization gives worse latency than 3×3.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,   # 7×7 kernels make LRF very expensive
            'pretrain_epochs':       20,
            'pretrain_lr':           0.001,
        },
    },

    # ── Vision: Mobile / lightweight ────────────────────────────────────────
    'mobilenet_v3_large': {
        'description':  'MobileNetV3-Large — depthwise + SE + hard-swish, designed for mobile inference.',
        'architecture': 'CNN (MobileNet)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     5.5,
        'loader':       _load_mobilenet_v3_large,
        'notes':        'Already highly optimized — pruning and clustering more effective than LRF. '
                        'Depthwise convs auto-skipped by LRF.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': False,  # mostly 1×1 after depthwise skip
            'pretrain_epochs':       20,
            'pretrain_lr':           0.001,
        },
    },
    'regnet_y_400mf': {
        'description':  'RegNet-Y-400MF — regular network design space, 4M params, 400MFlops budget.',
        'architecture': 'CNN (RegNet)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     4.3,
        'loader':       _load_regnet_y_400mf,
        'notes':        'Small model — compression floor test. Skip 3×3 for latency.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,
            'pretrain_epochs':       20,
            'pretrain_lr':           0.001,
        },
    },
    'shufflenet_v2_x1_0': {
        'description':  'ShuffleNetV2-x1.0 — channel shuffle operation, designed for fast mobile inference.',
        'architecture': 'CNN (ShuffleNet)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     2.3,
        'loader':       _load_shufflenet_v2,
        'notes':        'Tiny model — compression gains minimal. Good floor test.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          False,   # too small — LRF overhead > benefit
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,
            'pretrain_epochs':       20,
            'pretrain_lr':           0.001,
        },
    },
    'squeezenet1_1': {
        'description':  'SqueezeNet-1.1 — fire modules (squeeze+expand), < 1.5M params, no FC layers.',
        'architecture': 'CNN (SqueezeNet)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     1.2,
        'loader':       _load_squeezenet,
        'notes':        'Smallest model — compression likely hurts. Only clustering/fp16 recommended.',
        'recommended':  {
            'use_pruning':           False,   # too small — removes critical filters
            'use_low_rank':          False,   # too small — overhead dominates
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,
            'pretrain_epochs':       15,
            'pretrain_lr':           0.001,
        },
    },

    # ── Vision: Multi-scale / special architectures ──────────────────────────
    'inception_v3': {
        'description':  'Inception-v3 — parallel multi-scale convolutions (1×1, 3×3, 5×5), 27M params.',
        'architecture': 'CNN (Inception)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 299, 299),   # Inception requires 299×299
        'input_dtype':  _FLOAT,
        'params_m':     27.2,
        'loader':       _load_inception_v3,
        'notes':        'Requires 299×299 input. Mix of 1×1, 3×3, 5×5 branches. '
                        'Skip large kernels — only factorize 1×1 branches.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': True,
            'pretrain_epochs':       20,
            'pretrain_lr':           0.001,
        },
    },

    # ── Vision: Transformers ─────────────────────────────────────────────────
    'vit_b_16': {
        'description':  'ViT-B/16 — pure Vision Transformer, 16×16 patches, no convolutions at all. 86M params.',
        'architecture': 'Transformer (ViT)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     86.6,
        'loader':       _load_vit_b_16,
        'notes':        'No Conv2d — LRF only targets Linear (attention Q/K/V/proj, MLP). '
                        'skip_large_kernels is irrelevant (no conv). '
                        'Clustering very effective: attention weight distributions are uniform.',
        'recommended':  {
            'use_pruning':           False,  # Torch-Pruning only traces conv_proj; reducing
                                             # its output channels breaks patch embedding dim
                                             # that all 12 attention layers depend on.
                                             # Every forward pass fails with shape mismatch,
                                             # silently swallowed, 0% accuracy on all trials.
            'use_low_rank':          False,  # ViT uses fused in_proj_weight [3*768,768] inside
                                             # nn.MultiheadAttention — not 3 separate nn.Linear.
                                             # LRF only finds out_proj and factorizes it, breaking
                                             # attention output dim. Q/K/V stay full rank -> crash.
            'use_clustering':        True,
            'use_quantization':      True,
            'use_gptq':              True,   # ViT is 99% Linear (attention + MLP) — ideal for GPTQ
            'quant_mode':            'fp16',
            'lrf_skip_large_kernels': False,
            'pretrain_epochs':       15,
            'pretrain_lr':           0.00005,
        },
    },
    'swin_t': {
        'description':  'Swin-Tiny — hierarchical ViT with shifted windows. Hybrid CNN-like spatial structure.',
        'architecture': 'Transformer (Swin)',
        'task':         'image_classification',
        'dataset':      'cifar100',
        'num_classes':  100,
        'input_shape':  (1, 3, 224, 224),
        'input_dtype':  _FLOAT,
        'params_m':     28.3,
        'loader':       _load_swin_t,
        'notes':        'Mostly Linear layers in attention + patch merging. LRF very effective.',
        'recommended':  {
            'use_pruning':           True,
            'use_low_rank':          True,
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': False,
            'pretrain_epochs':       15,
            'pretrain_lr':           0.00001,
        },
    },

    # ── NLP: BERT family ─────────────────────────────────────────────────────
    'bert_base': {
        'description':  'BERT-base-uncased — 12-layer bidirectional transformer encoder. 110M params. SST-2 sentiment.',
        'architecture': 'NLP Transformer (BERT)',
        'task':         'text_classification',
        'dataset':      'sst2',
        'num_classes':  2,
        'input_shape':  (1, 128),   # (batch, seq_len) of token IDs
        'input_dtype':  _LONG,
        'params_m':     110.0,
        'loader':       _load_bert_base,
        'notes':        'requires: pip install transformers datasets\n'
                        'Attention Linear layers are primary LRF targets. '
                        'Structured pruning removes attention heads effectively.',
        'recommended':  {
            'use_pruning':           False,  # Torch-Pruning traces Conv2d only; 0 layers pruned on transformers
            'use_low_rank':          False,  # LRF on 768×768 attention increases params when rank>384
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': False,
            'pretrain_epochs':       5,
            'pretrain_lr':           0.00002,
        },
    },
    'distilbert': {
        'description':  'DistilBERT — knowledge-distilled BERT. 40% smaller, 60% faster, retains 97% of BERT performance.',
        'architecture': 'NLP Transformer (DistilBERT)',
        'task':         'text_classification',
        'dataset':      'sst2',
        'num_classes':  2,
        'input_shape':  (1, 128),
        'input_dtype':  _LONG,
        'params_m':     66.4,
        'loader':       _load_distilbert,
        'notes':        'requires: pip install transformers datasets\n'
                        'Good comparison to BERT: does additional compression help a '
                        'model that is already compressed via distillation?',
        'recommended':  {
            'use_pruning':           False,  # Torch-Pruning traces Conv2d only; 0 layers pruned on transformers
            'use_low_rank':          False,  # LRF on 768×768 attention increases params when rank>384
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': False,
            'pretrain_epochs':       5,
            'pretrain_lr':           0.00002,
        },
    },
    'roberta_base': {
        'description':  'RoBERTa-base — BERT with improved training (more data, dynamic masking). 125M params.',
        'architecture': 'NLP Transformer (RoBERTa)',
        'task':         'text_classification',
        'dataset':      'sst2',
        'num_classes':  2,
        'input_shape':  (1, 128),
        'input_dtype':  _LONG,
        'params_m':     125.0,
        'loader':       _load_roberta_base,
        'notes':        'requires: pip install transformers datasets\n'
                        'BPE tokenizer differs from BERT WordPiece — '
                        'weights should be more uniform, affecting clustering.',
        'recommended':  {
            'use_pruning':           False,  # Torch-Pruning traces Conv2d only; 0 layers pruned on transformers
            'use_low_rank':          False,  # LRF on 768×768 attention increases params when rank>384
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': False,
            'pretrain_epochs':       5,
            'pretrain_lr':           0.00002,
        },
    },
    'albert_base': {
        'description':  'ALBERT-base-v2 — parameter-sharing transformer. 12M params but 110M effective computation.',
        'architecture': 'NLP Transformer (ALBERT)',
        'task':         'text_classification',
        'dataset':      'sst2',
        'num_classes':  2,
        'input_shape':  (1, 128),
        'input_dtype':  _LONG,
        'params_m':     11.7,
        'loader':       _load_albert_base,
        'notes':        'requires: pip install transformers datasets\n'
                        'Cross-layer parameter sharing means LRF on shared weights '
                        'affects ALL layers simultaneously — fascinating compression behaviour.',
        'recommended':  {
            'use_pruning':           False,  # Torch-Pruning traces Conv2d only; 0 layers pruned on transformers
            'use_low_rank':          False,  # LRF on 768×768 attention increases params when rank>384
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': False,
            'pretrain_epochs':       5,
            'pretrain_lr':           0.00002,
        },
    },
    'distilgpt2': {
        'description':  'DistilGPT-2 — distilled autoregressive transformer adapted for classification. 82M params.',
        'architecture': 'NLP Transformer (GPT-2)',
        'task':         'text_classification',
        'dataset':      'sst2',
        'num_classes':  2,
        'input_shape':  (1, 128),
        'input_dtype':  _LONG,
        'params_m':     81.9,
        'loader':       _load_distilgpt2,
        'notes':        'requires: pip install transformers datasets\n'
                        'Causal (unidirectional) attention vs BERT bidirectional. '
                        'Uses EOS token as PAD. Unusual for classification — '
                        'tests generalization of compression to autoregressive models.',
        'recommended':  {
            'use_pruning':           False,  # Torch-Pruning traces Conv2d only; 0 layers pruned on transformers
            'use_low_rank':          False,  # LRF on 768×768 attention increases params when rank>384
            'use_clustering':        True,
            'use_quantization':      True,
            'lrf_skip_large_kernels': False,
            'pretrain_epochs':       5,
            'pretrain_lr':           0.00002,
        },
    },
}


# ============================================================================
# ── PUBLIC API ────────────────────────────────────────────────────────────────
# ============================================================================

def get_model_meta(model_name: str) -> dict:
    """
    Return metadata dict for a registered model.

    Args:
        model_name: Key in SUPPORTED_MODELS.

    Returns:
        Metadata dict (copy — safe to modify).

    Raises:
        ValueError if model_name is not registered.
    """
    if model_name not in SUPPORTED_MODELS:
        _list_available()
        raise ValueError(
            f"Unknown model '{model_name}'.  "
            f"Supported: {sorted(SUPPORTED_MODELS)}"
        )
    return dict(SUPPORTED_MODELS[model_name])


def load_model(model_name: str, num_classes: Optional[int] = None) -> nn.Module:
    """
    Instantiate a registered model with ImageNet (or HuggingFace Hub) pretrained
    weights and replace the output head for num_classes.

    Args:
        model_name:  Key in SUPPORTED_MODELS.
        num_classes: Override number of output classes.  None = use registry default.

    Returns:
        nn.Module on CPU (caller moves to device).
    """
    meta = get_model_meta(model_name)
    nc   = num_classes if num_classes is not None else meta['num_classes']
    print(f"\n  [Registry] Loading '{model_name}'  "
          f"({meta['architecture']}, {meta['params_m']}M params)")
    print(f"  [Registry] Task: {meta['task']}  Dataset: {meta['dataset']}  Classes: {nc}")
    if meta.get('notes'):
        # Print only the first line of notes to avoid cluttering the output
        note_first = meta['notes'].split('\n')[0]
        print(f"  [Registry] Note: {note_first}")
    return meta['loader'](nc)


def load_model_and_data(
    model_name:   str,
    device:       str             = 'cuda',
    train_sample: Optional[int]   = None,
    test_sample:  Optional[int]   = 500,
    batch_size:   int             = 32,
    model_path:   Optional[str]   = None,
    force_retrain: bool           = False,
    pretrain_epochs: int          = 10,
    pretrain_lr:     float        = 0.001,
) -> tuple:
    """
    Load a registered model AND its paired dataset in one call.

    This is the fast-path for testing: pick a model name, get back
    everything needed to run compress() immediately.

    Steps:
      1. Reads metadata from SUPPORTED_MODELS[model_name].
      2. Loads the model with pretrained weights + correct output head.
      3. Downloads and returns DataLoaders for the paired dataset.
      4. Optionally fine-tunes or loads a checkpoint if model_path given.

    Args:
        model_name:      Key in SUPPORTED_MODELS. e.g. 'resnet50', 'efficientnet_b0'.
        device:          'cuda' or 'cpu'. Auto-detected when 'cuda' unavailable.
        train_sample:    Max training samples. None = full dataset.
        test_sample:     Max test samples. None = full test set. Default 500 (fast).
        batch_size:      DataLoader batch size.
        model_path:      Optional path to a .pth checkpoint.
                         If the file exists, loads it (skips pretrain).
                         If None, returns the model with ImageNet pretrained weights
                         and no classification fine-tuning.
        force_retrain:   If True and model_path given, retrains even if checkpoint exists.
        pretrain_epochs: Fine-tuning epochs (only used if model_path set + no checkpoint).
        pretrain_lr:     Fine-tuning learning rate.

    Returns:
        (model, train_loader, test_loader, meta)
          model        — nn.Module on `device`, ready for compress()
          train_loader — DataLoader for training/calibration
          test_loader  — DataLoader for evaluation
          meta         — metadata dict from SUPPORTED_MODELS

    Example:
        model, train_loader, test_loader, meta = load_model_and_data('resnet50')
        result = compress(model, train_loader, num_classes=meta['num_classes'])

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS.
    """
    import torch as _torch

    # Resolve device — fall back to cpu if cuda requested but unavailable
    if device == 'cuda' and not _torch.cuda.is_available():
        print("  [Registry] CUDA not available — falling back to CPU.")
        device = 'cpu'

    meta = get_model_meta(model_name)
    print(f"\n  [Registry] Model  : {model_name}  ({meta['architecture']}, {meta['params_m']}M params)")
    print(f"  [Registry] Dataset: {meta['dataset']}  ({meta['num_classes']} classes)  device={device}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model = load_model(model_name, num_classes=meta['num_classes'])
    model = model.to(device)

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    # Import setup_data_for_model from helper_functions — deferred to avoid
    # circular imports at module load time.
    try:
        from sigularty.helper_functions import setup_data_for_model as _setup
    except ImportError:
        from sigularty.helper_functions import setup_data_for_model as _setup

    print(f"  [Registry] Loading dataset '{meta['dataset']}'...")
    train_loader, test_loader = _setup(
        model_name=model_name,
        train_sample=train_sample,
        test_sample=test_sample,
        batch_size=batch_size,
        device=device,
    )

    # ── 3. Optionally load / fine-tune checkpoint ─────────────────────────────
    if model_path is not None:
        import os as _os
        try:
            from sigularty.helper_functions import load_or_train_from_registry as _ltr
        except ImportError:
            from sigularty.helper_functions import load_or_train_from_registry as _ltr

        model = _ltr(
            model_name=model_name,
            train_loader=train_loader,
            test_loader=test_loader,
            model_path=model_path,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            device=device,
            force_retrain=force_retrain,
        )

    print(f"  [Registry] Ready. Call compress(model, train_loader, num_classes={meta['num_classes']})")
    return model, train_loader, test_loader, meta


def _list_available() -> None:
    """Print a formatted table of all registered models."""
    print("\n  Available models:")
    print(f"  {'Name':30s} {'Architecture':28s} {'Dataset':12s} {'Params':8s}")
    print(f"  {'─' * 84}")
    for name, meta in SUPPORTED_MODELS.items():
        print(f"  {name:30s} {meta['architecture']:28s} {meta['dataset']:12s} "
              f"{meta['params_m']:>5.1f}M")
    print()


def print_model_details(model_name: str) -> None:
    """Print full details for a single model."""
    meta = get_model_meta(model_name)
    print(f"\n  ═══ {model_name} ═══")
    print(f"  Description : {meta['description']}")
    print(f"  Architecture: {meta['architecture']}")
    print(f"  Task        : {meta['task']}")
    print(f"  Dataset     : {meta['dataset']}")
    print(f"  Classes     : {meta['num_classes']}")
    print(f"  Input shape : {meta['input_shape']}")
    print(f"  Params      : {meta['params_m']}M")
    if meta.get('notes'):
        print(f"  Notes       : {meta['notes']}")
    print()