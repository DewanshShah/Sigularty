"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import argparse
from typing import Optional, Tuple

import os
import zipfile

from pathlib import Path

import requests

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

import matplotlib.pyplot as plt
from torch import nn
import torch
import sys
from tqdm.auto import tqdm
import torchmetrics

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(10, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path


def plot_predictions(train_data, train_labels, test_data, test_labels,predictions = None):

    train_data = train_data.cpu() if train_data.device.type == "cuda" else train_data
    train_labels = train_labels.cpu() if train_labels.device.type == "cuda" else train_labels
    test_data = test_data.cpu() if test_data.device.type == "cuda" else test_data
    test_labels = test_labels.cpu() if test_labels.device.type == "cuda" else test_labels

    plt.figure(figsize = (10,7))

    plt.scatter(train_data,train_labels, c = "b",s = 4)

    plt.scatter(test_data,test_labels, c = "r",s = 4)

    if predictions is not None:
        plt.scatter(test_data, predictions, c="g",s = 4)

    plt.show()

def model_eval(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, accuracy_fn,device):
    
    loss, acc = 0,0

    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

            x,y = X.to(device), y.to(device)

            test_pred = model(x)

            loss = loss_fn(test_pred, y)
            loss += loss.item()

            acc += accuracy_fn(y, test_pred.argmax(dim=1))
        
        loss /= len(dataloader)
        acc /= len(dataloader)

    return {"model name": model.__class__.__name__, "loss": loss.item(), "accuracy": acc}

def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn, device):

    model.to(device)
    model.train()

    total_loss, total_accuracy = 0, 0


    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)
        acc = accuracy_fn(y, y_pred.argmax(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += acc

    # Return average loss and average accuracy across all batches
    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device):

    model.to(device)
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for x_test, y_test in dataloader:
            x_test, y_test = x_test.to(device), y_test.to(device)

            test_pred = model(x_test)

            test_loss += loss_fn(test_pred, y_test).item()
            test_acc += (test_pred.argmax(dim=1) == y_test).sum().item()/len(test_pred)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs = 1, device = "cuda" if torch.cuda.is_available() else "cpu", num_classes=3):

    accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           accuracy_fn=accuracy_fn)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device,
                                        accuracy_fn=accuracy_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

# ============================================================================
# SHARED COMPRESSION UTILITIES
# These functions are used by compression.py, visualization.py, and main.py.
# They are placed here because they are technique-agnostic and reused across
# multiple modules (compression measurement, latency benchmarking, accuracy
# evaluation).  Nothing below this line is technique-specific.
# ============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate total model size in megabytes (parameters + buffers).

    Args:
        model: Any nn.Module.

    Returns:
        Size in MB as a float.
    """
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in a model.

    Args:
        model: Any nn.Module.

    Returns:
        Integer count of parameters where requires_grad is True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(
    model: nn.Module,
    input_shape: tuple,
    device: str,
    num_iterations: int = 100,
    warmup: int = 10,
    input_dtype=None,
) -> dict:
    """
    Measure model inference latency with proper GPU synchronisation.

    Args:
        model:          Model to benchmark.
        input_shape:    Shape of a single dummy input tensor (includes batch dim).
        device:         'cuda' or 'cpu'.
        num_iterations: Number of timed forward passes.
        warmup:         Untimed warmup passes to prime caches / JIT kernels.
        input_dtype:    Override input tensor dtype.  None = float32 (vision).
                        Pass torch.long for NLP models using token IDs.

    Returns:
        Dict with keys: mean_ms, median_ms, p95_ms, p99_ms.
    """
    import time as _time

    # Use _eval_model if attached (static-quantized ResNet workaround)
    run_model = getattr(model, '_eval_model', model)
    run_model.eval()
    # INT8 quantized models cannot be moved with .to()
    _is_quantized = any(
        'quantized' in type(m).__name__.lower() for m in run_model.modules()
    )
    if not _is_quantized:
        run_model.to(device)
    try:
        _model_dtype = next(run_model.parameters()).dtype
    except StopIteration:
        _model_dtype = torch.float32
    # input_dtype overrides for NLP models (token IDs are int64, not float32)
    if input_dtype is not None:
        dummy_input = torch.zeros(input_shape, dtype=input_dtype).to(device=device)
    else:
        dummy_input = torch.randn(input_shape).to(device=device, dtype=_model_dtype)

    # Warmup (not timed)
    with torch.no_grad():
        for _ in range(warmup):
            _ = run_model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = _time.perf_counter()
            _ = run_model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = _time.perf_counter()
            latencies.append((end - start) * 1000)   # → milliseconds

    sorted_lat = sorted(latencies)
    n = len(latencies)
    return {
        'mean_ms':   sum(latencies) / n,
        'median_ms': sorted_lat[n // 2],
        'p95_ms':    sorted_lat[int(n * 0.95)],
        'p99_ms':    sorted_lat[int(n * 0.99)],
    }


def measure_accuracy(
    model: nn.Module,
    dataloader,
    device: str,
) -> float:
    """
    Measure top-1 classification accuracy on a DataLoader.

    For static-quantized models that have a ._eval_model attribute (set by
    _apply_static_quantization in compression.py), the eval_model is used
    for inference instead of the raw INT8 model.  This is necessary because
    ResNet residual additions (plain '+') crash on QuantizedCPU tensors.
    The size of the INT8 model is still reported correctly elsewhere.

    Failure handling (two distinct cases, both new):
      1. STRUCTURAL FAILURE — every single batch raised an exception (e.g. a
         Sequential-wrapped layer breaking a parent module's direct
         .weight/.bias access, such as nn.MultiheadAttention.out_proj after
         LRF).  Previously this silently returned a fake 0.0%, which is
         indistinguishable from "the model is just bad" and let several real
         bugs hide for a long time.  Now this RAISES a RuntimeError whose
         message contains "No samples were evaluated" — callers in
         optimization.py specifically check for this substring to trigger
         their consecutive-structural-failure search-abort protocol, and
         compression.py's fine-tune loops surface it instead of training on
         a model that can never produce a gradient signal.
      2. GENUINE ZERO — at least one batch succeeded (so `total > 0`), but
         literally nothing was predicted correctly (`correct == 0`).  This is
         a real, valid result (not a bug) so it is still RETURNED normally —
         but a major warning is printed, since 0.0% on a real evaluation
         almost always indicates something is badly wrong (wrong
         num_classes, NaN/dead weights, mismatched output head) even though
         it technically "ran".

      Per-batch tolerance is unchanged: individual batches that raise a
      shape/Sequential-related error are still skipped (not every odd batch
      is necessarily a sign of total failure) — what changed is what happens
      when ALL of them are skipped.

    Args:
        model:      Model to evaluate.
        dataloader: Iterable of (X, y) batches.
        device:     'cuda' or 'cpu'.

    Returns:
        Accuracy as a percentage in [0.0, 100.0].

    Raises:
        RuntimeError: if every batch failed (message contains
                      "No samples were evaluated").
    """
    # Use _eval_model if attached (static-quantized ResNet workaround)
    run_model = getattr(model, '_eval_model', model)
    run_model.eval()
    # INT8 quantized models cannot be moved with .to()
    _is_quantized = any(
        'quantized' in type(m).__name__.lower() for m in run_model.modules()
    )
    if not _is_quantized:
        run_model.to(device)
    # Detect model dtype once (handles fp16, float32, bfloat16)
    try:
        _model_dtype = next(run_model.parameters()).dtype
    except StopIteration:
        _model_dtype = torch.float32

    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            try:
                X, y = X.to(device), y.to(device)
                if X.is_floating_point():
                    X = X.to(dtype=_model_dtype)
                predictions = run_model(X).argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
            except (RuntimeError, AttributeError) as e:
                # Skip batches that cause shape mismatches or attribute errors
                # This can happen with LRF-wrapped Sequential layers on some inputs
                if 'shape' in str(e).lower() or 'sequential' in str(e).lower():
                    continue
                raise

    if total == 0:
        raise RuntimeError(
            "measure_accuracy: No samples were evaluated — every batch raised "
            "an exception. This is almost always a structural break (e.g. a "
            "Sequential-wrapped layer lacking the .weight/.bias attributes a "
            "parent module's forward() accesses directly, such as "
            "nn.MultiheadAttention.out_proj after Low-Rank Factorization), not "
            "ordinary poor accuracy. Reported as a hard failure rather than a "
            "silent 0.0% so search loops and pipelines can detect and react to it."
        )

    if correct == 0:
        print(
            "  🚨 MAJOR WARNING: measure_accuracy returned EXACTLY 0.0% "
            f"({total} samples evaluated, 0 correct). This is usually a real, "
            "severe failure (wrong num_classes, NaN/dead weights, mismatched "
            "output head) rather than ordinary poor performance — investigate "
            "before trusting downstream results."
        )

    return (correct / total) * 100


def model_has_multihead_attention(model: nn.Module) -> int:
    """
    Count nn.MultiheadAttention modules anywhere in `model`.

    Used by run_compression_pipeline() to detect architectures (e.g.
    torchvision's vit_b_16) where Low-Rank Factorization cannot be safely
    applied: nn.MultiheadAttention.forward() calls
    F.multi_head_attention_forward() with self.out_proj.weight / .bias
    accessed DIRECTLY as raw tensors — it never calls out_proj.forward() —
    so wrapping out_proj in nn.Sequential (what LRF's
    _decompose_linear_layer does) breaks every forward pass with
    AttributeError: 'Sequential' object has no attribute 'weight'.

    compression.py's apply_low_rank_factorization() and apply_adaptive_lrf()
    both independently re-check this and bail out to a no-op if called
    directly with such a model — this pipeline-level check exists purely to
    avoid wasting an entire epsilon search (15+ trials, each a full
    deepcopy+factorize+measure cycle) on a model where every trial would
    silently no-op and report unchanged baseline numbers.

    Args:
        model: Any nn.Module.

    Returns:
        Count of nn.MultiheadAttention modules found (0 if none).
    """
    return sum(1 for m in model.modules() if isinstance(m, nn.MultiheadAttention))


def gate_technique_accuracy(
    technique_label: str,
    pre_model: nn.Module,
    post_model: nn.Module,
    test_loader,
    device: str,
    accuracy_drop_threshold: float,
    pre_accuracy: Optional[float] = None,
) -> Tuple[nn.Module, float, bool]:
    """
    Global per-technique accuracy gate.

    Measures `post_model`'s accuracy (and `pre_model`'s, if `pre_accuracy`
    isn't already known) and compares the MARGINAL drop caused by this one
    technique against `accuracy_drop_threshold`.  Each technique gets its
    own independent budget — this function only ever looks at the
    immediately-before vs immediately-after accuracy for the technique being
    gated, never a cumulative total across the whole pipeline.

    If the drop exceeds the threshold: `post_model` is discarded (explicit
    `del` + `torch.cuda.empty_cache()` on CUDA) and `pre_model` is returned
    unchanged, so the pipeline reverts to the state immediately before this
    technique ran.

    If the drop is within the threshold (including accuracy IMPROVING, i.e.
    a negative "drop"): `post_model` is kept and returned.

    Callers should pass `pre_accuracy` whenever it is already known (e.g.
    carried forward from the previous technique's gate result) to avoid a
    redundant measurement — this function only measures `pre_model` itself
    when `pre_accuracy is None`.

    Args:
        technique_label:          Human-readable name, used in print output
                                  and as the dict key in callers' gating reports.
        pre_model:                Model state immediately before this technique ran.
        post_model:               Model state immediately after this technique ran.
        test_loader:              DataLoader used to measure accuracy.
        device:                   Device to measure `post_model` on (and
                                  `pre_model` too, if `pre_accuracy is None`).
                                  Callers are responsible for passing the
                                  device each model actually resides on —
                                  e.g. a dynamic-INT8-quantized model lives on
                                  CPU regardless of what device earlier
                                  technique stages ran on.
        accuracy_drop_threshold:  Max allowed marginal drop (pp) for this
                                  technique alone.
        pre_accuracy:             Known accuracy of `pre_model`, if already
                                  measured.  None = measure it now.

    Returns:
        (kept_model, kept_model_accuracy, was_kept)
          kept_model:          Either post_model (kept) or pre_model (reverted).
          kept_model_accuracy: Accuracy of whichever model was kept.
          was_kept:            True if post_model survived the gate.
    """
    if pre_accuracy is None:
        pre_accuracy = measure_accuracy(pre_model, test_loader, device)

    post_accuracy = measure_accuracy(post_model, test_loader, device)
    drop = pre_accuracy - post_accuracy

    if drop > accuracy_drop_threshold:
        print(
            f"  ❌ [{technique_label}] SKIPPED — accuracy dropped {drop:.2f}pp "
            f"({pre_accuracy:.2f}% → {post_accuracy:.2f}%), exceeding the "
            f"global threshold of {accuracy_drop_threshold:.1f}pp. Reverting "
            f"to the pre-{technique_label} model."
        )
        del post_model
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pre_model, pre_accuracy, False
    else:
        _direction = "improved by" if drop < 0 else "dropped"
        print(
            f"  ✅ [{technique_label}] kept — accuracy {pre_accuracy:.2f}% → "
            f"{post_accuracy:.2f}% ({_direction} {abs(drop):.2f}pp, within "
            f"{accuracy_drop_threshold:.1f}pp threshold)."
        )
        return post_model, post_accuracy, True


def report_technique_impact(
    technique_label: str,
    pre_model: nn.Module,
    post_model: nn.Module,
    pre_accuracy: float,
    post_accuracy: float,
    was_kept: bool,
    pre_device: str,
    post_device: str,
    original_accuracy: float,
    original_size_mb: float,
    original_latency_ms: float,
    input_shape: tuple = (1, 3, 224, 224),
    input_dtype: Optional[torch.dtype] = None,
    pre_size_mb: Optional[float] = None,
    pre_latency_ms: Optional[float] = None,
    pre_finetune_accuracy: Optional[float] = None,
    structural_summary: Optional[str] = None,
    structural_zero: bool = False,
    latency_iterations: int = 10,
    latency_warmup: int = 3,
) -> dict:
    """
    Print (and return) a detailed impact breakdown for one gated technique —
    what it changed, isolated from what every OTHER technique in the
    pipeline changed.

    This solves a real ambiguity that plain before/after accuracy numbers
    hide: a technique that is structurally a no-op on a given architecture
    (e.g. Low-Rank Factorization when every Conv2d is 3x3 and
    skip_large_kernels=True, or GPTQ when the only Linear layer is smaller
    than min_layer_size) can still show a healthy-looking accuracy IMPROVE-
    MENT in the pipeline's gate log, purely because its own internal
    recovery fine-tune ran a few epochs of gradient descent on top of an
    UNCHANGED model. Read in isolation, that gate line ("✅ kept — accuracy
    71.00% → 73.40%, improved by 2.40pp") looks like the technique earned
    its 2.40pp. It didn't — the fine-tune did, and the technique itself did
    nothing this run. This function exists to make that distinction
    impossible to miss.

    Three independent numbers are reported for accuracy, size, and latency:

      MARGINAL   — this technique's own before/after, i.e. exactly what
                   gate_technique_accuracy already measures for accuracy.
                   Answers "what did THIS step change, given whatever state
                   the pipeline was already in."
      CUMULATIVE — after this technique vs. the ABSOLUTE original model
                   (original_accuracy/size/latency, measured once at the
                   very start of the run and passed through unchanged to
                   every call). Answers "how much of the pipeline's total
                   compression so far can this technique take credit for" —
                   which for a downstream technique is usually "most of it
                   is carried over from earlier steps, not new." A true
                   no-op technique will show marginal ≈ 1.00x / 0.00pp while
                   cumulative still shows a large number — that gap IS the
                   signal that cumulative was inherited, not earned here.
      ALGORITHM vs. FINE-TUNE split (accuracy only, when available) —
                   Pruning, LRF, and Weight Clustering each bundle a
                   recovery fine-tune INSIDE the same function call this
                   gate measures around, so the plain marginal number
                   conflates "what the compression algorithm did to the
                   weights" with "what a few epochs of KD gradient descent
                   recovered afterward." When the technique's own function
                   supplies pre_finetune_accuracy (an accuracy snapshot
                   taken after the structural change but before its
                   internal fine-tune starts — opt-in, only computed when
                   the caller explicitly requested it), this splits the
                   marginal delta into its two components:
                       algorithm delta  = pre_accuracy - pre_finetune_accuracy
                       fine-tune delta  = pre_finetune_accuracy - post_accuracy
                   GPTQ and standard Quantization have no internal fine-tune
                   loop at all (their only recovery happens later, in the
                   shared post-quantization KD step), so their marginal
                   number is already a clean read with nothing to split —
                   callers simply don't pass pre_finetune_accuracy for them.
                   The final post-quantization KD step IS the fine-tune (no
                   separate algorithm phase precedes it), so it's excluded
                   from this split for the same reason — it reduces to
                   marginal + cumulative only.

    Zero-effect flagging: when structural_zero=True (the caller has already
    determined, from the technique's own structural report, that literally
    nothing was touched — e.g. "0/7 layers factorized"), this prints an
    explicit recommendation to disable the technique for this architecture,
    regardless of what the accuracy numbers show. This is deliberately
    LOUDER than a plain marginal-accuracy-near-zero check would be, because
    a technique can be structurally inert while STILL showing accuracy
    movement (entirely from its bundled fine-tune) — the exact failure mode
    this function exists to catch.

    Scope: pipeline-only by design. This is only ever called from inside
    apply_compression_pipeline() and run_compression_pipeline() — never
    from the hyperparameter search functions in optimization.py, which are
    explicitly reduced-fidelity (fewer fine-tune epochs, fewer calibration
    batches) and whose numbers wouldn't mean the same thing. See
    capture_pre_finetune_accuracy in apply_structured_pruning /
    apply_low_rank_factorization / apply_adaptive_lrf / apply_weight_
    clustering for how that boundary is enforced upstream — search code
    never sets that flag, so it never pays for or produces the snapshot
    this function would otherwise look for.

    Cost: always one get_model_size_mb() call each for pre/post (no forward
    passes, negligible). Latency costs one measure_latency() call each for
    pre/post, at REDUCED fidelity (latency_iterations/latency_warmup default
    to 10/3 — the same reduced-fidelity convention optimization.py's search
    functions already use for intermediate, non-final measurements) rather
    than the 100/10 the final compression report uses. No accuracy is
    measured here at all — pre_accuracy/post_accuracy are REQUIRED params
    because gate_technique_accuracy has, by construction, already measured
    both by the time this is called; re-measuring here would be pure waste.
    Latency measurement is wrapped in try/except — a forward-pass failure
    in a reporting helper must never take down the actual compression run,
    it should just print "N/A" and move on.

    Args:
        technique_label:       Human-readable name, matches the gate's label.
        pre_model:              Model state immediately before this technique.
        post_model:             RAW technique output (before gating's keep/
                                revert decision — always report on what the
                                technique actually produced, not on whichever
                                model the gate decided to keep).
        pre_accuracy:           Already-known accuracy of pre_model (from the
                                previous step's gate).
        post_accuracy:          Already-known accuracy of post_model. Callers
                                get this for free when the gate kept the
                                technique (its own measurement IS this
                                number); when reverted, callers must measure
                                post_model themselves once to report honestly
                                on what was discarded.
        was_kept:               The gate's keep/revert decision — used only
                                for the closing summary line, does not affect
                                any of the numeric calculations above it.
        pre_device/post_device: Device each model actually resides on. Kept
                                as two separate params (rather than one
                                shared device) because pre_model and
                                post_model can legitimately be on different
                                devices — e.g. dynamic INT8 quantization
                                moves post_model to CPU while pre_model
                                (fp32) stays on the original compute device.
                                Passing the wrong device here would silently
                                relocate a model via measure_latency's
                                internal .to(device) call — get this from
                                whatever the call site already computed for
                                its own gate call, never guess.
        original_accuracy/original_size_mb/original_latency_ms:
                                The ABSOLUTE original model's baseline
                                numbers, measured ONCE at the very start of
                                the run and threaded through unchanged to
                                every call — this is what makes "cumulative"
                                mean the same thing at every step.
        input_shape/input_dtype: Passed straight through to measure_latency.
        pre_size_mb/pre_latency_ms: Optional pre-computed values (mirrors
                                gate_technique_accuracy's pre_accuracy
                                parameter) — None measures fresh.
        pre_finetune_accuracy:  Optional snapshot of accuracy after the
                                structural change but before this technique's
                                OWN internal fine-tune. None = this technique
                                has no internal fine-tune to split out (GPTQ,
                                Quantization, the final KD step), or the
                                caller didn't request the snapshot (search
                                code never does — see Scope above).
        structural_summary:     Pre-formatted, technique-specific description
                                of what actually changed structurally (e.g.
                                "5/6 Conv2d layers pruned", "0/7 eligible
                                layers factorized"). Built by the caller from
                                that technique's own report dict, since each
                                technique's structural story is genuinely
                                different — this function stays technique-
                                agnostic and just prints whatever string it's
                                handed.
        structural_zero:        True when the caller has determined nothing
                                was structurally touched. Triggers the
                                explicit "consider disabling this technique"
                                recommendation.
        latency_iterations/latency_warmup: Forward passes for the reduced-
                                fidelity latency measurements above.

    Returns:
        Dict of every computed number (accuracy/size_mb/latency_ms, each
        with pre/post/marginal/cumulative — plus algorithm_delta_pp/
        finetune_delta_pp when pre_finetune_accuracy was supplied), for any
        future programmatic use (e.g. a results database).
    """
    # ── Size (always available — no forward passes, no device risk) ─────────
    if pre_size_mb is None:
        pre_size_mb = get_model_size_mb(pre_model)
    post_size_mb = get_model_size_mb(post_model)

    # ── Latency (reduced fidelity, defensive — never allowed to crash the
    #    real pipeline over a reporting feature) ──────────────────────────────
    def _safe_latency(m: nn.Module, dev: str) -> Optional[float]:
        try:
            return measure_latency(
                m, input_shape, dev,
                num_iterations=latency_iterations, warmup=latency_warmup,
                input_dtype=input_dtype,
            )['mean_ms']
        except Exception as exc:
            print(f"  [Impact] ⚠️  Latency measurement failed ({exc}) — reporting N/A.")
            return None

    if pre_latency_ms is None:
        pre_latency_ms = _safe_latency(pre_model, pre_device)
    post_latency_ms = _safe_latency(post_model, post_device)

    # ── Accuracy deltas (positive pp = dropped, matching gate_technique_
    #    accuracy's own "drop = pre - post" sign convention) ─────────────────
    marginal_pp   = pre_accuracy - post_accuracy
    cumulative_pp = original_accuracy - post_accuracy

    algorithm_delta_pp: Optional[float] = None
    finetune_delta_pp:  Optional[float] = None
    if pre_finetune_accuracy is not None:
        algorithm_delta_pp = pre_accuracy - pre_finetune_accuracy
        finetune_delta_pp  = pre_finetune_accuracy - post_accuracy

    # ── Size ratios (>1 = smaller/better — matches this codebase's existing
    #    baseline/compressed convention everywhere else: CQI, the final
    #    compression report, etc.) ─────────────────────────────────────────
    marginal_size_ratio   = (pre_size_mb / post_size_mb) if post_size_mb > 0 else None
    cumulative_size_ratio = (
        (original_size_mb / post_size_mb) if (original_size_mb and post_size_mb > 0) else None
    )

    # ── Latency speedups (>1 = faster, same convention) ──────────────────────
    marginal_speedup = (
        pre_latency_ms / post_latency_ms
        if (pre_latency_ms and post_latency_ms and post_latency_ms > 0) else None
    )
    cumulative_speedup = (
        original_latency_ms / post_latency_ms
        if (original_latency_ms and post_latency_ms and post_latency_ms > 0) else None
    )

    # ── Print ──────────────────────────────────────────────────────────────
    print(f"\n  [Impact] {technique_label}")
    if structural_summary:
        print(f"    Structural : {structural_summary}")

    if pre_finetune_accuracy is not None:
        _alg_dir = "improved by" if algorithm_delta_pp < 0 else "dropped"
        _ft_dir  = "improved by" if finetune_delta_pp  < 0 else "dropped"
        print(f"    Algorithm  : {pre_accuracy:.2f}% → {pre_finetune_accuracy:.2f}%  "
              f"({_alg_dir} {abs(algorithm_delta_pp):.2f}pp — raw structural effect, "
              f"before recovery fine-tune)")
        print(f"    Fine-tune  : {pre_finetune_accuracy:.2f}% → {post_accuracy:.2f}%  "
              f"({_ft_dir} {abs(finetune_delta_pp):.2f}pp — recovery fine-tune's own contribution)")

    _marg_dir = "improved by" if marginal_pp < 0 else "dropped"
    _cum_dir  = "improved by" if cumulative_pp < 0 else "dropped"
    print(f"    Accuracy   : {pre_accuracy:.2f}% → {post_accuracy:.2f}%   "
          f"marginal {_marg_dir} {abs(marginal_pp):.2f}pp   |   "
          f"cumulative {_cum_dir} {abs(cumulative_pp):.2f}pp vs. original")

    _size_marg_str = f"{marginal_size_ratio:.2f}×" if marginal_size_ratio is not None else "N/A"
    _size_cum_str  = f"{cumulative_size_ratio:.2f}×" if cumulative_size_ratio is not None else "N/A"
    print(f"    Size       : {pre_size_mb:.3f} MB → {post_size_mb:.3f} MB   "
          f"marginal {_size_marg_str}   |   cumulative {_size_cum_str} vs. original")

    _lat_pre_str  = f"{pre_latency_ms:.3f} ms" if pre_latency_ms is not None else "N/A"
    _lat_post_str = f"{post_latency_ms:.3f} ms" if post_latency_ms is not None else "N/A"
    _lat_marg_str = f"{marginal_speedup:.2f}×" if marginal_speedup is not None else "N/A"
    _lat_cum_str  = f"{cumulative_speedup:.2f}×" if cumulative_speedup is not None else "N/A"
    print(f"    Latency    : {_lat_pre_str} → {_lat_post_str}   "
          f"marginal {_lat_marg_str}   |   cumulative {_lat_cum_str} vs. original   "
          f"({latency_iterations} iter, reduced fidelity)")

    print(f"    Result     : {'kept' if was_kept else 'REVERTED by accuracy gate'}")

    if structural_zero:
        print(
            f"    ⚠️  Zero structural effect — {technique_label} did not change "
            f"a single weight this run. Any accuracy movement shown above is "
            f"entirely the recovery fine-tune, not {technique_label} itself. "
            f"Consider disabling this technique for this architecture."
        )

    return {
        'technique':  technique_label,
        'was_kept':   was_kept,
        'accuracy': {
            'pre': pre_accuracy, 'post': post_accuracy,
            'marginal_pp': marginal_pp, 'cumulative_pp': cumulative_pp,
            'pre_finetune': pre_finetune_accuracy,
            'algorithm_delta_pp': algorithm_delta_pp,
            'finetune_delta_pp':  finetune_delta_pp,
        },
        'size_mb': {
            'pre': pre_size_mb, 'post': post_size_mb,
            'marginal_ratio': marginal_size_ratio,
            'cumulative_ratio': cumulative_size_ratio,
        },
        'latency_ms': {
            'pre': pre_latency_ms, 'post': post_latency_ms,
            'marginal_speedup': marginal_speedup,
            'cumulative_speedup': cumulative_speedup,
        },
        'structural_zero': structural_zero,
    }


# ============================================================================
# MODEL TRAINING UTILITIES
# train_model and its private epoch helpers live here because they are
# >50 lines and are architecture-agnostic — pass any nn.Module + any DataLoader.
# setup_data() and load_or_train_model() own the dataset / architecture
# specifics; everything below is generic PyTorch training boilerplate.
# ============================================================================

def _one_train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    accuracy_fn,
    device: str,
) -> tuple:
    """
    Run one supervised training epoch.

    Args:
        model:       Model in train mode (set internally).
        loader:      DataLoader yielding (x, y) batches.
        optimizer:   Optimiser instance (stepped each batch).
        loss_fn:     Loss function.
        accuracy_fn: torchmetrics-style callable (pred, target) → scalar.
        device:      'cuda' or 'cpu'.

    Returns:
        (avg_loss, avg_acc) floats averaged over all batches.
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc  += accuracy_fn(logits.argmax(1), y).item()
    n = len(loader)
    return total_loss / n, total_acc / n


def _one_eval_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: str,
) -> tuple:
    """
    Run one evaluation epoch (no gradients).

    Args:
        model:       Model (set to eval mode internally).
        loader:      DataLoader yielding (x, y) batches.
        loss_fn:     Loss function.
        accuracy_fn: torchmetrics-style callable (pred, target) → scalar.
        device:      'cuda' or 'cpu'.

    Returns:
        (avg_loss, avg_acc) floats averaged over all batches.
    """
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            total_acc  += accuracy_fn(logits.argmax(1), y).item()
    n = len(loader)
    return total_loss / n, total_acc / n


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: str,
    save_path: Optional[str] = None,
) -> nn.Module:
    """
    Generic training loop for any classification model.

    Uses Adam + ReduceLROnPlateau (halves LR when validation accuracy stops
    improving for 2 epochs).  Saves the best checkpoint to save_path after
    every improvement if a path is given.

    Works with any architecture — just pass the model, loaders, and class
    count.  The model must already have its output head set to num_classes
    before being passed in.

    Args:
        model:        Any nn.Module with output dim = num_classes.
        train_loader: Training DataLoader.
        test_loader:  Validation/test DataLoader.
        num_classes:  Number of output classes (for torchmetrics accuracy).
        epochs:       Number of training epochs.
        lr:           Initial learning rate for Adam.
        device:       'cuda' or 'cpu'.
        save_path:    Optional .pth path; best checkpoint saved here.

    Returns:
        Trained model (same object, returned for chaining).
    """
    print(f"\n🏋️  Training for {epochs} epoch(s)  lr={lr}  classes={num_classes}  device={device}")
    model.to(device)

    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2,
    )
    loss_fn     = nn.CrossEntropyLoss()
    accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

    best_acc = 0.0
    for epoch in tqdm(range(epochs), desc="Training"):
        tr_loss, tr_acc = _one_train_epoch(
            model, train_loader, optimizer, loss_fn, accuracy_fn, device,
        )
        te_loss, te_acc = _one_eval_epoch(
            model, test_loader, loss_fn, accuracy_fn, device,
        )
        scheduler.step(te_acc)
        print(
            f"  Epoch {epoch + 1}/{epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc * 100:.2f}%  "
            f"test_acc={te_acc * 100:.2f}%"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            if save_path:
                os.makedirs(
                    os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                    exist_ok=True,
                )
                torch.save(model.state_dict(), save_path)
                print(f"    ✓ Checkpoint saved ({best_acc * 100:.2f}%)")

    print(f"\n✅ Training done. Best test accuracy: {best_acc * 100:.2f}%")
    return model



# ============================================================================
# DATA + MODEL LOADING
# setup_data() builds DataLoaders from Oxford Flowers-102.
# load_or_train_model() returns an EfficientNet-B0 fine-tuned on that dataset.
# Both live here so run_compression_pipeline() is fully self-contained and
# main.py owns only constants + main().
# ============================================================================

def setup_data(
    train_sample: Optional[int] = None,
    test_sample:  "Optional[int]" = 500,
    batch_size:   int             = 32,
    device:       str             = "cuda",
) -> tuple:
    """
    Load Oxford Flowers-102 and return (train_loader, test_loader).

    Dataset details:
      Flowers-102 has 102 flower categories photographed in the UK.
      Split sizes: train=1020, val=1020, test=6149.
      train+val are concatenated so the model sees 2040 images during training.
      Test split is used for evaluation only (subsettable via test_sample).

    Transform (ImageNet normalisation at 224x224):
      Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet stats)

    Args:
        train_sample: Max training images.  None = all 2040.
        test_sample:  Max test images.      None = all 6149.  Default 500.
        batch_size:   Batch size for both loaders.
        device:       Used only to set pin_memory=True when 'cuda'.

    Returns:
        (train_loader, test_loader)
    """
    from torchvision import datasets, transforms
    from torch.utils.data import ConcatDataset, DataLoader, Subset

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("\n📦 Loading Oxford Flowers-102...")
    train_ds = datasets.Flowers102(root="data", split="train", download=True, transform=transform)
    val_ds   = datasets.Flowers102(root="data", split="val",   download=True, transform=transform)
    test_ds  = datasets.Flowers102(root="data", split="test",  download=True, transform=transform)

    # Combine train + val for maximum training signal (2040 images total)
    train_full = ConcatDataset([train_ds, val_ds])
    test_full  = test_ds

    if train_sample is not None and train_sample > 0:
        train_full = Subset(train_full, list(range(min(train_sample, len(train_full)))))
    if test_sample is not None and test_sample > 0:
        test_full  = Subset(test_full,  list(range(min(test_sample,  len(test_full)))))

    print(f"  Train: {len(train_full)} samples | Test: {len(test_full)} samples")

    pin = device == "cuda"
    train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True,  pin_memory=pin, num_workers=2)
    test_loader  = DataLoader(test_full,  batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=2)
    return train_loader, test_loader


def load_or_train_model(
    train_loader,
    test_loader,
    num_classes: int,
    model_path:    str,
    epochs:        int,
    lr:            float,
    device:        str,
    force_retrain: bool = False,
) -> nn.Module:
    """
    Return an EfficientNet-B0 fine-tuned for num_classes output classes.

    If a checkpoint exists at model_path and force_retrain is False, loads
    it and reports current accuracy.  Otherwise fine-tunes from ImageNet
    weights using train_model() and saves the best checkpoint.

    EfficientNet-B0 architecture:
      The final classifier is nn.Sequential(Dropout, Linear(1280, 1000)).
      We replace classifier[1] with Linear(1280, num_classes) to adapt it
      to any target dataset.  All other weights stay as pretrained.

    Args:
        train_loader:  Training DataLoader.
        test_loader:   Validation/test DataLoader.
        num_classes:   Output class count.
        model_path:    Path to save/load the .pth checkpoint.
        epochs:        Training epochs (only used when training).
        lr:            Learning rate (only used when training).
        device:        'cuda' or 'cpu'.
        force_retrain: Retrain even if checkpoint exists.

    Returns:
        Fine-tuned EfficientNet-B0 as nn.Module.
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    if os.path.exists(model_path) and not force_retrain:
        print(f"\n✅ Loading checkpoint from '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        acc = measure_accuracy(model, test_loader, device)
        print(f"   Current test accuracy: {acc:.2f}%\n")
    else:
        reason = "force_retrain=True" if force_retrain else f"checkpoint not found at '{model_path}'"
        print(f"\n⚠️  {reason}. Training from scratch...")
        model = train_model(
            model, train_loader, test_loader,
            num_classes=num_classes,
            epochs=epochs, lr=lr, device=device, save_path=model_path,
        )

    return model


# ============================================================================
# REGISTRY-BASED DATA + MODEL LOADING
# These functions replace the hardcoded Flowers102 + EfficientNet-B0 pipeline
# with a model-agnostic system driven by model_registry.py.
#
# setup_data_for_model()      — returns (train_loader, test_loader) for any model
# load_or_train_from_registry() — loads checkpoint or fine-tunes any model
#
# Supported datasets:
#   flowers102  — Oxford Flowers-102 (existing)
#   cifar100    — CIFAR-100, 50k/10k, 100 classes, resized to model's input size
#   sst2        — Stanford Sentiment Treebank v2, binary, requires transformers+datasets
# ============================================================================

def _setup_cifar100_data(
    input_size: int = 224,
    train_sample: Optional[int] = None,
    test_sample:  "Optional[int]" = 500,
    batch_size:   int             = 32,
    device:       str             = "cuda",
) -> tuple:
    """
    Load CIFAR-100 and return (train_loader, test_loader).

    CIFAR-100 details:
      50 000 training images (32×32), 10 000 test images, 100 classes.
      Images are resized to input_size for pretrained models expecting 224×224.
      Uses ImageNet normalisation stats (standard for transfer learning).

    Args:
        input_size:   Spatial size for Resize (224 for most models, 299 for Inception).
        train_sample: Max training images.  None = all 50 000.
        test_sample:  Max test images.      None = all 10 000.
        batch_size:   Batch size.
        device:       Sets pin_memory when 'cuda'.

    Returns:
        (train_loader, test_loader)
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    resize_to = max(input_size + 32, 256)   # 256 for 224, 331 for 299

    transform_train = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n📦 Loading CIFAR-100  (resized to {input_size}×{input_size})...")
    train_ds = datasets.CIFAR100(root="data", train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)

    if train_sample is not None and train_sample > 0:
        train_ds = Subset(train_ds, list(range(min(train_sample, len(train_ds)))))
    if test_sample is not None and test_sample > 0:
        test_ds  = Subset(test_ds,  list(range(min(test_sample,  len(test_ds)))))

    print(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=pin, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=2)
    return train_loader, test_loader

def _setup_cifar10_data(
    input_size: int = 32,
    model_name: str = "cifar10",
    train_sample: Optional[int] = None,
    test_sample:  "Optional[int]" = 500,
    batch_size:   int             = 32,
    device:       str             = "cuda",
) -> tuple:
    """
    Load CIFAR-10 and return (train_loader, test_loader).

    CIFAR-10 details:
      50 000 training images (32×32), 10 000 test images, 10 classes.
      Classes: airplane, automobile, bird, cat, deer, dog, frog, horse,
               ship, truck.
      Native resolution is 32×32 — no resize needed for custom_cnn.
      Uses CIFAR-10 mean/std (computed from the training set) rather than
      ImageNet stats, because custom_cnn is trained from scratch, not
      fine-tuned from ImageNet weights.

    Train augmentation:
      RandomCrop(32, padding=4) + RandomHorizontalFlip are the two standard
      augmentations for CIFAR-10, improving generalisation by ~2-3%.

    Args:
        input_size:   Spatial size. Always 32 for custom_cnn.
        train_sample: Max training images.  None = all 50 000.
        test_sample:  Max test images.      None = all 10 000.  Default 500.
        batch_size:   Batch size.
        device:       Sets pin_memory when 'cuda'.

    Returns:
        (train_loader, test_loader)
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    # CIFAR-10 channel statistics (computed from training set)
    _MEAN = [0.4914, 0.4822, 0.4465]
    _STD  = [0.2470, 0.2435, 0.2616]

    transform_train = transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

    print(f"\n📦 Loading CIFAR-10  (native {input_size}×{input_size}, no resize)...")
    train_ds = datasets.CIFAR10(root="data", train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

    if train_sample is not None and train_sample > 0:
        train_ds = Subset(train_ds, list(range(min(train_sample, len(train_ds)))))
    if test_sample is not None and test_sample > 0:
        test_ds  = Subset(test_ds,  list(range(min(test_sample,  len(test_ds)))))

    print(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=pin, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=2)
    return train_loader, test_loader


def _setup_sst2_data(
    model_name: str,
    max_length:   int = 128,
    train_sample: Optional[int] = None,
    test_sample:  "Optional[int]" = 500,
    batch_size:   int             = 32,
    device:       str             = "cuda",
) -> tuple:
    """
    Load SST-2 (Stanford Sentiment Treebank) and return (train_loader, test_loader).

    SST-2 details:
      67 349 training sentences (binary sentiment: positive / negative).
      872 validation sentences (used as test set — no public test labels).
      Returns (input_ids, label) tensors padded to max_length.

    The DataLoader yields (input_ids [B, max_length] int64, labels [B] int64).
    This is identical in shape to any other (X, y) loader, so the compression
    pipeline's training loop and measure_accuracy work without modification.

    Requires: pip install transformers datasets

    Args:
        model_name:   Registry key — used to select the correct tokenizer.
        max_length:   Token sequence length (pad/truncate to this).
        train_sample: Max training sentences.  None = all 67 349.
        test_sample:  Max validation sentences.  None = all 872.
        batch_size:   Batch size.
        device:       Sets pin_memory when 'cuda'.

    Returns:
        (train_loader, test_loader)
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "SST-2 requires additional packages.\n"
            "Install with: pip install transformers datasets --break-system-packages"
        )

    # Map registry model names to HuggingFace tokenizer identifiers
    _TOKENIZER_MAP = {
        'bert_base':    'bert-base-uncased',
        'distilbert':   'distilbert-base-uncased',
        'roberta_base': 'roberta-base',
        'albert_base':  'albert-base-v2',
        'distilgpt2':   'distilgpt2',
    }
    tok_name = _TOKENIZER_MAP.get(model_name, 'bert-base-uncased')
    print(f"\n📦 Loading SST-2  (tokenizer: {tok_name}, max_length={max_length})...")

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:           # GPT-2 has no PAD → use EOS
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset('nyu-mll/glue', 'sst2')

    sentences_train = list(raw['train']['sentence'])
    labels_train    = list(raw['train']['label'])
    sentences_val   = list(raw['validation']['sentence'])
    labels_val      = list(raw['validation']['label'])

    def _tokenize(sentences):
        enc = tokenizer(sentences, max_length=max_length,
                        padding='max_length', truncation=True, return_tensors='pt')
        return enc['input_ids']

    class _SST2Dataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, labels):
            self.input_ids = input_ids
            self.labels    = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return self.input_ids[i], self.labels[i]

    train_ds = _SST2Dataset(_tokenize(sentences_train), labels_train)
    test_ds  = _SST2Dataset(_tokenize(sentences_val),   labels_val)

    if train_sample is not None and train_sample > 0:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(train_sample, len(train_ds)))))
    if test_sample is not None and test_sample > 0:
        from torch.utils.data import Subset
        test_ds  = Subset(test_ds,  list(range(min(test_sample,  len(test_ds)))))

    print(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    pin = device == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  pin_memory=pin, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=0)
    return train_loader, test_loader


def setup_data_for_model(
    model_name:   str,
    train_sample: Optional[int] = None,
    test_sample:  "Optional[int]" = 500,
    batch_size:   int             = 32,
    device:       str             = "cuda",
) -> tuple:
    """
    Return (train_loader, test_loader) for the dataset associated with model_name.

    Dispatches based on model_registry.SUPPORTED_MODELS[model_name]['dataset']:
      'flowers102' → setup_data()         (existing Flowers-102 loader)
      'cifar100'   → _setup_cifar100_data()
      'sst2'       → _setup_sst2_data()   (requires transformers + datasets)

    Args:
        model_name:   Key in model_registry.SUPPORTED_MODELS.
        train_sample: Max training samples.  None = full dataset.
        test_sample:  Max test samples.  None = full test set.
        batch_size:   DataLoader batch size.
        device:       'cuda' or 'cpu'.

    Returns:
        (train_loader, test_loader)
    """
    from sigularty.model_registry import get_model_meta
    meta    = get_model_meta(model_name)
    dataset = meta['dataset']

    if dataset == 'flowers102':
        return setup_data(
            train_sample=train_sample,
            test_sample=test_sample,
            batch_size=batch_size,
            device=device,
        )
    elif dataset == 'cifar100':
        # Use the model's spatial input size for resizing
        input_size = meta['input_shape'][-1]   # (1, 3, H, W) → W
        return _setup_cifar100_data(
            input_size=input_size,
            train_sample=train_sample,
            test_sample=test_sample,
            batch_size=batch_size,
            device=device,
        )
    elif dataset == "cifar10":
        return _setup_cifar10_data(
            model_name=model_name,
            train_sample=train_sample,
            test_sample=test_sample,
            batch_size=batch_size,
            device=device,
        )
    elif dataset == 'sst2':
        return _setup_sst2_data(
            model_name=model_name,
            train_sample=train_sample,
            test_sample=test_sample,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}' for model '{model_name}'. "
            f"Supported datasets: flowers102, cifar100, sst2."
        )


def load_or_train_from_registry(
    model_name:    str,
    train_loader,
    test_loader,
    model_path:    str,
    epochs:        int,
    lr:            float,
    device:        str,
    force_retrain: bool = False,
) -> nn.Module:
    """
    Load or fine-tune any model from the registry.

    If a checkpoint exists at model_path and force_retrain is False, loads
    the state dict and reports current test accuracy.  Otherwise fine-tunes
    from pretrained (ImageNet/HuggingFace Hub) weights and saves the best
    checkpoint to model_path.

    The checkpoint path includes the model name so different models never
    overwrite each other's checkpoints.

    Args:
        model_name:    Key in model_registry.SUPPORTED_MODELS.
        train_loader:  Training DataLoader.
        test_loader:   Evaluation DataLoader.
        model_path:    .pth file path.  The model name is injected into the
                       filename stem if it is not already present, so
                       'models/checkpoint.pth' becomes
                       'models/resnet50_checkpoint.pth' for resnet50.
        epochs:        Fine-tuning epochs (only used when training).
        lr:            Learning rate (only used when training).
        device:        'cuda' or 'cpu'.
        force_retrain: Retrain even if checkpoint exists.

    Returns:
        Fine-tuned nn.Module.
    """
    from sigularty.model_registry import load_model, get_model_meta

    meta        = get_model_meta(model_name)
    num_classes = meta['num_classes']

    # model_path already contains the model name, set by main.py
    # (e.g. 'models/resnet50.pth', 'models/efficientnet_b0_flowers102.pth').
    # No path manipulation needed here.
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

    # ── Load architecture from registry ──────────────────────────────────────
    model = load_model(model_name, num_classes=num_classes)

    if os.path.exists(model_path) and not force_retrain:
        print(f"\n✅ Loading checkpoint from '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        acc = measure_accuracy(model, test_loader, device)
        print(f"   Current test accuracy: {acc:.2f}%\n")
    else:
        reason = "force_retrain=True" if force_retrain else f"No checkpoint at '{model_path}'"
        print(f"\n⚠️  {reason}  — fine-tuning from pretrained weights...")
        model = train_model(
            model, train_loader, test_loader,
            num_classes=num_classes,
            epochs=epochs,
            lr=lr,
            device=device,
            save_path=model_path,
        )
        print(f"   ✓ Checkpoint saved → '{model_path}'")

    return model


# ============================================================================
# ONNX EXPORT AND RUNTIME
# Two functions are provided:
#
#   export_to_onnx  — converts any nn.Module to an ONNX file via torch.onnx.export.
#                     Uses a dummy forward pass to trace the compute graph.
#                     Verifies the exported graph with onnx.checker before returning.
#
#   run_onnx_inference — loads the exported file into an ONNX Runtime session
#                         and runs a real DataLoader through it, measuring
#                         top-1 accuracy and mean batch latency.
#
# Why ONNX?
#   ONNX is the standard interchange format for deploying PyTorch models to
#   production runtimes (TensorRT, OpenVINO, ONNX Runtime, CoreML, etc.).
#   Exporting after compression lets you verify the full end-to-end pipeline
#   produces a graph that inference engines can consume.
#
# Compatibility note:
#   Dynamic quantization (torch.qint8) uses custom ops that onnx.export cannot
#   always trace.  export_to_onnx guards for this and prints a clear message.
#   Static quantization and fp16 export cleanly.  Un-quantized and LRF+clustered
#   models always export successfully.
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    device: str = "cpu",
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """
    Export a PyTorch model to ONNX format and verify the graph.

    The model is traced using a single random dummy input of input_shape.
    After export, onnx.checker.check_model validates the graph structure.

    Dynamic quantization (qint8) uses PyTorch custom ops that the ONNX exporter
    cannot trace.  This function detects that case, warns the user, and raises
    a RuntimeError with a clear remediation message rather than silently
    producing a broken file.

    Args:
        model:          Any nn.Module.  Should be in eval() mode before calling.
                        The model is moved to `device` internally.
        save_path:      Destination file path, e.g. 'compressed_model.onnx'.
        input_shape:    Shape of a single input tensor (includes batch dim).
                        Default (1, 3, 224, 224) matches ResNet-50 on CIFAR-10.
        device:         Device for the dummy forward pass ('cpu' recommended for
                        ONNX export; CUDA exports require a CUDA-capable machine).
        opset_version:  ONNX opset.  17 is the latest stable opset supported by
                        most runtimes as of 2025.
        dynamic_batch:  If True, marks the batch dimension as dynamic so the
                        exported model accepts any batch size at inference time.

    Returns:
        Absolute path to the exported .onnx file.

    Raises:
        RuntimeError: If the model contains qint8 quantized ops that cannot be
                      exported, or if onnx.checker finds errors in the graph.
        ImportError:  If the 'onnx' package is not installed.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with: pip install onnx"
        )

    # Guard against dynamic-quantized models (qint8 custom ops are not ONNX-traceable)
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if 'Quantized' in cls_name or 'quantized' in cls_name.lower():
            raise RuntimeError(
                f"Module '{name}' ({cls_name}) contains dynamic INT8 quantized ops "
                f"that torch.onnx.export cannot trace.\n"
                f"Remediation: export the model BEFORE applying dynamic quantization, "
                f"or switch to fp16 / static quantization which export cleanly."
            )

    model.eval()
    model.to(device)

    dummy_input = torch.randn(input_shape, device=device)

    # Build dynamic axes dict if requested
    dynamic_axes = None
    if dynamic_batch:
        # input_shape[0] is the batch dimension
        dynamic_axes = {
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    print(f"\n[ONNX Export] Exporting to '{save_path}'  (opset {opset_version})...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,          # fold constants for smaller graph
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

    # Validate the exported graph
    print("[ONNX Export] Verifying graph with onnx.checker...")
    onnx_model = onnx.load(save_path)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as exc:
        raise RuntimeError(f"ONNX graph validation failed: {exc}") from exc

    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    print(f"[ONNX Export] ✅ Graph valid.  File size: {size_mb:.2f} MB")
    print(f"[ONNX Export] Saved → '{os.path.abspath(save_path)}'\n")
    return os.path.abspath(save_path)


def run_onnx_inference(
    onnx_path: str,
    dataloader,
    input_shape: tuple = (1, 3, 224, 224),
    num_latency_iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Load an ONNX model and run it through ONNX Runtime.

    Measures:
      - Top-1 classification accuracy on the full DataLoader.
      - Mean batch latency (ms) over num_latency_iterations dummy forward passes.

    ONNX Runtime is used instead of PyTorch here, which is the whole point:
    the exported graph runs on ORT's optimized C++ execution provider, not on
    any PyTorch kernel.  This is what your deployment target actually runs.

    Args:
        onnx_path:               Path to the .onnx file produced by export_to_onnx.
        dataloader:              DataLoader yielding (X, y) float32 CPU batches.
        input_shape:             Shape of a single dummy input for latency timing
                                 (includes batch dim, e.g. (1, 3, 224, 224)).
        num_latency_iterations:  Number of timed dummy forward passes.
        warmup:                  Untimed warmup passes before timing starts.

    Returns:
        Dict with keys:
            accuracy_pct      (float)  top-1 accuracy on the DataLoader
            mean_latency_ms   (float)  mean per-batch latency
            median_latency_ms (float)
            p95_latency_ms    (float)
            p99_latency_ms    (float)
            onnx_path         (str)    absolute path to the model file

    Raises:
        ImportError:  If 'onnxruntime' is not installed.
        FileNotFoundError: If onnx_path does not exist.
    """
    import time as _time

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "The 'onnxruntime' package is required to run ONNX models. "
            "Install it with: pip install onnxruntime"
        )

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at '{onnx_path}'")

    print(f"\n[ONNX Runtime] Loading session from '{onnx_path}'...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                   providers=['CPUExecutionProvider'])

    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[ONNX Runtime] Session ready.  Input: '{input_name}'  Output: '{output_name}'")

    # ── Accuracy on the DataLoader ───────────────────────────────────────────
    print("[ONNX Runtime] Measuring accuracy...")
    correct, total = 0, 0
    for X, y in dataloader:
        # ORT expects numpy float32; ensure correct dtype regardless of loader dtype
        x_np = X.numpy().astype("float32")
        outputs = session.run([output_name], {input_name: x_np})[0]  # (batch, num_classes)
        predictions = outputs.argmax(axis=1)
        correct += (predictions == y.numpy()).sum()
        total   += y.size(0)

    accuracy_pct = (correct / total) * 100 if total > 0 else 0.0
    print(f"[ONNX Runtime] Accuracy: {accuracy_pct:.2f}%  ({correct}/{total})")

    # ── Latency on dummy input ────────────────────────────────────────────────
    print(f"[ONNX Runtime] Measuring latency ({num_latency_iterations} iterations)...")
    dummy_np = torch.randn(input_shape).numpy().astype("float32")

    # Warmup
    for _ in range(warmup):
        session.run([output_name], {input_name: dummy_np})

    latencies = []
    for _ in range(num_latency_iterations):
        t0 = _time.perf_counter()
        session.run([output_name], {input_name: dummy_np})
        latencies.append((_time.perf_counter() - t0) * 1000)

    sorted_lat = sorted(latencies)
    n = len(latencies)
    mean_lat   = sum(latencies) / n
    median_lat = sorted_lat[n // 2]
    p95_lat    = sorted_lat[int(n * 0.95)]
    p99_lat    = sorted_lat[int(n * 0.99)]

    print(f"[ONNX Runtime] Latency — mean: {mean_lat:.3f} ms  "
          f"median: {median_lat:.3f} ms  p95: {p95_lat:.3f} ms")

    return {
        'accuracy_pct':      accuracy_pct,
        'mean_latency_ms':   mean_lat,
        'median_latency_ms': median_lat,
        'p95_latency_ms':    p95_lat,
        'p99_latency_ms':    p99_lat,
        'onnx_path':         os.path.abspath(onnx_path),
    }


# ============================================================================
# MAIN PIPELINE ORCHESTRATION
# The main() function body exceeds 50 lines so it lives here.
# main.py imports and calls run_compression_pipeline() directly.
# parse_args() is also >50 lines so it moves here too.
# ============================================================================


# ============================================================================
# ── AUTOMATIC LEARNING RATE FINDER ───────────────────────────────────────────
# ============================================================================

def find_best_lr(
    model: nn.Module,
    dataloader,
    *,
    device: str            = 'cuda',
    num_classes: int       = 10,
    start_lr: float        = 1e-7,
    end_lr: float          = 10.0,
    num_steps: int         = 100,
    smooth_window: int     = 5,
    diverge_threshold: float = 4.0,
) -> float:
    """
    Find the optimal learning rate using the LR range test (Smith 2017).

    Exponentially increases learning rate from start_lr to end_lr over
    num_steps mini-batches, recording loss at each step.  Returns the
    learning rate just before the loss stops decreasing — the "steepest
    descent" point.

    This solves the guessing game: run this once before training/fine-tuning,
    use the returned value (or 1/10 of it for Adam) as your learning rate.

    Rule of thumb for using the result:
        - SGD:  use the returned lr directly
        - Adam: use returned_lr / 10  (Adam is more aggressive per step)

    Parameters
    ----------
    model      : nn.Module — must already be on device.
    dataloader : DataLoader yielding (inputs, labels).
    device     : 'cuda' or 'cpu'.
    num_classes: For accuracy metric (not used in LR finder itself).
    start_lr   : Starting (lowest) learning rate.
    end_lr     : End (highest) learning rate.
    num_steps  : Number of steps (batches) to sweep.
    smooth_window   : Loss smoothing window (reduces noise).
    diverge_threshold: Stop early if loss > best_loss * this factor.

    Returns
    -------
    float — suggested learning rate (steepest descent point).

    Example
    -------
    >>> best_lr = find_best_lr(model, train_loader, device='cuda', num_classes=100)
    >>> print(f"Suggested LR: {best_lr:.2e}")
    >>> # Use best_lr for SGD, or best_lr/10 for Adam
    """
    import copy as _copy
    import math as _math

    print(f"\n[LR Finder] Sweeping lr from {start_lr:.1e} to {end_lr:.1e} "
          f"over {num_steps} steps...")

    # Work on a copy — don't modify the caller's model
    m = _copy.deepcopy(model).to(device)
    m.train()

    optimizer = torch.optim.SGD(m.parameters(), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    # Multiplicative factor per step
    factor = (end_lr / start_lr) ** (1.0 / num_steps)

    lrs:    list = []
    losses: list = []
    best_loss   = float('inf')
    avg_loss    = 0.0
    beta        = 0.98   # exponential smoothing for loss

    loader_iter = iter(dataloader)
    step = 0

    while step < num_steps:
        try:
            X, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            try:
                X, y = next(loader_iter)
            except StopIteration:
                break

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        try:
            out  = m(X)
            loss = criterion(out, y)
        except Exception:
            break

        loss_val = loss.item()
        if _math.isnan(loss_val) or _math.isinf(loss_val):
            break

        # Exponential smoothing
        avg_loss  = beta * avg_loss + (1 - beta) * loss_val
        smooth    = avg_loss / (1 - beta ** (step + 1))

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(smooth)

        if smooth < best_loss:
            best_loss = smooth
        elif smooth > diverge_threshold * best_loss:
            print(f"  [LR Finder] Loss diverged at lr={lrs[-1]:.2e} — stopping early.")
            break

        loss.backward()
        optimizer.step()

        # Increase lr for next step
        for pg in optimizer.param_groups:
            pg['lr'] *= factor

        step += 1

    del m

    if len(losses) < 3:
        print("  [LR Finder] Not enough data — returning default 1e-3")
        return 1e-3

    # Smooth losses with a sliding window
    if smooth_window > 1 and len(losses) > smooth_window:
        kernel = [1.0 / smooth_window] * smooth_window
        smoothed = []
        for i in range(len(losses)):
            start = max(0, i - smooth_window // 2)
            end   = min(len(losses), i + smooth_window // 2 + 1)
            smoothed.append(sum(losses[start:end]) / (end - start))
    else:
        smoothed = losses

    # Find steepest descent — largest negative gradient
    # Skip first and last 10% to avoid noise at boundaries
    trim = max(1, len(smoothed) // 10)
    grads = [smoothed[i] - smoothed[i-1] for i in range(1, len(smoothed))]
    search_grads = grads[trim:-trim] if len(grads) > 2 * trim else grads
    search_lrs   = lrs[1 + trim: 1 + trim + len(search_grads)]

    if not search_grads or not search_lrs:
        best_lr = lrs[len(lrs) // 2]
    else:
        steepest_idx = search_grads.index(min(search_grads))
        best_lr = search_lrs[steepest_idx]

    print(f"  [LR Finder] Suggested LR : {best_lr:.2e}")
    print(f"  [LR Finder] For SGD      : {best_lr:.2e}")
    print(f"  [LR Finder] For Adam     : {best_lr / 10:.2e}  (÷10 rule)")

    # Try to save a plot
    try:
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(8, 4))
        ax.plot(lrs[:len(smoothed)], smoothed, color='steelblue', linewidth=1.5)
        ax.axvline(x=best_lr, color='red', linestyle='--',
                   label=f'Suggested: {best_lr:.2e}')
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss (smoothed)')
        ax.set_title('LR Range Test — Pick the LR at the steepest descent')
        ax.legend()
        ax.grid(alpha=0.3)
        _plt.tight_layout()
        _plt.savefig('lr_finder.png', dpi=150, bbox_inches='tight')
        _plt.close(fig)
        print("  [LR Finder] Plot saved → 'lr_finder.png'")
    except Exception:
        pass

    return best_lr


def parse_args() -> "argparse.Namespace":
    """
    Build and parse CLI arguments for the compression toolkit.

    Every hyperparameter constant defined in main.py has a corresponding flag
    here so the full pipeline can be configured from the terminal without
    editing any source file.

    Returns:
        Parsed argparse.Namespace.
    """
    import argparse

    # These defaults mirror the constants in main.py; they are passed in as
    # arguments rather than read as globals so this function is self-contained.
    p = argparse.ArgumentParser(
        description="Model Compression Toolkit — generic model compression pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve',   # prevents duplicate-flag crash on Colab re-import
    )

    # ── Pipeline technique flags ──────────────────────────────────────────────
    p.add_argument('--no-low-rank',     dest='use_low_rank',     action='store_false',
                   help="Disable Low-Rank Factorization.")
    p.add_argument('--no-clustering',   dest='use_clustering',   action='store_false',
                   help="Disable Weight Clustering.")
    p.add_argument('--no-quantization', dest='use_quantization', action='store_false',
                   help="Disable Quantization.")
    p.set_defaults(use_low_rank=False, use_clustering=False, use_quantization=False)

    # ── Low-Rank Factorization ────────────────────────────────────────────────
    # ── Structured Pruning ───────────────────────────────────────────────────
    p.add_argument('--pruning',             dest='use_pruning', action='store_true', default=False,
                   help="Enable Structured Pruning (runs first in pipeline).")
    p.add_argument('--pruning-ratio',       type=float, default=0.3,
                   metavar='FLOAT', help="Global fraction of channels to prune (0.0–1.0).")
    p.add_argument('--pruning-model-type',  type=str, default='classifier',
                   choices=['classifier', 'embedding', 'generative', 'unknown'],
                   help="Architecture class — controls safety clamping.")
    p.add_argument('--pruning-fine-tune-epochs', type=int, default=3,
                   metavar='INT', help="Fine-tune epochs after pruning (0=skip).")
    p.add_argument('--pruning-fine-tune-lr',     type=float, default=0.0001,
                   metavar='FLOAT', help="Fine-tune LR for pruning recovery.")
    p.add_argument('--pruning-cal-batches',      type=int, default=50,
                   metavar='INT', help="Calibration batches for activation collection.")
    p.add_argument('--pruning-iterative-steps',  type=int, default=1,
                   metavar='INT', help="Torch-Pruning iterative steps (1=single-shot).")
    p.add_argument('--pruning-round-to',         type=int, default=None,
                   metavar='INT', help="Round pruned channels to this multiple (8/16 for Tensor Cores).")
    p.add_argument('--pruning-isomorphic',       action='store_true', default=False,
                   help="Force same pruning structure on all coupled groups.")
    p.add_argument('--pruning-max-ratio',        type=float, default=1.0,
                   metavar='FLOAT', help='Hard cap: no group pruned more than this fraction.')
    p.add_argument('--pruning-report-path', type=str, default='pruning_report.png',
                   metavar='PATH', help="Output path for the pruning report PNG.")

    # ── Low-Rank Factorization ────────────────────────────────────────────────
    p.add_argument('--lrf-epsilon',        type=float, default=0.5,
                   metavar='FLOAT', help="LRF rank ratio (0.0, 1.0].")
    p.add_argument('--lrf-min-layer-size', type=int,   default=64,
                   metavar='INT',   help="Skip layers with dim <= this.")
    p.add_argument('--lrf-min-rank',        type=int,   default=2,
                   metavar='INT',   help="Skip LRF layers where rank < this (prevents rank-1 collapse).")
    p.add_argument('--lrf-skip-large-kernels', dest='lrf_skip_large_kernels',
                   action='store_true', default=False,
                   help="Skip Conv2d with kernel>1×1 during LRF. "
                        "Use for ResNet/VGG/DenseNet to prevent latency regression.")
    p.add_argument('--lrf-fine-tune-epochs', type=int, default=0,
                   metavar='INT', help="KD recovery fine-tune epochs after LRF "
                        "(0=skip, default). Always knowledge distillation "
                        "against the original model — never plain cross-entropy.")
    p.add_argument('--lrf-fine-tune-lr',     type=float, default=0.0001,
                   metavar='FLOAT', help="Learning rate for the LRF recovery fine-tune.")

    # ── Weight Clustering ─────────────────────────────────────────────────────
    p.add_argument('--num-clusters',             type=int,   default=16,
                   metavar='INT',   help="k-means k.")
    p.add_argument('--cluster-fine-tune-epochs', type=int,   default=5,
                   metavar='INT',   help="Fine-tune epochs after clustering (0=skip).")
    p.add_argument('--cluster-fine-tune-lr',     type=float, default=0.0001,
                   metavar='FLOAT', help="Fine-tune learning rate.")

    # ── Quantization ──────────────────────────────────────────────────────────
    p.add_argument('--quant-mode',        type=str, default='dynamic',
                   choices=['dynamic', 'static', 'fp16'],
                   help="Quantization mode.")
    p.add_argument('--quant-cal-batches', type=int, default=100,
                   metavar='INT', help="Calibration batches for static mode.")

    # ── Data / training ───────────────────────────────────────────────────────
    p.add_argument('--train-samples',    type=int,   default=1000,
                   metavar='INT',   help="Training samples (0 = all 50 000).")
    p.add_argument('--test-samples',     type=int,   default=500,
                   metavar='INT',   help="Test samples (0 = all 10 000).")
    p.add_argument('--batch-size',       type=int,   default=32,
                   metavar='INT',   help="DataLoader batch size.")
    p.add_argument('--pretrain-epochs',  type=int,   default=10,
                   metavar='INT',   help="Pre-training epochs.")
    p.add_argument('--pretrain-lr',      type=float, default=0.001,
                   metavar='FLOAT', help="Pre-training learning rate.")
    p.add_argument('--force-retrain',    action='store_true', default=False,
                   help="Re-train even if a checkpoint exists.")

    # ── ONNX export ───────────────────────────────────────────────────────────
    p.add_argument('--export-onnx',      action='store_true', default=False,
                   help="Export the compressed model to ONNX after the pipeline.")
    p.add_argument('--onnx-path',        type=str, default='compressed_model.onnx',
                   metavar='PATH', help="Output path for the ONNX file.")
    p.add_argument('--onnx-opset',       type=int, default=17,
                   metavar='INT',  help="ONNX opset version.")
    p.add_argument('--run-onnx',         action='store_true', default=False,
                   help="Run the exported ONNX model through ONNX Runtime.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument('--active-model', dest='active_model', type=str, default=None,
                   metavar='NAME',
                   help="Model to compress. Overrides ACTIVE_MODEL in main.py. "
                        "Run --list-models to see all options.")
    p.add_argument('--list-models',  dest='list_models', action='store_true', default=False,
                   help="Print all 20 available models and exit.")
    p.add_argument('--device',       type=str,  default='cuda' if torch.cuda.is_available() else 'cpu',
                   choices=['cpu', 'cuda'], help="Compute device.")
    p.add_argument('--report-path',  type=str,  default='compression_report.png',
                   metavar='PATH',  help="Output path for the PNG report.")
    p.add_argument('--model-path',   type=str,  default='models/efficientnet_b0_flowers102.pth',
                   metavar='PATH',  help="Path to save / load model checkpoint.")
    p.add_argument('--find-optimal-epsilon', action='store_true', default=False,
                   help="Run anchor+ternary epsilon search before the main pipeline.")
    p.add_argument('--epsilon-search-num-trials', dest='epsilon_search_num_trials',
                   type=int, default=15, metavar='INT',
                   help="Total evaluation budget for epsilon search.")
    p.add_argument('--find-optimal-pruning', dest='find_optimal_pruning',
                   action='store_true', default=False,
                   help="Run pruning hyperparameter search before the main pipeline.")
    p.add_argument('--pruning-search-num-trials', dest='pruning_search_num_trials',
                   type=int, default=16, metavar='INT',
                   help="Total evaluation budget for pruning search.")
    p.add_argument('--fine-tune-abort-threshold', dest='fine_tune_abort_threshold',
                   type=float, default=30.0, metavar='PP',
                   help="Abort a Pruning/LRF recovery fine-tune after epoch 1 if "
                        "the accuracy drop vs. the original model already exceeds "
                        "this many percentage points. Direct value, not a multiplier.")

    args, _ = p.parse_known_args()   # parse_known_args ignores Colab kernel args
    return args


def run_compression_pipeline(args: argparse.Namespace) -> dict:
    """
    Full compression toolkit pipeline, driven by a parsed args namespace.

    Steps:
      1. Load CIFAR-10 / registry dataset.
      2. Load or fine-tune the active model.
      2a. nn.MultiheadAttention safety check (auto-disables LRF + epsilon
          search for incompatible architectures — see
          model_has_multihead_attention()'s docstring).
      2b. Measure baseline accuracy ONCE — reused for every gating decision,
          the final report, and the post-KD recovery check, instead of
          re-measuring the same number 3-4 separate times.
      3. (Optional) pruning hyperparameter search.
      4. (Optional) epsilon grid-search.
      5. Apply structural compression (BN Fusion, Pruning, LRF, Clustering)
         via apply_compression_pipeline, with per-technique accuracy gating
         enabled (test_loader + the global accuracy_drop_threshold).
      6. GPTQ (gated) → standard Quantization (gated) → final KD recovery
         fine-tune (gated, plus the 6d cumulative-recovery retry check).
      7. Evaluate accuracy and latency.
      8. (Optional) Export to ONNX and run through ONNX Runtime.
      9. Generate compression report PNG.

    Per-technique accuracy gating (global, applies to every technique):
      Every technique that can mutate the model — Pruning, LRF, Clustering,
      GPTQ, Quantization, and the final KD step — is measured before and
      after it runs.  If a technique's OWN marginal accuracy drop exceeds
      args.accuracy_drop_threshold, that technique's result is discarded and
      the pipeline reverts to the pre-technique state.  Each technique has an
      independent budget; an earlier costly technique does not eat into a
      later technique's allowance.  Pruning/LRF/Clustering are gated inside
      apply_compression_pipeline (compression.py); GPTQ/Quantization/the
      final KD step are gated here, since they run outside that function
      (the two-phase float32-then-quantize design described in README.md).

      A technique that gets reverted is NOT listed in the final report's
      "techniques used" — the report only ever describes what the returned
      model actually contains.

    Args:
        args: Namespace returned by parse_args() (or constructed manually for
              programmatic use — see main.py for the full field list).

    Returns:
        Final metrics dict as returned by generate_compression_report().
    """
    # These imports are deferred to keep helper_functions.py importable without
    # the full compression / visualization stack installed.
    from sigularty.compression import apply_compression_pipeline
    from sigularty.visualization import generate_compression_report, plot_epsilon_landscape
    from sigularty.optimization import find_optimal_epsilon_smart, find_optimal_pruning_params

    # ── Device override — always re-detect so Colab argv can't corrupt it ──────
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # fp16 only makes sense on GPU; fall back to dynamic INT8 if no GPU
    if getattr(args, 'quant_mode', 'fp16') == 'fp16' and args.device == 'cpu':
        print("  ⚠️  fp16 requested but no GPU available — switching to dynamic INT8.")
        args.quant_mode = 'dynamic'

    # ── Global accuracy-drop threshold — single source of truth for EVERY
    #    gating decision below (both hyperparameter searches AND the
    #    pipeline's per-technique gate use this exact same value).
    _threshold = getattr(args, 'accuracy_drop_threshold', 10.0)

    # ── 0. Banner ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL COMPRESSION TOOLKIT")
    print("=" * 70)
    print(f"  Active Model   : {getattr(args,'active_model','efficientnet_b0')}")
    print(f"  Device         : {args.device}")
    print(f"  Pruning        : {getattr(args,'use_pruning',False)}"
          + (f"  ratio={getattr(args,'pruning_ratio',0.3)}  type={getattr(args,'pruning_model_type','classifier')}"
             f"  max={getattr(args,'pruning_max_ratio',1.0)}"
             if getattr(args,'use_pruning',False) else ""))
    print(f"  Low-Rank (LRF) : {args.use_low_rank}  ε={args.lrf_epsilon}")
    print(f"  Clustering     : {args.use_clustering}  k={args.num_clusters}")
    print(f"  Quantization   : {args.use_quantization}  mode={args.quant_mode}")
    print(f"  Accuracy gate  : {_threshold:.1f}pp per technique (global, "
          f"independent budget per technique)")
    print(f"  Export ONNX    : {args.export_onnx}"
          + (f"  → '{args.onnx_path}'" if args.export_onnx else ""))
    print("=" * 70)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    train_sample  = args.train_samples if args.train_samples > 0 else None
    test_sample   = args.test_samples  if args.test_samples  > 0 else None
    active_model  = getattr(args, 'active_model', 'efficientnet_b0')

    # If a model from the registry is selected, use its dataset and input shape.
    # Fall back to the legacy Flowers102 + EfficientNet flow when active_model is
    # not in the registry (preserves full backward compatibility).
    _use_registry = False
    try:
        from sigularty.model_registry import get_model_meta, SUPPORTED_MODELS
        if active_model in SUPPORTED_MODELS:
            _use_registry = True
            _meta = get_model_meta(active_model)
            args.num_classes  = _meta['num_classes']
            args.input_shape  = _meta['input_shape']
            args.input_dtype  = _meta.get('input_dtype', None)
    except ImportError:
        pass

    if _use_registry:
        train_loader, test_loader = setup_data_for_model(
            model_name=active_model,
            train_sample=train_sample,
            test_sample=test_sample,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        train_loader, test_loader = setup_data(
            train_sample=train_sample,
            test_sample=test_sample,
            batch_size=args.batch_size,
            device=args.device,
        )

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: LOAD / TRAIN MODEL")
    print("=" * 70)
    if _use_registry:
        model = load_or_train_from_registry(
            model_name=active_model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_path=args.model_path,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            device=args.device,
            force_retrain=args.force_retrain,
        )
    else:
        model = load_or_train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=getattr(args, 'num_classes', 102),
            model_path=args.model_path,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            device=args.device,
            force_retrain=args.force_retrain,
        )
    model.eval()

    # ── 2a. Architecture safety check — nn.MultiheadAttention vs LRF ──────────
    # LRF's _decompose_linear_layer wraps the target Linear in nn.Sequential.
    # nn.MultiheadAttention.forward() calls F.multi_head_attention_forward()
    # with self.out_proj.weight / .bias accessed DIRECTLY as raw tensors — it
    # never calls out_proj.forward(), so wrapping out_proj in nn.Sequential
    # breaks every forward pass with AttributeError: 'Sequential' object has
    # no attribute 'weight'.  compression.py's LRF functions already refuse
    # to touch such models defensively (return an unchanged deepcopy), but
    # disabling LRF here ALSO skips the epsilon search entirely, avoiding 15+
    # wasted search trials that would otherwise each deepcopy, no-op, and
    # report unchanged baseline numbers as if nothing happened.
    _mha_count = model_has_multihead_attention(model)
    if _mha_count > 0 and (getattr(args, 'use_low_rank', False) or getattr(args, 'find_optimal_epsilon', False)):
        print(f"\n  ⚠️  [LRF Safety] Detected {_mha_count} nn.MultiheadAttention "
              f"module(s) in '{active_model}'. LRF cannot safely factorize "
              f"out_proj (nn.MultiheadAttention.forward() bypasses out_proj's "
              f"own forward() and accesses .weight/.bias directly — wrapping "
              f"it in nn.Sequential breaks every forward pass). Disabling "
              f"use_low_rank and find_optimal_epsilon for this run.")
        args.use_low_rank = False
        args.find_optimal_epsilon = False

    # ── CQI weights — read from args (set in main.py constants) ─────────────
    _cqi_w = dict(
        w_accuracy = getattr(args, 'cqi_w_accuracy', 1.0),
        w_size     = getattr(args, 'cqi_w_size',     1.0),
        w_latency  = getattr(args, 'cqi_w_latency',  1.0),
        w_kl       = getattr(args, 'cqi_w_kl',       1.0),
    )

    # ── 2b. Baseline accuracy — measured ONCE, reused everywhere ─────────────
    # (gating's starting point, the final report's "original accuracy", and
    # the 6d post-KD cumulative-recovery check).  Previously this exact
    # number was independently re-measured up to 3 separate times across the
    # pipeline (once for each search, once again in Step 6 Evaluation).
    print("\n  Measuring baseline (original model) accuracy once for reuse "
          "throughout this run...")
    orig_accuracy = measure_accuracy(model, test_loader, args.device)
    print(f"  Baseline accuracy: {orig_accuracy:.2f}%")

    # ── Baseline size/latency — measured once here too, alongside accuracy,
    #    so every per-technique impact report below (report_technique_impact,
    #    gate_technique_accuracy's companion) can compare against the SAME
    #    true-original numbers rather than re-measuring repeatedly. This is
    #    what makes "cumulative vs. original" mean the same original at
    #    every step in the pipeline, not a moving target. Latency uses
    #    reduced fidelity (10 iterations, 3 warmup — matching optimization.py
    #    search functions' own convention for non-final measurements) since
    #    this is on top of what Step 6's final report already pays for at
    #    full fidelity (100/10) later.
    orig_size_mb = get_model_size_mb(model)
    _impact_input_shape = getattr(args, 'input_shape', (1, 3, 224, 224))
    _impact_input_dtype = getattr(args, 'input_dtype', None)
    orig_latency_ms = measure_latency(
        model, _impact_input_shape, args.device,
        num_iterations=10, warmup=3, input_dtype=_impact_input_dtype,
    )['mean_ms']
    print(f"  Baseline size: {orig_size_mb:.3f} MB   "
          f"Baseline latency: {orig_latency_ms:.3f} ms  "
          f"(10 iter, reduced fidelity — reused for per-technique impact reporting)\n")

    # ── 2c. Pruning hyperparameter search (optional) ──────────────────────────
    # Gated on use_pruning: if pruning is disabled for this run, never spend
    # 20 trials searching for parameters that won't be used.
    if getattr(args, 'find_optimal_pruning', False) and getattr(args, 'use_pruning', True):
        print("\n" + "=" * 70)
        print("STEP 2 (OPTIONAL): PRUNING HYPERPARAMETER SEARCH")
        print("=" * 70)
    elif getattr(args, 'find_optimal_pruning', False) and not getattr(args, 'use_pruning', True):
        print("\n  [prune-search] Skipped — use_pruning=False for this run.")
    if getattr(args, 'find_optimal_pruning', False) and getattr(args, 'use_pruning', True):
        _baseline_size = get_model_size_mb(model)
        _baseline_lat_ms = None
        optimal_prune_params, _prune_results, _prune_history = find_optimal_pruning_params(
            model=model,
            dataloader=train_loader,
            test_loader=test_loader,
            device=args.device,
            baseline_accuracy=orig_accuracy,
            baseline_size=_baseline_size,
            baseline_latency_ms=_baseline_lat_ms,
            num_trials=getattr(args, 'pruning_search_num_trials', 16),
            cache_path='pruning_search_cache.json',
            model_type=getattr(args, 'pruning_model_type', 'classifier'),
            num_classes=getattr(args, 'num_classes', 102),
            num_calibration_batches=getattr(args, 'pruning_cal_batches', 50),
            max_pruning_ratio=getattr(args, 'pruning_max_ratio', 1.0),
            residual_max_ratio=getattr(args, 'pruning_residual_max_ratio', None),
            round_to=getattr(args, 'pruning_round_to', None),
            isomorphic=getattr(args, 'pruning_isomorphic', False),
            # BUGFIX: previously always defaulted to the vision shape
            # (1,3,224,224) with no dtype override, since neither was passed
            # here. For NLP registry models (input_ids as int64 token IDs of
            # shape (1,128)) that silently crashed the baseline latency
            # measurement inside a 4D-vs-2D shape mismatch deep in the HF
            # forward pass — caught by a bare except and swallowed as
            # "latency measurement failed, falling back to 1.0ms". Passing
            # the real registry-derived shape/dtype (set earlier in this
            # function from model_registry metadata) fixes that for good.
            input_shape=getattr(args, 'input_shape', (1, 3, 224, 224)),
            input_dtype=getattr(args, 'input_dtype', None),
            search_ft_epochs=getattr(args, 'pruning_search_ft_epochs', 1),
            search_ft_lr=getattr(args, 'pruning_search_ft_lr', 1e-4),
            # BUGFIX: this call previously omitted accuracy_drop_threshold
            # entirely, silently falling back to the function's own default
            # of 5.0pp instead of your configured global threshold.
            accuracy_drop_threshold=_threshold,
            # NEW: wires FINE_TUNE_ABORT_THRESHOLD through to every per-trial
            # recovery fine-tune this search runs (see _kd_recovery_fine_tune
            # in compression.py for the actual abort check).
            early_abort_threshold=getattr(args, 'fine_tune_abort_threshold', 30.0),
            **_cqi_w,
        )
        if optimal_prune_params is None:
            print("  ❌  Pruning search: no viable config. Disabling pruning.")
            args.use_pruning = False
        else:
            args.pruning_ratio         = optimal_prune_params.get('pruning_ratio',     args.pruning_ratio)
            args.pruning_max_ratio     = optimal_prune_params.get('pruning_max_ratio', args.pruning_max_ratio)
            args.pruning_iterative_steps = optimal_prune_params.get('iterative_steps', getattr(args, 'pruning_iterative_steps', 1))
            print(f"  → Optimal pruning params: {optimal_prune_params}")

    # ── 3. Epsilon grid-search (optional) ────────────────────────────────────
    if args.find_optimal_epsilon and getattr(args, 'use_low_rank', True):
        print("\n" + "=" * 70)
        print("STEP 3 (OPTIONAL): EPSILON GRID-SEARCH")
        print("=" * 70)
    elif args.find_optimal_epsilon and not getattr(args, 'use_low_rank', True):
        print("\n  [ε-search] Skipped — use_low_rank=False for this model "
              "(registry recommendation or nn.MultiheadAttention safety check).")
    if args.find_optimal_epsilon and getattr(args, 'use_low_rank', True):
        _baseline_size = get_model_size_mb(model)
        _baseline_lat_ms = None
        optimal_eps, _search_results, _search_history = find_optimal_epsilon_smart(
            model=model,
            test_loader=test_loader,
            device=args.device,
            baseline_accuracy=orig_accuracy,
            baseline_size=_baseline_size,
            baseline_latency_ms=_baseline_lat_ms,
            min_layer_size=args.lrf_min_layer_size,
            min_rank=getattr(args, 'lrf_min_rank', 2),
            skip_large_kernels=getattr(args, 'lrf_skip_large_kernels', False),
            cache_path='epsilon_cache.json',
            num_trials=getattr(args, 'epsilon_search_num_trials', 15),
            # Same fix as the pruning search above — real registry shape/dtype
            # instead of the vision-only (1,3,224,224) default.
            input_shape=getattr(args, 'input_shape', (1, 3, 224, 224)),
            input_dtype=getattr(args, 'input_dtype', None),
            # dataloader/num_classes are what actually let EPSILON_SEARCH_FT_EPOCHS
            # do anything — apply_low_rank_factorization only fine-tunes when
            # all three of dataloader/device/num_classes are supplied. Without
            # this, search_ft_epochs>0 in main.py was silently a no-op here.
            dataloader=train_loader,
            num_classes=getattr(args, 'num_classes', 102),
            search_ft_epochs=getattr(args, 'epsilon_search_ft_epochs', 0),
            search_ft_lr=getattr(args, 'epsilon_search_ft_lr', 0.0001),
            accuracy_drop_threshold=_threshold,
            # NEW: same early-abort wiring as the pruning search. Note this
            # can't help much here specifically since EPSILON_SEARCH_FT_EPOCHS
            # is 1 by default — there's no "remaining epochs" to skip after
            # the only epoch finishes. It's here for consistency and for
            # anyone who raises EPSILON_SEARCH_FT_EPOCHS later.
            early_abort_threshold=getattr(args, 'fine_tune_abort_threshold', 30.0),
            **{k: v for k, v in _cqi_w.items() if k != 'w_kl'},  # no KL for LRF
        )
        plot_epsilon_landscape(
            results=_search_results,
            search_history=_search_history,
            optimal_epsilon=optimal_eps,
            baseline_accuracy=orig_accuracy,
            baseline_size=_baseline_size,
            save_path=getattr(args, 'epsilon_landscape_path', 'epsilon_landscape.png'),
        )
        if optimal_eps is None:
            print("  ❌  LRF search: score<1.0 or drop>threshold. Disabling LRF.")
            args.use_low_rank = False
        else:
            args.lrf_epsilon = optimal_eps
            print(f"  → Using optimal ε={optimal_eps:.2f} for main pipeline run.")

    # ── 4. Auto-switch static → dynamic ─────────────────────────────────────
    if args.use_quantization and args.quant_mode == 'static':
        print("  ⚠️  static quant: residual additions have no QuantizedCPU kernel.")
        print("      Auto-switching to dynamic INT8.\n")
        args.quant_mode = 'dynamic'

    # ── 5. Compression pipeline — two phases ─────────────────────────────────
    #
    # WHY TWO PHASES?
    #   dynamic INT8 (qint8) has no CUDA kernel — it runs on CPU only.
    #   Measuring original model on GPU (7 ms) and compressed on CPU (113 ms)
    #   is a meaningless comparison.
    #
    #   Phase A: BN Fusion + Pruning + LRF + Clustering → float32 → stays on
    #            GPU → latency here.  Per-technique accuracy gating is active
    #            throughout this phase (test_loader + _threshold passed in).
    #   Phase B: GPTQ → Quantization (fp16 or INT8) → final KD recovery, all
    #            gated individually here in run_compression_pipeline since
    #            they run outside apply_compression_pipeline.
    #
    #   fp16 stays on GPU (Tensor Cores), dynamic INT8 moves to CPU.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: STRUCTURAL COMPRESSION  (BN Fusion · Pruning · LRF · Clustering)")
    print("=" * 70)

    # Phase A: All structural compression — float32, GPU-capable, individually
    # gated (Pruning / LRF / Clustering each revert on their own marginal
    # accuracy drop, independent budgets).  Quantization is deferred to
    # Phase B so latency can be measured in float32.
    compressed_pre_quant = apply_compression_pipeline(
        model=model,
        dataloader=train_loader,
        device=args.device,
        # ── Tier 1 ──────────────────────────────────────────────────────────
        use_bn_fusion=getattr(args, 'use_bn_fusion', True),
        # ── Tier 2A ─────────────────────────────────────────────────────────
        use_sensitivity=getattr(args, 'use_sensitivity', False),
        sensitivity_batches=getattr(args, 'sensitivity_batches', 10),
        # ── Pruning ─────────────────────────────────────────────────────────
        use_pruning=getattr(args, 'use_pruning', False),
        pruning_ratio=getattr(args, 'pruning_ratio', 0.3),
        pruning_model_type=getattr(args, 'pruning_model_type', 'classifier'),
        pruning_num_classes=getattr(args, 'num_classes', 102),
        pruning_fine_tune_epochs=getattr(args, 'pruning_fine_tune_epochs', 3),
        pruning_fine_tune_lr=getattr(args, 'pruning_fine_tune_lr', 0.0001),
        pruning_cal_batches=getattr(args, 'pruning_cal_batches', 50),
        pruning_iterative_steps=getattr(args, 'pruning_iterative_steps', 1),
        pruning_round_to=getattr(args, 'pruning_round_to', None),
        pruning_isomorphic=getattr(args, 'pruning_isomorphic', False),
        pruning_max_ratio=getattr(args, 'pruning_max_ratio', 1.0),
        pruning_residual_max_ratio=getattr(args, 'pruning_residual_max_ratio', None),
        # ── LRF (adaptive or standard) ───────────────────────────────────────
        use_low_rank=args.use_low_rank,
        lrf_epsilon=args.lrf_epsilon,
        lrf_min_layer_size=args.lrf_min_layer_size,
        lrf_min_rank=getattr(args, 'lrf_min_rank', 2),
        lrf_skip_large_kernels=getattr(args, 'lrf_skip_large_kernels', False),
        lrf_adaptive=getattr(args, 'lrf_adaptive', False),
        lrf_energy_threshold=getattr(args, 'lrf_energy_threshold', 0.99),
        lrf_fine_tune_epochs=getattr(args, 'lrf_fine_tune_epochs', 0),
        lrf_fine_tune_lr=getattr(args, 'lrf_fine_tune_lr', 0.0001),
        lrf_num_classes=getattr(args, 'num_classes', 102),
        # ── Clustering ──────────────────────────────────────────────────────
        use_clustering=args.use_clustering,
        cluster_num_clusters=args.num_clusters,
        cluster_fine_tune_epochs=args.cluster_fine_tune_epochs,
        cluster_fine_tune_lr=args.cluster_fine_tune_lr,
        cluster_num_classes=getattr(args, 'num_classes', 102),
        cluster_layers=None,
        # ── Tier 2B: KD fine-tune — deferred to after quantization ──────────
        # KD is passed to clustering for its post-k-means recovery step,
        # but the standalone KD pipeline step is disabled here and runs
        # after GPTQ so it recovers accuracy from all quantization losses.
        use_kd_finetune=False,
        kd_teacher=model,  # original model is teacher for clustering KD (and LRF KD)
        kd_epochs=getattr(args, 'kd_epochs', 3),
        kd_lr=getattr(args, 'kd_lr', 0.0001),
        kd_temperature=getattr(args, 'kd_temperature', 4.0),
        kd_alpha=getattr(args, 'kd_alpha', 0.7),
        kd_num_classes=getattr(args, 'num_classes', 102),
        # ── Quantization deferred ────────────────────────────────────────────
        use_quantization=False,
        # ── Tier 3B: GPTQ (also deferred — runs in Phase B) ─────────────────
        use_gptq=False,
        # ── Per-technique accuracy gating ────────────────────────────────────
        test_loader=test_loader,
        accuracy_drop_threshold=_threshold,
        # NEW: same early-abort protection as the searches, now applied to
        # Pruning's and LRF's ACTUAL recovery fine-tunes in the real pipeline
        # (not just their hyperparameter searches) — matches what main.py's
        # own docs for FINE_TUNE_ABORT_THRESHOLD always said this should cover.
        early_abort_threshold=getattr(args, 'fine_tune_abort_threshold', 30.0),
        # ── Per-technique impact reporting ───────────────────────────────────
        # Absolute-original baseline numbers, measured once above — passed
        # through so every impact report inside apply_compression_pipeline
        # computes "cumulative vs. original" against the same reference the
        # rest of this function (Phase B below) uses.
        original_size_mb=orig_size_mb,
        original_latency_ms=orig_latency_ms,
        input_shape=_impact_input_shape,
        input_dtype=_impact_input_dtype,
    )

    # ── Build techniques_used from the gating report attached by
    #    apply_compression_pipeline — only list techniques that actually
    #    survived their accuracy gate, not ones that were attempted and
    #    reverted.  Falls back to "assume kept" (old behaviour) if gating
    #    metadata is absent for some reason.
    _gating_report = getattr(compressed_pre_quant, '_gating_report', {})
    # Computed once, used both for the techniques_used list below AND to
    # gate whether plot_pruning_report even attempts to render anything
    # further down. True only when apply_structured_pruning caught a
    # Torch-Pruning graph-construction failure it cannot recover from (see
    # _torch_pruning_incompatible_result in compression.py) and returned
    # the model unchanged -- treated identically to use_pruning having been
    # False for this run, never as a technique that "ran and did nothing."
    # _gating_report.get('Structured Pruning', True) alone is NOT enough to
    # exclude this case: that dict defaults to True when the key is simply
    # absent (the correct default when gating itself was never turned on),
    # and the incompatibility path deliberately never adds a gating_report
    # entry at all -- so without this separate positive check, the default
    # would silently flip back to "included."
    _pruning_report_final = getattr(compressed_pre_quant, '_pruning_report', None) or {}
    _pruning_incompatible  = _pruning_report_final.get('torch_pruning_incompatible', False)
    techniques_used = []
    if getattr(args, 'use_bn_fusion', True):
        techniques_used.append("BN Fusion")
    if getattr(args, 'use_pruning', False) and not _pruning_incompatible \
            and _gating_report.get('Structured Pruning', True):
        techniques_used.append(
            f"Structured Pruning (ratio={getattr(args,'pruning_ratio',0.3)}, "
            f"max={getattr(args,'pruning_max_ratio',1.0)})"
        )
    if args.use_low_rank and _gating_report.get('Low-Rank Factorization', True):
        techniques_used.append(f"Low-Rank Factorization (ε={args.lrf_epsilon})")
    if args.use_clustering and _gating_report.get('Weight Clustering', True):
        techniques_used.append(f"Weight Clustering (k={args.num_clusters})")

    current_accuracy = getattr(compressed_pre_quant, '_gating_accuracy', orig_accuracy)
    print(f"\n  [Gating] Accuracy after structural compression: "
          f"{current_accuracy:.2f}%  (baseline: {orig_accuracy:.2f}%)")

    # Generate pruning report if pruning survived its gate (and wasn't a
    # torch_pruning_incompatible skip -- see _pruning_incompatible above;
    # there is nothing meaningful to plot for a model that was never
    # actually touched).
    if getattr(args, 'use_pruning', False) and not _pruning_incompatible \
            and _gating_report.get('Structured Pruning', True) \
            and hasattr(compressed_pre_quant, '_pruning_report'):
        from sigularty.visualization import plot_pruning_report
        _pruning_report_path = getattr(args, 'pruning_report_path', 'pruning_report.png')
        plot_pruning_report(
            pruning_report=compressed_pre_quant._pruning_report,
            save_path=_pruning_report_path,
        )

    # Phase B: GPTQ → Quantization → final KD recovery.
    #
    # ORDER: GPTQ runs FIRST, standard quantization SECOND.
    # If dynamic INT8 runs before GPTQ, it replaces every nn.Linear with
    # _Int8Linear. GPTQ then finds zero nn.Linear instances -> 0/0 layers.
    # fp16 does not replace the module class, but running GPTQ first is
    # consistent regardless of quant_mode.
    #
    # Mutual exclusion: GPTQ INT4 + dynamic INT8 both quantize Linear weights.
    # Running both would discard the Hessian correction. Skip dynamic when GPTQ ran.
    # fp16 after GPTQ is fine: it halves conv weights that GPTQ didn't touch.

    compressed_model = compressed_pre_quant

    # Tracks whether the FINAL compressed_model actually ended up as a
    # dynamic-INT8 (CPU-resident) model.  This is now a real tracked flag
    # rather than re-deriving it from args.quant_mode at every later point,
    # because gating can REVERT quantization — if dynamic quant gets
    # reverted, the model is back on its original device even though
    # args.quant_mode still says 'dynamic'.  Every later device-dependent
    # decision (the fp16 recast, the final KD step's device, Step 6's
    # acc_device) reads this flag instead of re-checking args.quant_mode.
    _final_quant_is_dynamic = False

    # ── Step 4: GPTQ (gated) ───────────────────────────────────────────────────
    _use_gptq = getattr(args, 'use_gptq', False)
    if _use_gptq:
        print("\n" + "=" * 70)
        print(f"STEP 4: GPTQ INT{getattr(args, 'gptq_bits', 4)}")
        print("=" * 70)
        from sigularty.compression import apply_gptq_quantization
        import copy as _copy_gptq
        _gptq_bits = getattr(args, 'gptq_bits', 4)
        _pre_gptq_model    = compressed_model
        _pre_gptq_accuracy = current_accuracy
        _gptq_result = apply_gptq_quantization(
            _copy_gptq.deepcopy(compressed_model),
            dataloader=train_loader,
            device=args.device,
            bits=_gptq_bits,
            num_calibration_batches=getattr(args, 'gptq_cal_batches', 16),
            block_size=getattr(args, 'gptq_block_size', 128),
            # NEW: only ask for trainable (QAT/STE) GPTQ layers when a KD
            # fine-tune is actually going to run afterward (Step 5B below) —
            # that's the only thing that would ever use shadow_weight's
            # trainability. Otherwise this is exactly the old frozen
            # _Int4Linear, zero added memory overhead. See _Int4LinearQAT's
            # docstring for why this matters: without it, the final KD step
            # (whose whole documented job is recovering accuracy lost to
            # every prior step, explicitly including quantization) could
            # never actually touch a single GPTQ-quantized weight.
            qat=getattr(args, 'use_kd_finetune', False),
        )
        compressed_model, current_accuracy, _gptq_kept = gate_technique_accuracy(
            f"GPTQ INT{_gptq_bits}", _pre_gptq_model, _gptq_result, test_loader,
            args.device, _threshold, pre_accuracy=_pre_gptq_accuracy,
        )
        # GPTQ has no internal fine-tune loop (its only recovery happens
        # later, in the shared post-quantization KD step), so there is no
        # algorithm/fine-tune split to report here — its marginal number is
        # already a clean read. total_eligible_layers/layers_quantized comes
        # straight from apply_gptq_quantization's own ._gptq_report, so a
        # "0/0" here (e.g. the only Linear layer is smaller than
        # min_layer_size) is exactly the "nothing to quantize" case worth
        # flagging loudly rather than reading as a harmless 0.00pp drop.
        _gptq_report_data = getattr(_gptq_result, '_gptq_report', {}) or {}
        _gptq_n_quantized  = _gptq_report_data.get('layers_quantized', 0)
        _gptq_n_eligible   = _gptq_report_data.get('total_eligible_layers', 0)
        _gptq_post_acc = (
            current_accuracy if _gptq_kept
            else measure_accuracy(_gptq_result, test_loader, args.device)
        )
        report_technique_impact(
            f"GPTQ INT{_gptq_bits}",
            pre_model=_pre_gptq_model, post_model=_gptq_result,
            pre_accuracy=_pre_gptq_accuracy, post_accuracy=_gptq_post_acc,
            was_kept=_gptq_kept,
            pre_device=args.device, post_device=args.device,
            original_accuracy=orig_accuracy, original_size_mb=orig_size_mb,
            original_latency_ms=orig_latency_ms,
            input_shape=_impact_input_shape, input_dtype=_impact_input_dtype,
            structural_summary=(
                f"{_gptq_n_quantized}/{_gptq_n_eligible} eligible Linear "
                f"layer(s) quantized to INT{_gptq_bits}"
            ),
            structural_zero=(_gptq_n_quantized == 0),
        )
        if not _gptq_kept:
            # GPTQ was reverted — treat it as if it had never been requested
            # for every downstream decision (skip-messages, KD device, etc.).
            _use_gptq = False

    techniques_used_gptq_label = f"GPTQ INT{getattr(args, 'gptq_bits', 4)}"
    if _use_gptq:
        techniques_used.append(techniques_used_gptq_label)

    # ── Step 5: Standard quantization (gated) ──────────────────────────────────
    # fp16 is gated on GPTQ being enabled too. fp16 alone only gives ~2x
    # compression and adds accuracy risk; it is only worth the risk as part
    # of the GPTQ + fp16 combo (GPTQ handles Linear layers at 8x, fp16 handles
    # the remaining conv/norm weights at 2x). Without GPTQ, skip fp16 entirely
    # and leave the model at full precision.
    _skip_dynamic = _use_gptq and getattr(args, 'quant_mode', 'fp16') == 'dynamic'
    _skip_fp16_no_gptq = (not _use_gptq) and getattr(args, 'quant_mode', 'fp16') == 'fp16'
    _skip_quant = _skip_dynamic or _skip_fp16_no_gptq

    if args.use_quantization and _skip_fp16_no_gptq:
        print(f"\n  [Quantization] Skipping fp16 — use_gptq=False (or GPTQ was "
              f"reverted by the accuracy gate). fp16 alone gives only ~2x "
              f"compression; not worth the accuracy risk without GPTQ's 8x "
              f"compression on Linear layers alongside it.")

    if args.use_quantization and not _skip_quant:
        _step_num = 5 if _use_gptq else 4
        print("\n" + "=" * 70)
        print(f"STEP {_step_num}: QUANTIZATION ({args.quant_mode.upper()})")
        print("=" * 70)
        from sigularty.compression import apply_quantization
        import copy as _copy_quant
        _pre_quant_model    = compressed_model
        _pre_quant_accuracy = current_accuracy
        quant_input = _copy_quant.deepcopy(compressed_model)
        if args.quant_mode == 'dynamic':
            quant_input = quant_input.cpu()
        _quant_result = apply_quantization(
            quant_input,
            dataloader=train_loader,
            mode=args.quant_mode,
            num_calibration_batches=args.quant_cal_batches,
        )
        _quant_eval_device = 'cpu' if args.quant_mode == 'dynamic' else args.device
        compressed_model, current_accuracy, _quant_kept = gate_technique_accuracy(
            f"Quantization ({args.quant_mode})", _pre_quant_model, _quant_result,
            test_loader, _quant_eval_device, _threshold,
            # pre_accuracy is supplied explicitly so the gate measures ONLY
            # the post-quant model (on its correct device) — it never has to
            # re-measure pre_quant_model, which avoids accidentally moving
            # that fp32 model's device placement via measure_accuracy()'s
            # internal .to(device) call (which would otherwise corrupt the
            # pre-model's device if this technique gets reverted below).
            pre_accuracy=_pre_quant_accuracy,
        )
        # Same device-safety reasoning as the gate call above applies here:
        # pre_device=args.device (where _pre_quant_model genuinely still
        # lives — quantization never touches it) and post_device=
        # _quant_eval_device (cpu for dynamic, args.device otherwise) are
        # passed explicitly rather than reused from one shared variable, so
        # measure_latency's internal .to(device) call never relocates a
        # model to the wrong place. No internal fine-tune loop exists for
        # this technique (recovery happens later, in the shared post-
        # quantization KD step), so there is no algorithm/fine-tune split
        # to report — the marginal number is already a clean read.
        _quant_post_acc = (
            current_accuracy if _quant_kept
            else measure_accuracy(_quant_result, test_loader, _quant_eval_device)
        )
        report_technique_impact(
            f"Quantization ({args.quant_mode})",
            pre_model=_pre_quant_model, post_model=_quant_result,
            pre_accuracy=_pre_quant_accuracy, post_accuracy=_quant_post_acc,
            was_kept=_quant_kept,
            pre_device=args.device, post_device=_quant_eval_device,
            original_accuracy=orig_accuracy, original_size_mb=orig_size_mb,
            original_latency_ms=orig_latency_ms,
            input_shape=_impact_input_shape, input_dtype=_impact_input_dtype,
            structural_summary=f"all parameters cast to {args.quant_mode}",
            structural_zero=False,
        )
        if _quant_kept:
            techniques_used.append(f"Quantization ({args.quant_mode})")
            _final_quant_is_dynamic = (args.quant_mode == 'dynamic')
        # else: reverted — compressed_model is back to the fp32 pre-quant
        # model, which never moved off args.device, so
        # _final_quant_is_dynamic correctly stays False.
    elif args.use_quantization and _skip_dynamic:
        _step_num = 5 if _use_gptq else 4
        print(f"\n  [Step {_step_num}] Skipping dynamic INT8 — GPTQ already ran on "
              f"Linear layers. Use quant_mode='fp16' to also compress conv weights.")

    # ── Step 5B: Final KD recovery fine-tune (gated + 6d retry) ───────────────
    # This is the definitive KD step — it recovers accuracy lost from both
    # structural compression AND quantization, using the original model as
    # teacher.  It is gated like every other technique (revert if KD itself
    # somehow makes things worse), and ADDITIONALLY checked against the
    # cumulative drop vs the ORIGINAL baseline (not just pre-KD) — if KD
    # wasn't enough to bring the whole pipeline back under threshold, the
    # per-epoch test-accuracy trend is inspected for one bounded, single
    # retry attempt with an adjusted learning rate.
    if getattr(args, 'use_kd_finetune', False):
        _kd_epochs = getattr(args, 'kd_epochs', 3)
        _kd_lr     = getattr(args, 'kd_lr', 0.0001)
        print("\n" + "=" * 70)
        print(f"STEP 5B: KD FINE-TUNE  ({_kd_epochs} epoch(s) post-quantization)")
        print("=" * 70)
        from sigularty.compression import fine_tune_with_distillation
        _kd_device = 'cpu' if _final_quant_is_dynamic else args.device
        _pre_kd_model    = compressed_model
        _pre_kd_accuracy = current_accuracy
        try:
            _kd_result = fine_tune_with_distillation(
                student=compressed_model,
                teacher=model,
                dataloader=train_loader,
                num_classes=getattr(args, 'num_classes', 102),
                epochs=_kd_epochs,
                lr=_kd_lr,
                device=_kd_device,
                temperature=getattr(args, 'kd_temperature', 4.0),
                alpha=getattr(args, 'kd_alpha', 0.7),
                max_batches=getattr(args, 'kd_max_batches', 50),
                test_loader=test_loader,
            )
        except Exception as _kd_e:
            print(f"  ⚠️  KD fine-tune failed: {_kd_e}")
            _kd_result = None

        if _kd_result is not None:
            compressed_model, current_accuracy, _kd_kept = gate_technique_accuracy(
                "KD Fine-tune", _pre_kd_model, _kd_result, test_loader, _kd_device,
                _threshold, pre_accuracy=_pre_kd_accuracy,
            )
            # KD *is* the fine-tune — there's no separate structural/
            # algorithm phase preceding it the way there is for Pruning/LRF/
            # Clustering, so no pre_finetune_accuracy is passed and the
            # algorithm/fine-tune split is naturally skipped (marginal +
            # cumulative only). fine_tune_with_distillation upcasts the
            # whole student to float32 internally for training stability
            # (see its docstring), so if the model entered this step fp16-
            # quantized, the size line below can show a real, expected
            # INCREASE right after KD — not a bug, just the training-
            # precision round-trip, before the pipeline's later fp16-recast
            # shrinks it back down.
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
                original_accuracy=orig_accuracy, original_size_mb=orig_size_mb,
                original_latency_ms=orig_latency_ms,
                input_shape=_impact_input_shape, input_dtype=_impact_input_dtype,
                pre_size_mb=_pre_kd_size_mb,
                structural_summary=(
                    "n/a — recovers accuracy lost to every prior technique, "
                    "no structural change of its own"
                ),
                structural_zero=False,
            )
            if _kd_kept:
                techniques_used.append(
                    f"KD Fine-tune (T={getattr(args,'kd_temperature',4.0):.1f}, "
                    f"α={getattr(args,'kd_alpha',0.7):.2f})"
                )

                # ── 6d: cumulative recovery-quality check + one bounded retry ──
                cumulative_drop = orig_accuracy - current_accuracy
                if cumulative_drop > _threshold:
                    _history = getattr(_kd_result, '_kd_history', [])
                    _test_accs = [h['test_acc'] for h in _history if h.get('test_acc') is not None]
                    _retry_lr: Optional[float] = None
                    _retry_reason: Optional[str] = None

                    if len(_test_accs) >= 2:
                        _deltas = [_test_accs[i] - _test_accs[i - 1] for i in range(1, len(_test_accs))]
                        _avg_delta = sum(_deltas) / len(_deltas)
                        _all_positive = all(d > 0 for d in _deltas)
                        if abs(_avg_delta) < 0.5:
                            _retry_lr = _kd_lr * 0.5
                            _retry_reason = (
                                f"stagnant (avg epoch-to-epoch change "
                                f"{_avg_delta:+.2f}pp, |Δ|<0.5pp)"
                            )
                        elif _all_positive and _avg_delta < 2.0:
                            _retry_lr = _kd_lr * 1.5
                            _retry_reason = (
                                f"small constant increase (avg {_avg_delta:+.2f}pp/"
                                f"epoch, <2pp)"
                            )

                    if _retry_lr is not None:
                        print(
                            f"\n  [KD Recovery] Cumulative accuracy drop "
                            f"{cumulative_drop:.2f}pp still exceeds the "
                            f"{_threshold:.1f}pp threshold after KD. Trend is "
                            f"{_retry_reason} — retrying ONCE with lr "
                            f"{_kd_lr:.2e} → {_retry_lr:.2e}, restarting fresh "
                            f"from the pre-KD checkpoint. This result is "
                            f"accepted unconditionally regardless of outcome — "
                            f"no further retries.\n"
                        )
                        try:
                            _retry_result = fine_tune_with_distillation(
                                student=_pre_kd_model,
                                teacher=model,
                                dataloader=train_loader,
                                num_classes=getattr(args, 'num_classes', 102),
                                epochs=_kd_epochs,
                                lr=_retry_lr,
                                device=_kd_device,
                                temperature=getattr(args, 'kd_temperature', 4.0),
                                alpha=getattr(args, 'kd_alpha', 0.7),
                                max_batches=getattr(args, 'kd_max_batches', 50),
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
                        except Exception as _retry_e:
                            print(f"  ⚠️  KD recovery retry failed ({_retry_e}) — "
                                  f"keeping the pre-retry result.")
                    else:
                        print(
                            f"\n  [KD Recovery] Cumulative accuracy drop "
                            f"{cumulative_drop:.2f}pp still exceeds the "
                            f"{_threshold:.1f}pp threshold, but the per-epoch "
                            f"trend doesn't match the stagnant/small-increase "
                            f"retry conditions — accepting the result as-is. "
                            f"Consider less aggressive compression settings.\n"
                        )

    # ── Collapse any QAT-mode GPTQ layers back to frozen, compact form ───────
    # Safe to call unconditionally: if qat was never requested (no fine-tune
    # was going to run, or GPTQ wasn't used at all), this just walks the
    # model, finds zero _Int4LinearQAT instances, and returns unchanged.
    # Must happen here — after the KD block above has fully finished, kept
    # or reverted either way — because whatever model comes out of it may
    # still be carrying a full fp32 shadow weight per GPTQ'd layer, and
    # nothing past this point (the final report, ONNX export, saving to
    # disk) should ever see that: it needs to look like an ordinary,
    # frozen-INT4 GPTQ model from here on.
    from sigularty.compression import collapse_qat_layers
    compressed_model = collapse_qat_layers(compressed_model)
    if _use_gptq and getattr(args, 'use_kd_finetune', False):
        # Collapse reconstructs the exact same weight the QAT layer's own
        # forward() was already computing (fake_quantize(shadow_weight) —
        # the straight-through estimator's forward VALUE, independent of
        # its backward-only .detach() correction), so this should not
        # change accuracy — re-measuring anyway rather than trusting that
        # by assumption, since this is the one point where the model's
        # actual computation graph just changed shape. _kd_device is
        # guaranteed set by this point: it's assigned unconditionally near
        # the top of the `if use_kd_finetune` block above, which is exactly
        # the condition gating entry into this branch too.
        current_accuracy = measure_accuracy(compressed_model, test_loader, _kd_device)

    # ── Device for evaluating the final compressed model ──────────────────────
    # fp16 stays on the main compute device; dynamic INT8 forces CPU (fbgemm).
    # Uses _final_quant_is_dynamic (a tracked flag, see above) rather than
    # re-deriving from args.quant_mode, because quantization gating can have
    # reverted dynamic quant back to a GPU-resident fp32 model.
    acc_device = 'cpu' if _final_quant_is_dynamic else args.device

    # Re-apply fp16 if quantization mode is fp16 — fine_tune_with_distillation
    # internally casts the student to float32 for stable training, so without
    # this the reported size would equal the float32 baseline even though
    # quant_mode='fp16' was requested.  Only relevant when fp16 quantization
    # actually survived its gate (i.e. it's in techniques_used).
    if getattr(args, 'use_quantization', False) and getattr(args, 'quant_mode', '') == 'fp16' \
            and any('Quantization (fp16)' in t for t in techniques_used):
        compressed_model = compressed_model.to(torch.float16)
        current_accuracy = measure_accuracy(compressed_model, test_loader, acc_device)
        print(f"\n  [fp16 recast] Re-measured accuracy after final fp16 cast: "
              f"{current_accuracy:.2f}%")

    # ── 6. Evaluate accuracy and latency ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: EVALUATION")
    print("=" * 70)

    # Use input shape and dtype from the registry (set in step 1 above).
    # Falls back to the standard 224×224 vision shape when not using the registry.
    INPUT_SHAPE  = getattr(args, 'input_shape',  (1, 3, 224, 224))
    INPUT_DTYPE  = getattr(args, 'input_dtype',  None)

    # --- Accuracy --- both numbers are already known: orig_accuracy was
    # measured once at the very start of this function, and current_accuracy
    # was already measured on the correct device by the last gate (or the
    # fp16 recast above) that touched compressed_model.  Re-measuring here
    # would just repeat work for an identical number.
    print(f"  Original   accuracy : {orig_accuracy:.2f}%  (measured once, reused throughout)")
    comp_accuracy = current_accuracy
    print(f"  Compressed accuracy : {comp_accuracy:.2f}%  [on {acc_device}]")
    print(f"  Accuracy drop       : {orig_accuracy - comp_accuracy:.2f}%")

    # --- Latency — measured on the final compressed model ---
    print(f"\n  Measuring original model latency (on {args.device})...")
    orig_latency = measure_latency(model, INPUT_SHAPE, args.device,
                                   num_iterations=100, warmup=10,
                                   input_dtype=INPUT_DTYPE)

    print(f"  Measuring compressed model latency (on {acc_device})...")
    comp_latency = measure_latency(compressed_model, INPUT_SHAPE, acc_device,
                                   num_iterations=100, warmup=10,
                                   input_dtype=INPUT_DTYPE)

    speedup = orig_latency['mean_ms'] / comp_latency['mean_ms'] if comp_latency['mean_ms'] else 0.0
    print(f"\n  Original   latency  : {orig_latency['mean_ms']:.3f} ms (mean, {args.device})")
    print(f"  Compressed latency  : {comp_latency['mean_ms']:.3f} ms (mean, {acc_device})")
    print(f"  Speedup             : {speedup:.2f}x")

    # swap eval_device to acc_device so the report call below uses the right one
    eval_device = acc_device

    # ── 6. ONNX export and runtime (optional) ────────────────────────────────
    onnx_results = None
    if args.export_onnx:
        print("\n" + "=" * 70)
        print("STEP 5: ONNX EXPORT")
        print("=" * 70)
        try:
            onnx_abs_path = export_to_onnx(
                model=compressed_model,
                save_path=args.onnx_path,
                input_shape=INPUT_SHAPE,
                device='cpu',          # always export on CPU for portability
                opset_version=args.onnx_opset,
                dynamic_batch=True,
            )
            techniques_used.append(f"ONNX export (opset {args.onnx_opset})")

            if args.run_onnx:
                print("\n" + "=" * 70)
                print("STEP 6: ONNX RUNTIME INFERENCE")
                print("=" * 70)
                onnx_results = run_onnx_inference(
                    onnx_path=onnx_abs_path,
                    dataloader=test_loader,
                    input_shape=INPUT_SHAPE,
                    num_latency_iterations=100,
                    warmup=10,
                )
                print(f"\n  ONNX Runtime accuracy : {onnx_results['accuracy_pct']:.2f}%")
                print(f"  ONNX Runtime latency  : {onnx_results['mean_latency_ms']:.3f} ms (mean)")
                acc_delta = orig_accuracy - onnx_results['accuracy_pct']
                print(f"  Accuracy vs original  : {acc_delta:.2f}% drop")

        except (RuntimeError, ImportError) as exc:
            print(f"  ⚠️  ONNX step skipped: {exc}")

    # ── 7. Compression report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP {'7' if args.export_onnx and args.run_onnx else '6' if args.export_onnx else '6'}: COMPRESSION REPORT")
    print("=" * 70)

    # Extract pruning report if available — passes layer stats into the report
    _pruning_report = None
    if getattr(args, 'use_pruning', False) and not _pruning_incompatible \
            and _gating_report.get('Structured Pruning', True) \
            and hasattr(compressed_pre_quant, '_pruning_report'):
        _pruning_report = compressed_pre_quant._pruning_report

    metrics = {
        'original_accuracy':     orig_accuracy,
        'compressed_accuracy':   comp_accuracy,
        'original_size_mb':      get_model_size_mb(model),
        'compressed_size_mb':    get_model_size_mb(compressed_model),
        'original_params':       count_parameters(model),
        'compressed_params':     count_parameters(compressed_model),
        'original_latency_ms':   orig_latency['mean_ms'],
        'compressed_latency_ms': comp_latency.get('mean_ms'),   # GPU, float32 pre-quant
        'techniques_used':       techniques_used,
        'onnx_results':          onnx_results,
        'pruning_report':        _pruning_report,   # None when pruning not used / reverted
    }

    final_metrics = generate_compression_report(
        original_model=model,
        compressed_model=compressed_model,
        metrics_dict=metrics,
        save_path=args.report_path,
        dataloader=test_loader,
        device=eval_device,
        input_shape=INPUT_SHAPE,
        latency_iterations=100,
        latency_warmup=10,
    )

    print("\n✅ Done. Report saved to:", args.report_path)
    return final_metrics


# ============================================================================
# COMPRESSION SUPPORT UTILITIES
# Moved here from compression.py because they are generic training/evaluation
# helpers that are not themselves compression algorithms.
#
#   get_nested_layer          — dotted-path module traversal (used by clustering)
#   run_behavioral_probe      — KL/cosine output comparison (used after pruning)
#   fine_tune_with_distillation — the single fine-tuning function used everywhere
#
# Design rule: compression.py contains ONLY compression algorithms.
# All training loops, evaluation helpers, and model traversal utilities live here.
#
# NOTE: this file's fine_tune_with_distillation is a DUPLICATE of the one in
# compression.py (pre-existing duplication, not introduced by these changes).
# Both copies have been updated identically: per-epoch training accuracy is
# now ALWAYS tracked (not just when test_loader is given), the epoch-1-
# exactly-0.0% safety check uses that training accuracy, and the returned
# model gets `._kd_history` attached for callers that need the full
# per-epoch trend (e.g. run_compression_pipeline's 6d recovery-quality
# retry check).
# ============================================================================

def get_nested_layer(model: nn.Module, dotted_name: str) -> nn.Module:
    """
    Traverse a dotted attribute path to retrieve a nested sub-module.

    Example: get_nested_layer(model, 'layer1.0.conv1')

    Args:
        model:       Root nn.Module to traverse.
        dotted_name: Dot-separated attribute path string.

    Returns:
        The target nn.Module at the end of the path.

    Raises:
        AttributeError: If any segment of the path does not exist.
    """
    layer = model
    for part in dotted_name.split('.'):
        layer = getattr(layer, part)
    return layer


def run_behavioral_probe(
    original_model: nn.Module,
    modified_model: nn.Module,
    dataloader,
    device: str,
    num_batches: int = 10,
) -> dict:
    """
    Compare original and modified model outputs on calibration data to assess
    how much a compression step changed the model's functional behaviour.

    Detection logic:
      If the output is 2D (batch × C), use KL divergence between
      softmax(original) and softmax(modified) — the classifier path.
      Otherwise (embeddings, regression outputs), use mean cosine similarity.

    KL divergence severity thresholds (lower is better):
      < 0.01  → negligible   (compression had almost no effect)
      < 0.05  → acceptable   (safe to deploy)
      < 0.15  → moderate     (consider reducing compression aggressiveness)
      ≥ 0.15  → high         (model behaviour changed significantly)

    Args:
        original_model: Unmodified baseline model.
        modified_model: Compressed model to compare against.
        dataloader:     DataLoader for calibration inputs.
        device:         'cuda' or 'cpu'.
        num_batches:    Number of batches to probe over.

    Returns:
        Dict with keys: metric, value, severity, recommendation.
    """
    orig_outputs:   list = []
    mod_outputs:    list = []

    original_model.eval()
    modified_model.eval()

    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            X = X.to(device)
            try:
                o = original_model(X)
                p = modified_model(X)
                orig_outputs.append(o.float().cpu())
                mod_outputs.append(p.float().cpu())
            except Exception:
                continue

    if not orig_outputs:
        return {
            'metric':         'none',
            'value':          None,
            'severity':       'unknown',
            'recommendation': 'Probe could not run — no valid batches completed.',
        }

    orig_cat = torch.cat(orig_outputs, dim=0)
    mod_cat  = torch.cat(mod_outputs,  dim=0)

    if orig_cat.ndim == 2:
        # Classifier path: KL divergence between softmax distributions
        p_orig = torch.softmax(orig_cat,  dim=-1).clamp(min=1e-8)
        p_mod  = torch.softmax(mod_cat,   dim=-1).clamp(min=1e-8)
        kl     = (p_orig * (p_orig / p_mod).log()).sum(dim=-1).mean().item()
        metric = 'kl_divergence'
        value  = kl
        if kl < 0.01:
            severity       = 'negligible'
            recommendation = 'Compression had negligible impact. Safe to deploy as-is.'
        elif kl < 0.05:
            severity       = 'acceptable'
            recommendation = 'Small behaviour change. Run evaluation to confirm accuracy.'
        elif kl < 0.15:
            severity       = 'moderate'
            recommendation = 'Noticeable behaviour shift. Consider reducing compression aggressiveness or adding fine-tuning.'
        else:
            severity       = 'high'
            recommendation = 'Significant behaviour change. Reduce compression ratio or increase fine-tuning epochs.'
    else:
        # Embedding/regression path: cosine similarity
        orig_flat = orig_cat.view(orig_cat.size(0), -1)
        mod_flat  = mod_cat.view(mod_cat.size(0),   -1)
        cos    = nn.functional.cosine_similarity(orig_flat, mod_flat, dim=1).mean().item()
        metric = 'cosine_similarity'
        value  = cos
        if cos > 0.99:
            severity       = 'negligible'
            recommendation = 'Outputs nearly identical. Safe to deploy.'
        elif cos > 0.95:
            severity       = 'acceptable'
            recommendation = 'Minor output drift. Evaluate on downstream task.'
        elif cos > 0.85:
            severity       = 'moderate'
            recommendation = 'Moderate output drift. Consider reducing compression aggressiveness.'
        else:
            severity       = 'high'
            recommendation = 'Significant output drift. Reduce compression ratio or increase fine-tuning.'

    severity_icons = {'negligible': '✅', 'acceptable': '✅', 'moderate': '⚠️', 'high': '❌'}
    print(f"\n  [Probe] Metric: {metric}  Value: {value:.4f}  Severity: {severity.upper()}")
    print(f"  {severity_icons.get(severity, '?')} {recommendation}")

    return {
        'metric':         metric,
        'value':          value,
        'severity':       severity,
        'recommendation': recommendation,
    }


def fine_tune_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    dataloader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: str,
    temperature: float = 4.0,
    alpha: float = 0.7,
    max_batches: int = 0,
    test_loader=None,
) -> nn.Module:
    """
    Fine-tune a compressed model using knowledge distillation from the original.

    This is the SINGLE fine-tuning function used everywhere in the pipeline:
      - After weight clustering     (replaces old _fine_tune_after_clustering, when kd_teacher is supplied)
      - As the dedicated KD step    (pipeline step use_kd_finetune=True)
      - As the FINAL post-quantization recovery step (run_compression_pipeline)

    (Structured Pruning and Low-Rank Factorization use the separate
    _kd_recovery_fine_tune() helper in compression.py instead — they need the
    {epoch, loss, acc} history format their existing reports already expect.)

    The student learns to:
      1. Predict correct class labels (weighted by alpha).
      2. Match the teacher's soft output distribution (weighted by 1-alpha).

    Loss formula:
      L = alpha * CrossEntropy(student_logits, y_hard)
        + (1 - alpha) * T² * KLDiv(softmax(teacher/T) || log_softmax(student/T))

    The T² multiplier compensates for the flattening of distributions at high T —
    without it, raising T would automatically shrink the distillation loss
    regardless of alpha.

    Why KD beats plain CE fine-tuning:
      Hard labels say "this is class 3" (one-hot).
      Teacher soft labels say "82% class 3, 12% class 7, 6% class 1" —
      encoding inter-class relationships learned over the full training run.
      The student recovers 2–5% more accuracy from this richer signal on
      the same number of epochs and same calibration data.

    Tracks per-epoch training accuracy (ALWAYS) and test accuracy (when
    test_loader is given), attached to the returned model as
    `._kd_history = [{epoch, loss, train_acc, test_acc}, ...]` — used by
    run_compression_pipeline's post-quantization KD step to assess recovery
    quality and decide whether a single bounded LR-adjusted retry is
    worthwhile.

    Safety check: if epoch 1's TRAINING accuracy is EXACTLY 0.0%, a major
    warning is printed and remaining epochs are skipped — almost always a
    structural break rather than slow convergence.

    Args:
        student:     Compressed model to fine-tune. NOT modified; deep copy made.
        teacher:     Original uncompressed model. Never modified.
        dataloader:  Training DataLoader.
        num_classes: Number of output classes.
        epochs:      Number of fine-tuning epochs.
        lr:          Learning rate for Adam.
        device:      'cuda' or 'cpu'.
        temperature: Softmax temperature T. Higher → softer distributions →
                     more inter-class information. Try 2–6. Default 4.
        alpha:       Hard-label (task) loss weight. 1-alpha = distillation weight.
                     alpha=0.7: 70% task loss, 30% distillation. Default 0.7.
        max_batches: Max batches per epoch. 0 = full dataloader.
        test_loader: Optional evaluation DataLoader. When provided, test accuracy
                     is reported and tracked at the end of each epoch.

    Returns:
        Fine-tuned student model (new object; student is NOT modified), with
        `._kd_history` attached.

    Input dtype handling (NLP vs vision):
        Batches are moved to `device` and cast to float32 ONLY if they are
        already floating-point (vision pixel tensors). Integer batches
        (NLP token-ID tensors, torch.long, from e.g. RoBERTa/BERT/GPT-2
        dataloaders) are left untouched. This mirrors measure_accuracy's
        existing convention and matters concretely: nn.Embedding.forward()
        performs an index lookup and requires Long/Int indices — casting
        token IDs to float32 raises "Expected tensor for argument #1
        'indices' to have one of the following scalar types: Long, Int;
        but got ... FloatTensor" deep inside torch.embedding(). An earlier
        version of this function (and a since-removed duplicate copy that
        used to live in compression.py) cast every batch to float()
        unconditionally, which broke every NLP model in the registry the
        first time KD fine-tuning ran on one.
    """
    # Normalise the entire student to float32 for training.
    # Mixed-precision models (e.g. after fp16 quantization) can have fp16 weights
    # but fp32 biases, causing: "Input type (c10::Half) and bias type (float)".
    # Training in float32 with Adam is always numerically safe; the caller can
    # re-quantize the returned model to fp16 if storage precision is required.
    student_ft = copy.deepcopy(student).to(device).float()
    teacher_ft = copy.deepcopy(teacher).to(device).float()
    teacher_ft.eval()
    student_ft.train()

    optimizer   = torch.optim.Adam(student_ft.parameters(), lr=lr)
    ce_loss     = nn.CrossEntropyLoss()
    kl_loss     = nn.KLDivLoss(reduction='batchmean')
    accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)

    print(f"\n  [KD Fine-tune] {epochs} epoch(s)  T={temperature}  alpha={alpha}  lr={lr}")
    print(f"  [KD Fine-tune] Loss = {alpha:.1f}xCE + {1-alpha:.1f}xT²xKL(teacher||student)")

    history: list = []

    for epoch in tqdm(range(epochs), desc="  KD Fine-tune"):
        total_loss = n_batches = 0
        total_train_acc = 0.0
        student_ft.train()

        for _i, (X, y) in enumerate(dataloader):
            if max_batches > 0 and _i >= max_batches:
                break
            X = X.to(device, non_blocking=True)
            if X.is_floating_point():
                X = X.to(dtype=torch.float32)   # vision pixels only — never touches int64 token IDs
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher_ft(X)

            student_logits = student_ft(X)

            hard_loss     = ce_loss(student_logits, y)
            p_teacher     = torch.softmax(teacher_logits / temperature, dim=1)
            log_p_student = torch.log_softmax(student_logits / temperature, dim=1)
            soft_loss     = kl_loss(log_p_student, p_teacher) * (temperature ** 2)
            loss          = alpha * hard_loss + (1.0 - alpha) * soft_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            total_train_acc += accuracy_fn(student_logits.argmax(1), y).item()
            n_batches       += 1

        avg_loss      = total_loss / n_batches if n_batches > 0 else 0.0
        avg_train_acc = (total_train_acc / n_batches) * 100 if n_batches > 0 else 0.0

        test_acc: Optional[float] = None
        if test_loader is not None:
            student_ft.eval()
            test_correct = test_total = 0
            with torch.no_grad():
                for X_t, y_t in test_loader:
                    X_t = X_t.to(device, non_blocking=True)
                    if X_t.is_floating_point():
                        X_t = X_t.to(dtype=torch.float32)
                    y_t = y_t.to(device, non_blocking=True)
                    test_correct += accuracy_fn(student_ft(X_t).argmax(1), y_t).item()
                    test_total   += 1
            test_acc = (test_correct / test_total) * 100 if test_total > 0 else 0.0
            print(f"    Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
                  f"train_acc={avg_train_acc:.2f}%  test_acc={test_acc:.2f}%")
        else:
            print(f"    Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
                  f"train_acc={avg_train_acc:.2f}%")

        history.append({'epoch': epoch + 1, 'loss': avg_loss,
                         'train_acc': avg_train_acc, 'test_acc': test_acc})

        if epoch == 0 and avg_train_acc == 0.0:
            print(f"\n  🚨 [KD Fine-tune] MAJOR WARNING: epoch 1 training accuracy is "
                  f"EXACTLY 0.0%. This usually signals a structural break (shape "
                  f"mismatch, dead/NaN gradients, wrong num_classes) rather than "
                  f"ordinary slow convergence. Skipping remaining {epochs - 1} "
                  f"epoch(s) to save compute.\n")
            break

    print("  [KD Fine-tune] Done.")
    student_ft._kd_history = history
    return student_ft


if __name__ == "__main__":
    sys.exit("It is only a module not meant to run independently")