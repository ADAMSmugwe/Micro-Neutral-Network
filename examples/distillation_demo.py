"""
Day 25 – Knowledge Distillation Demo
======================================
Demonstrates training a small "student" CNN on MNIST in two ways:

  1. From scratch  – only ground-truth one-hot labels
  2. Distillation  – guided by a pre-trained "teacher" CNN's soft predictions

The teacher is a medium CNN with BatchNorm (higher capacity).
The student is a tiny CNN (≈10x fewer parameters).

Expected results (approximate)
--------------------------------
  Teacher (medium CNN)       ~98-99 % test accuracy
  Student – from scratch     ~95-97 % test accuracy
  Student – distilled        ~97-98 % test accuracy  ← closes the gap
"""

import sys
import os
import time
import struct
import gzip
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.network import Network
from src.distillation import DistillationTrainer, softmax_temperature
from src.layers import (
    Conv2D, MaxPool2D, Flatten, ReLU,
    ConvBatchNorm, GlobalAvgPool2D, Layer,
)

# ── configuration ─────────────────────────────────────────────────────────────

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
BASE_URL    = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

TEACHER_EPOCHS   = 15      # epochs to train the teacher
STUDENT_EPOCHS   = 30      # epochs for from-scratch and distilled student
BATCH_SIZE       = 128
LR               = 0.001
TEMPERATURE      = 4.0     # distillation temperature
ALPHA            = 0.7     # weight for hard-label cross-entropy
N_TRAIN          = 10000   # subset for speed  (set to 60000 for full dataset)
N_TEST           = 2000

# ── data helpers ──────────────────────────────────────────────────────────────

def _download(filename: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"  Downloading {filename} ...", end=" ", flush=True)
        urllib.request.urlretrieve(BASE_URL + filename, path)
        print("done")
    return path


def load_mnist():
    def read_images(path):
        with gzip.open(path, "rb") as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), np.uint8).reshape(n, r, c)

    def read_labels(path):
        with gzip.open(path, "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), np.uint8)

    X_train = read_images(_download(FILES["train_images"])).astype(np.float32) / 255.0
    y_train = read_labels(_download(FILES["train_labels"]))
    X_test  = read_images(_download(FILES["test_images"])).astype(np.float32)  / 255.0
    y_test  = read_labels(_download(FILES["test_labels"]))
    return X_train, y_train, X_test, y_test


def one_hot(y, n=10):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def batch_accuracy(net: Network, X: np.ndarray, y_int: np.ndarray,
                   batch_size: int = 256) -> float:
    """Evaluate classification accuracy in mini-batches."""
    preds = []
    net.eval_mode()
    for i in range(0, len(X), batch_size):
        logits = net.forward_logits(X[i : i + batch_size])
        preds.append(np.argmax(logits, axis=1))
    return float(np.mean(np.concatenate(preds) == y_int))


def param_count(net: Network) -> int:
    """Count total trainable parameters."""
    total = 0
    for layer in net.layers:
        if hasattr(layer, "weights"):
            total += layer.weights.size + layer.biases.size
        elif hasattr(layer, "filters"):
            total += layer.filters.size + layer.biases.size
        if hasattr(layer, "gamma"):
            total += layer.gamma.size + layer.beta.size
    return total


# ── network builders ──────────────────────────────────────────────────────────

def build_teacher() -> Network:
    """Medium CNN  –  higher capacity teacher.

    Input: (batch, 28, 28, 1)
      Conv 1→16,  3×3, pad=1 → ConvBN → ReLU  : (28,28,16)
      Conv 16→32, 3×3, pad=1 → ConvBN → ReLU  : (28,28,32)
      MaxPool 2×2                               : (14,14,32)
      Conv 32→64, 3×3, pad=1 → ConvBN → ReLU  : (14,14,64)
      Conv 64→64, 3×3, pad=1 → ConvBN → ReLU  : (14,14,64)
      MaxPool 2×2                               : (7,7,64)
      GlobalAvgPool2D                           : (64,)
      Dense 64→10, linear   (logits output)
    """
    net = Network()

    def cbn_relu(in_c, out_c):
        net.add_layer(Conv2D(in_c, out_c, filter_size=3, stride=1, padding=1))
        net.add_layer(ConvBatchNorm(out_c))
        net.add_layer(ReLU())

    cbn_relu(1, 16)
    cbn_relu(16, 32)
    net.add_layer(MaxPool2D(pool_size=2, stride=2))

    cbn_relu(32, 64)
    cbn_relu(64, 64)
    net.add_layer(MaxPool2D(pool_size=2, stride=2))

    net.add_layer(GlobalAvgPool2D())
    net.add_layer(Layer(64, 10, activation="linear"))
    net.set_loss("cross_entropy")
    return net


def build_student() -> Network:
    """Tiny CNN  –  the student we want to compress knowledge into.

    Input: (batch, 28, 28, 1)
      Conv 1→8,  3×3, pad=1 → ReLU  : (28,28,8)
      MaxPool 2×2                    : (14,14,8)
      Conv 8→16, 3×3, pad=1 → ReLU  : (14,14,16)
      MaxPool 2×2                    : (7,7,16)
      Flatten                        : (784,)
      Dense 784→64, relu
      Dense 64→10, linear  (logits output)
    """
    net = Network()
    net.add_layer(Conv2D(1,  8,  filter_size=3, stride=1, padding=1))
    net.add_layer(ReLU())
    net.add_layer(MaxPool2D(pool_size=2, stride=2))
    net.add_layer(Conv2D(8, 16, filter_size=3, stride=1, padding=1))
    net.add_layer(ReLU())
    net.add_layer(MaxPool2D(pool_size=2, stride=2))
    net.add_layer(Flatten())
    net.add_layer(Layer(784, 64, activation="relu"))
    net.add_layer(Layer(64, 10, activation="linear"))
    net.set_loss("cross_entropy")
    return net


# ── training helpers ──────────────────────────────────────────────────────────

def train_standard(net: Network, X: np.ndarray, Y: np.ndarray,
                   y_int: np.ndarray, X_val: np.ndarray, y_val_int: np.ndarray,
                   epochs: int, lr: float, batch_size: int,
                   label: str) -> dict:
    """Standard training loop using cross-entropy on logits.

    Since both teacher and student end in a *linear* layer (logits), we
    manually apply softmax before computing cross-entropy loss during
    backward.  The network's backward() still works correctly because
    'linear' activation treats dA = dZ (identity derivative), so
    passing (softmax(logits) - y_true)/N as the upstream gradient is exact.
    """
    history = {"train_loss": [], "val_acc": []}
    n = len(X)

    for epoch in range(epochs):
        net.train_mode()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            Xb, Yb = X[idx], Y[idx]

            logits = net.forward(Xb)                # linear output = logits
            probs  = softmax_temperature(logits)     # apply softmax for loss

            # Cross-entropy gradient w.r.t. logits: (probs - y) / N
            grad = (probs - Yb) / len(Yb)
            net._backward_from_grad(grad)
            net.update(lr=lr, optimizer="adam")

            eps = 1e-12
            batch_loss = -float(
                np.mean(np.sum(Yb * np.log(np.clip(probs, eps, 1.0)), axis=1))
            )
            epoch_loss += batch_loss
            n_batches  += 1

        train_loss = epoch_loss / n_batches
        val_acc    = batch_accuracy(net, X_val, y_val_int)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d} | loss: {train_loss:.4f}  val_acc: {val_acc:.4f}")

    return history


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    teacher_acc: float,
    scratch_history: dict,
    distill_history: dict,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Day 25 – Knowledge Distillation on MNIST", fontsize=14, fontweight="bold")

    # ── val accuracy curves ────────────────────────────────────────────────
    ax = axes[0]
    epochs_scratch = range(1, len(scratch_history["val_acc"]) + 1)
    epochs_distill = range(1, len(distill_history["val_acc"]) + 1)

    ax.plot(epochs_scratch, scratch_history["val_acc"],
            color="#E07B54", linewidth=2, label="Student (from scratch)")
    ax.plot(epochs_distill, distill_history["val_acc"],
            color="#4E8FD6", linewidth=2, label=f"Student (distilled, T={TEMPERATURE})")
    ax.axhline(teacher_acc, color="#52A35A", linewidth=2,
               linestyle="--", label="Teacher (reference)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy Over Training")
    ax.legend()
    ax.set_ylim([0.88, 1.01])
    ax.grid(True, alpha=0.3)

    # ── training loss curves ───────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs_scratch, scratch_history["train_loss"],
            color="#E07B54", linewidth=2, label="Student (from scratch)")
    ax.plot(epochs_distill, distill_history["train_loss"],
            color="#4E8FD6", linewidth=2, label="Student (distilled)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "distillation_results.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nPlot saved → {path}")
    plt.close()


def plot_temperature_ablation(
    X_val: np.ndarray, y_val_int: np.ndarray,
    teacher: Network,
    Y_val: np.ndarray,
    X_train: np.ndarray, Y_train: np.ndarray,
) -> None:
    """Quick ablation: train one student per temperature and compare accuracy."""
    temperatures = [1.0, 2.0, 4.0, 8.0, 16.0]
    accs = []

    print("\n── Temperature Ablation ──────────────────────────────────────────")
    for T in temperatures:
        s = build_student()
        trainer = DistillationTrainer(teacher, s, temperature=T, alpha=ALPHA)
        trainer.train(
            X_train, Y_train,
            epochs=15, lr=LR, batch_size=BATCH_SIZE,
            verbose=False,
        )
        acc = batch_accuracy(s, X_val, y_val_int)
        accs.append(acc)
        print(f"  T = {T:5.1f}  →  val acc: {acc:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(t) for t in temperatures], accs, color="#4E8FD6", edgecolor="white")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Effect of Distillation Temperature on Student Accuracy")
    ax.set_ylim([max(0.0, min(accs) - 0.02), min(1.0, max(accs) + 0.02)])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "distillation_temperature_ablation.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Temperature ablation plot saved → {path}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Day 25 – Knowledge Distillation")
    print("=" * 62)

    # ── load and preprocess MNIST ──────────────────────────────────────────
    print("\nLoading MNIST …")
    X_train_all, y_train_all, X_test_all, y_test_all = load_mnist()

    # Reshape to (N, H, W, C)
    X_train_all = X_train_all[:, :, :, np.newaxis]
    X_test_all  = X_test_all[:, :, :, np.newaxis]

    # Use a subset for speed
    X_train = X_train_all[:N_TRAIN]
    y_train = y_train_all[:N_TRAIN]
    X_test  = X_test_all[:N_TEST]
    y_test  = y_test_all[:N_TEST]

    Y_train = one_hot(y_train)
    Y_test  = one_hot(y_test)

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")

    # ── build networks ─────────────────────────────────────────────────────
    teacher = build_teacher()
    student_scratch = build_student()
    student_distill = build_student()

    print(f"\nTeacher  parameters: {param_count(teacher):,}")
    print(f"Student  parameters: {param_count(student_scratch):,}")
    print(f"Compression ratio  : {param_count(teacher) / param_count(student_scratch):.1f}×")

    # ── Step 1: Train the teacher ──────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Step 1 – Training Teacher ({TEACHER_EPOCHS} epochs)")
    print(f"{'─'*62}")
    t0 = time.time()
    train_standard(
        teacher, X_train, Y_train, y_train,
        X_test, y_test,
        epochs=TEACHER_EPOCHS, lr=LR, batch_size=BATCH_SIZE,
        label="Teacher",
    )
    teacher_acc = batch_accuracy(teacher, X_test, y_test)
    print(f"\nTeacher test accuracy : {teacher_acc:.4f}  ({time.time()-t0:.1f}s)")

    # ── Step 2: Train student from scratch (baseline) ─────────────────────
    print(f"\n{'─'*62}")
    print(f"  Step 2 – Training Student from Scratch ({STUDENT_EPOCHS} epochs)")
    print(f"{'─'*62}")
    t0 = time.time()
    scratch_history = train_standard(
        student_scratch, X_train, Y_train, y_train,
        X_test, y_test,
        epochs=STUDENT_EPOCHS, lr=LR, batch_size=BATCH_SIZE,
        label="Scratch",
    )
    scratch_acc = batch_accuracy(student_scratch, X_test, y_test)
    print(f"\nStudent (scratch) test accuracy : {scratch_acc:.4f}  ({time.time()-t0:.1f}s)")

    # ── Step 3: Train student with distillation ────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Step 3 – Distillation Training  T={TEMPERATURE}, α={ALPHA}  ({STUDENT_EPOCHS} epochs)")
    print(f"{'─'*62}")
    t0 = time.time()
    trainer = DistillationTrainer(
        teacher, student_distill,
        temperature=TEMPERATURE,
        alpha=ALPHA,
    )
    distill_history = trainer.train(
        X_train, Y_train,
        epochs=STUDENT_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        val_data=(X_test, Y_test),
        verbose=True,
    )
    distill_acc = batch_accuracy(student_distill, X_test, y_test)
    print(f"\nStudent (distilled) test accuracy : {distill_acc:.4f}  ({time.time()-t0:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Results Summary")
    print(f"{'='*62}")
    print(f"  {'Model':<30} {'Params':>9}  {'Accuracy':>9}")
    print(f"  {'-'*52}")
    print(f"  {'Teacher (medium CNN)':<30} {param_count(teacher):>9,}  {teacher_acc:>9.2%}")
    print(f"  {'Student – from scratch':<30} {param_count(student_scratch):>9,}  {scratch_acc:>9.2%}")
    print(f"  {'Student – distilled':<30} {param_count(student_distill):>9,}  {distill_acc:>9.2%}")
    gap_closed = (distill_acc - scratch_acc) / (teacher_acc - scratch_acc + 1e-9) * 100
    print(f"\n  Distillation closed {gap_closed:.1f}% of the teacher–scratch accuracy gap.")
    print(f"  Temperature used : {TEMPERATURE}   alpha (hard weight) : {ALPHA}")
    print(f"{'='*62}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_results(teacher_acc, scratch_history, distill_history)
    plot_temperature_ablation(X_test, y_test, teacher, Y_test, X_train, Y_train)

    print("\nDay 25 complete!")


if __name__ == "__main__":
    main()
