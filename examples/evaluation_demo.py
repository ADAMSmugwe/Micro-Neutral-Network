"""
Day 19: Model Evaluation and Interpretability
Trains a small CNN on 5000 MNIST samples, then runs a full evaluation suite:
  - Confusion matrix heatmap
  - Per-class precision / recall / F1
  - Misclassified examples grid
  - Learned filter visualisation
  - One-vs-rest ROC curves with AUC
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import struct
import gzip
import urllib.request
import time

from src.layers import Conv2D, MaxPool2D, Flatten, ReLU, Layer
from src.network import Network
from src.metrics import (
    confusion_matrix,
    classification_report,
    print_classification_report,
    plot_confusion_matrix,
    show_misclassified,
    plot_filters,
    multiclass_roc,
    plot_roc_curves,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'evaluation')
os.makedirs(OUT_DIR, exist_ok=True)


# ── MNIST loader ──────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist')
BASE_URL   = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images':  't10k-images-idx3-ubyte.gz',
    'test_labels':  't10k-labels-idx1-ubyte.gz',
}


def _download(filename):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f'Downloading {filename} ...', end=' ', flush=True)
        urllib.request.urlretrieve(BASE_URL + filename, path)
        print('done')
    return path


def _read_images(path):
    with gzip.open(path, 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_labels(path):
    with gzip.open(path, 'rb') as f:
        _, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist():
    X_train = _read_images(_download(FILES['train_images'])).astype(np.float32) / 255.0
    y_train = _read_labels(_download(FILES['train_labels']))
    X_test  = _read_images(_download(FILES['test_images'])).astype(np.float32) / 255.0
    y_test  = _read_labels(_download(FILES['test_labels']))
    return X_train, y_train, X_test, y_test


def one_hot(y, n=10):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


# ── Model ─────────────────────────────────────────────────────────────────────

def build_cnn():
    net = Network()
    net.add_layer(Conv2D(in_channels=1,  out_channels=8,  filter_size=3, stride=1, padding=1))
    net.add_layer(ReLU())
    net.add_layer(MaxPool2D(pool_size=2, stride=2))
    net.add_layer(Conv2D(in_channels=8,  out_channels=16, filter_size=3, stride=1, padding=1))
    net.add_layer(ReLU())
    net.add_layer(MaxPool2D(pool_size=2, stride=2))
    net.add_layer(Flatten())
    net.add_layer(Layer(7 * 7 * 16, 128, 'relu'))
    net.add_layer(Layer(128, 10, 'softmax'))
    net.set_loss('cross_entropy')
    return net


def batch_predict(net, X, batch_size=256):
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(net.forward(X[i:i + batch_size]))
    return np.concatenate(preds, axis=0)


# ── Training ──────────────────────────────────────────────────────────────────

print('Loading MNIST...')
X_tr_full, y_tr_full, X_te_full, y_te_full = load_mnist()

np.random.seed(42)
N_TRAIN, N_TEST = 5000, 1000
idx = np.random.permutation(len(X_tr_full))
X_train = X_tr_full[idx[:N_TRAIN]].reshape(-1, 28, 28, 1)
y_train = y_tr_full[idx[:N_TRAIN]]
X_test  = X_te_full[:N_TEST].reshape(-1, 28, 28, 1)
y_test  = y_te_full[:N_TEST]

Y_train = one_hot(y_train)

print(f'Train: {X_train.shape}  Test: {X_test.shape}\n')

np.random.seed(0)
net = build_cnn()

EPOCHS, BATCH_SIZE, LR = 10, 128, 0.01
print(f'Training for {EPOCHS} epochs...')
for epoch in range(EPOCHS):
    t0 = time.time()
    net.train_mode()
    perm = np.random.permutation(N_TRAIN)
    X_s, Y_s = X_train[perm], Y_train[perm]
    for i in range(0, N_TRAIN, BATCH_SIZE):
        Xb, Yb = X_s[i:i + BATCH_SIZE], Y_s[i:i + BATCH_SIZE]
        pred = net.forward(Xb)
        net.backward(Yb, pred)
        net.update(lr=LR)
    net.eval_mode()
    test_probs = batch_predict(net, X_test)
    test_acc   = np.mean(np.argmax(test_probs, axis=1) == y_test)
    print(f'  Epoch {epoch + 1}/{EPOCHS}  test acc: {test_acc:.3f}  ({time.time()-t0:.1f}s)')

net.eval_mode()


# ── Evaluation ────────────────────────────────────────────────────────────────

print('\n=== Evaluation ===')
test_probs = batch_predict(net, X_test)           # (N, 10) softmax probabilities
y_pred     = np.argmax(test_probs, axis=1)        # integer predictions

overall_acc = np.mean(y_pred == y_test)
print(f'\nOverall accuracy: {overall_acc:.4f}  ({int(overall_acc * N_TEST)}/{N_TEST} correct)')

# 1. Per-class report
CLASSES = [str(i) for i in range(10)]
print('\n--- Per-class metrics ---')
report = classification_report(y_test, y_pred, classes=CLASSES)
print_classification_report(report)

# 2. Confusion matrix
print('\n--- Confusion matrix ---')
cm = confusion_matrix(y_test, y_pred)
print(cm)

print('\nTop confused pairs (true → pred, count):')
cm_offdiag = cm.copy()
np.fill_diagonal(cm_offdiag, 0)
flat_idx = np.argsort(cm_offdiag.ravel())[::-1][:5]
for fi in flat_idx:
    t, p = divmod(fi, 10)
    print(f'  {t} → {p}: {cm_offdiag[t, p]}')

# 3. Plots (all saved to outputs/evaluation/)
try:
    plot_confusion_matrix(
        y_test, y_pred, classes=CLASSES,
        save_path=os.path.join(OUT_DIR, 'confusion_matrix.png')
    )

    show_misclassified(
        X_test, y_test, y_pred, n=10,
        save_path=os.path.join(OUT_DIR, 'misclassified.png')
    )

    # First Conv2D layer is layers[0]
    conv1 = net.layers[0]
    plot_filters(
        conv1,
        save_path=os.path.join(OUT_DIR, 'conv1_filters.png')
    )

    # One-vs-rest ROC curves
    roc_results = multiclass_roc(y_test, test_probs, n_classes=10)
    print('\n--- Per-class AUC (one-vs-rest) ---')
    for c in range(10):
        print(f'  Class {c}: AUC = {roc_results[c][2]:.4f}')
    macro_auc = np.mean([roc_results[c][2] for c in range(10)])
    print(f'  Macro AUC: {macro_auc:.4f}')

    plot_roc_curves(
        roc_results, n_classes=10,
        save_path=os.path.join(OUT_DIR, 'roc_curves.png')
    )

except ImportError:
    print('\n(matplotlib not available — skipping plots)')

print(f'\nAll outputs saved to: {OUT_DIR}')
print('Done.')
