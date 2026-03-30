"""
Day 26 – Transfer Learning Demo
================================
Demonstrates adapting a CNN pretrained on MNIST to classify Fashion-MNIST
items using two strategies:

  A) Train from scratch      – baseline, 30 epochs
  B) Feature extraction      – freeze base, train new head only, 10 epochs
  C) Fine-tuning             – unfreeze last 3 layers, lower LR, 15 epochs

Fashion-MNIST is a drop-in MNIST replacement (same 28×28 shape, 10 classes)
that uses clothing items instead of digits.

Run from the project root::

    python examples/transfer_learning_demo.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import time
import struct
import gzip
import urllib.request

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.layers import Conv2D, MaxPool2D, Flatten, ReLU, Layer
from src.network import Network
from src.transfer import TransferLearning


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

MNIST_CACHE   = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist')
FASHION_CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'fashion_mnist')

FASHION_URLS = {
    'train_images': 'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-images-idx3-ubyte.gz',
    'train_labels': 'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-labels-idx1-ubyte.gz',
    'test_images':  'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
    'test_labels':  'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
}

MNIST_URLS = {
    'train_images': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    'test_images':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
}


def _download(url, cache_dir, filename):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        print(f'  Downloading {filename} ...', end=' ', flush=True)
        urllib.request.urlretrieve(url, path)
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
    print('Loading MNIST...')
    X_tr = _read_images(_download(MNIST_URLS['train_images'], MNIST_CACHE, 'train-images-idx3-ubyte.gz'))
    y_tr = _read_labels(_download(MNIST_URLS['train_labels'], MNIST_CACHE, 'train-labels-idx1-ubyte.gz'))
    X_te = _read_images(_download(MNIST_URLS['test_images'],  MNIST_CACHE, 't10k-images-idx3-ubyte.gz'))
    y_te = _read_labels(_download(MNIST_URLS['test_labels'],  MNIST_CACHE, 't10k-labels-idx1-ubyte.gz'))
    return X_tr.astype(np.float32) / 255.0, y_tr, X_te.astype(np.float32) / 255.0, y_te


def load_fashion_mnist():
    """Load Fashion-MNIST, falling back to keras if the download fails."""
    print('Loading Fashion-MNIST...')
    try:
        X_tr = _read_images(_download(FASHION_URLS['train_images'], FASHION_CACHE, 'train-images-idx3-ubyte.gz'))
        y_tr = _read_labels(_download(FASHION_URLS['train_labels'], FASHION_CACHE, 'train-labels-idx1-ubyte.gz'))
        X_te = _read_images(_download(FASHION_URLS['test_images'],  FASHION_CACHE, 't10k-images-idx3-ubyte.gz'))
        y_te = _read_labels(_download(FASHION_URLS['test_labels'],  FASHION_CACHE, 't10k-labels-idx1-ubyte.gz'))
        return X_tr.astype(np.float32) / 255.0, y_tr, X_te.astype(np.float32) / 255.0, y_te
    except Exception as e:
        print(f'  Direct download failed ({e}), trying keras...')
        from tensorflow.keras.datasets import fashion_mnist
        (X_tr, y_tr), (X_te, y_te) = fashion_mnist.load_data()
        return X_tr.astype(np.float32) / 255.0, y_tr, X_te.astype(np.float32) / 255.0, y_te


def one_hot(y, n=10):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def eval_accuracy(net, X, y_int, batch_size=256):
    net.eval_mode()
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(np.argmax(net.forward(X[i:i + batch_size]), axis=1))
    return float(np.mean(np.concatenate(preds) == y_int))


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_cnn(seed=None):
    """Small CNN: Conv(1→8) → Pool → Conv(8→16) → Pool → Flatten → Dense(128) → Dense(10)."""
    if seed is not None:
        np.random.seed(seed)
    net = Network()
    net.add_layer(Conv2D(1,  8,  filter_size=3, stride=1, padding=1))
    net.add_layer(ReLU())
    net.add_layer(MaxPool2D(pool_size=2, stride=2))
    net.add_layer(Conv2D(8,  16, filter_size=3, stride=1, padding=1))
    net.add_layer(ReLU())
    net.add_layer(MaxPool2D(pool_size=2, stride=2))
    net.add_layer(Flatten())
    net.add_layer(Layer(7 * 7 * 16, 128, 'relu'))
    net.add_layer(Layer(128, 10, 'softmax'))
    net.set_loss('cross_entropy')
    return net


# ---------------------------------------------------------------------------
# Training loop with accuracy tracking
# ---------------------------------------------------------------------------

def train_model(net, X_train, Y_train, X_test, y_test_int,
                epochs, lr, batch_size=128, label=''):
    history = {'train_acc': [], 'test_acc': [], 'epoch_time': []}
    n = X_train.shape[0]
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'  epochs={epochs}  lr={lr}  batch={batch_size}  n_train={n}')

    # Show frozen/trainable layer summary
    param_layers = [l for l in net.layers if hasattr(l, 'trainable')]
    frozen = sum(1 for l in param_layers if not l.trainable)
    print(f'  Parametrized layers: {len(param_layers)} total, {frozen} frozen, {len(param_layers)-frozen} trainable')
    print(f'{"="*60}')

    for epoch in range(epochs):
        t0 = time.time()
        net.train_mode()
        perm = np.random.permutation(n)
        Xs, Ys = X_train[perm], Y_train[perm]

        for i in range(0, n, batch_size):
            Xb, Yb = Xs[i:i + batch_size], Ys[i:i + batch_size]
            pred = net.forward(Xb)
            net.backward(Yb, pred)
            net.update(lr=lr, optimizer='adam')

        elapsed = time.time() - t0
        tr_acc  = eval_accuracy(net, X_train, np.argmax(Y_train, axis=1))
        te_acc  = eval_accuracy(net, X_test, y_test_int)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['epoch_time'].append(elapsed)
        print(f'  Epoch {epoch+1:3d}/{epochs}  train: {tr_acc:.3f}  test: {te_acc:.3f}  ({elapsed:.1f}s)')

    total_time = sum(history['epoch_time'])
    print(f'  Final test accuracy: {history["test_acc"][-1]:.3f}  total time: {total_time:.1f}s')
    return history


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    # ── Load datasets ────────────────────────────────────────────────────────
    X_mnist_tr, y_mnist_tr, X_mnist_te, y_mnist_te = load_mnist()

    # Use 10k MNIST samples for pretraining
    N_MNIST = 10_000
    idx = np.random.permutation(len(X_mnist_tr))[:N_MNIST]
    X_mnist_tr = X_mnist_tr[idx].reshape(-1, 28, 28, 1)
    Y_mnist_tr = one_hot(y_mnist_tr[idx])

    X_fashion_tr, y_fashion_tr, X_fashion_te, y_fashion_te = load_fashion_mnist()

    # Use 10k Fashion samples so experiments run in reasonable time
    N_FASHION_TRAIN = 10_000
    N_FASHION_TEST  = 2_000
    fidx = np.random.permutation(len(X_fashion_tr))[:N_FASHION_TRAIN]
    X_fa_tr = X_fashion_tr[fidx].reshape(-1, 28, 28, 1)
    Y_fa_tr = one_hot(y_fashion_tr[fidx])
    X_fa_te = X_fashion_te[:N_FASHION_TEST].reshape(-1, 28, 28, 1)
    y_fa_te = y_fashion_te[:N_FASHION_TEST]

    FASHION_CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # ── Pretrain on MNIST ────────────────────────────────────────────────────
    print('\n>>> Pretraining CNN on MNIST...')
    base_net = build_cnn(seed=0)
    train_model(base_net, X_mnist_tr, Y_mnist_tr,
                X_mnist_te[:1000].reshape(-1, 28, 28, 1), y_mnist_te[:1000],
                epochs=15, lr=0.001, label='Pretraining on MNIST')

    print(f'\nMNIST pretrain test acc: '
          f'{eval_accuracy(base_net, X_mnist_te[:1000].reshape(-1,28,28,1), y_mnist_te[:1000]):.3f}')

    # ── Experiment A: Train from scratch on Fashion-MNIST ────────────────────
    scratch_net = build_cnn(seed=1)
    hist_scratch = train_model(
        scratch_net, X_fa_tr, Y_fa_tr, X_fa_te, y_fa_te,
        epochs=30, lr=0.001,
        label='Experiment A — Train from scratch on Fashion-MNIST (30 epochs)'
    )

    # ── Experiment B: Feature extraction ────────────────────────────────────
    # Deep-copy base model so we don't mutate it for experiment C
    fe_net = copy.deepcopy(base_net)
    tl_fe = TransferLearning(fe_net, new_output_size=10)
    tl_fe.prepare_for_feature_extraction()
    frozen, total = tl_fe.frozen_count()
    print(f'\n  Feature extraction: {frozen}/{total} layers frozen.')

    hist_fe = train_model(
        fe_net, X_fa_tr, Y_fa_tr, X_fa_te, y_fa_te,
        epochs=10, lr=0.001,
        label='Experiment B — Feature extraction (10 epochs, base frozen)'
    )

    # ── Experiment C: Fine-tuning ────────────────────────────────────────────
    ft_net = copy.deepcopy(fe_net)   # start from feature-extraction result
    tl_ft = TransferLearning(ft_net, new_output_size=10)
    tl_ft.prepare_for_fine_tuning(n_unfreeze=3)   # unfreeze last 3 param layers
    frozen, total = tl_ft.frozen_count()
    print(f'\n  Fine-tuning: {frozen}/{total} layers frozen, {total-frozen} trainable.')

    hist_ft = train_model(
        ft_net, X_fa_tr, Y_fa_tr, X_fa_te, y_fa_te,
        epochs=15, lr=0.0001,   # 10x lower LR to preserve pretrained features
        label='Experiment C — Fine-tuning (15 epochs, low LR=0.0001)'
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    acc_scratch = hist_scratch['test_acc'][-1]
    acc_fe      = hist_fe['test_acc'][-1]
    acc_ft      = hist_ft['test_acc'][-1]

    time_scratch = sum(hist_scratch['epoch_time'])
    time_fe      = sum(hist_fe['epoch_time'])
    time_ft      = sum(hist_ft['epoch_time'])

    print('\n' + '='*60)
    print('  SUMMARY')
    print('='*60)
    print(f'  {"Experiment":<40} {"Test Acc":>8}  {"Time (s)":>9}')
    print(f'  {"-"*57}')
    print(f'  {"A. From scratch (30 epochs)":<40} {acc_scratch:>8.3f}  {time_scratch:>9.1f}')
    print(f'  {"B. Feature extraction (10 epochs)":<40} {acc_fe:>8.3f}  {time_fe:>9.1f}')
    print(f'  {"C. Fine-tuning (15 epochs)":<40} {acc_ft:>8.3f}  {time_ft:>9.1f}')
    print('='*60)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Test accuracy over epochs
    ax = axes[0]
    ax.plot(range(1, len(hist_scratch['test_acc']) + 1),
            hist_scratch['test_acc'], 'r-o', markersize=4, label=f'A. Scratch (best {max(hist_scratch["test_acc"]):.3f})')
    ax.plot(range(1, len(hist_fe['test_acc']) + 1),
            hist_fe['test_acc'],      'b-s', markersize=4, label=f'B. Feature extraction (best {max(hist_fe["test_acc"]):.3f})')
    # Plot fine-tuning starting from where feature extraction ended
    ft_offset = len(hist_fe['test_acc'])
    ax.plot(range(ft_offset + 1, ft_offset + len(hist_ft['test_acc']) + 1),
            hist_ft['test_acc'],      'g-^', markersize=4, label=f'C. Fine-tuning (best {max(hist_ft["test_acc"]):.3f})')
    ax.axvline(x=ft_offset + 0.5, color='gray', linestyle='--', alpha=0.5, label='Fine-tuning starts')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Fashion-MNIST Test Accuracy')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bar chart: final accuracy vs total training time
    ax2 = axes[1]
    labels  = ['A. Scratch\n(30 ep)', 'B. Feature\nExtraction\n(10 ep)', 'C. Fine-\nTuning\n(15 ep)']
    accs    = [acc_scratch, acc_fe, acc_ft]
    times   = [time_scratch, time_fe, time_ft]
    colors  = ['#e74c3c', '#3498db', '#2ecc71']
    x = np.arange(len(labels))
    bars = ax2.bar(x, accs, color=colors, alpha=0.8, width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Final Test Accuracy')
    ax2.set_ylim(0, 1)
    ax2.set_title('Final Accuracy vs Training Time')
    for bar, acc, t in zip(bars, accs, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}\n({t:.0f}s)', ha='center', va='bottom', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Transfer Learning: MNIST → Fashion-MNIST', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'transfer_learning_results.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {os.path.normpath(out_path)}')

    # ── Per-class breakdown for fine-tuned model ─────────────────────────────
    ft_net.eval_mode()
    preds = np.argmax(ft_net.forward(X_fa_te), axis=1)
    print('\nPer-class accuracy (fine-tuned model):')
    for cls in range(10):
        mask = y_fa_te == cls
        cls_acc = np.mean(preds[mask] == cls)
        bar = '#' * int(cls_acc * 20)
        print(f'  {FASHION_CLASSES[cls]:>12}: {cls_acc:.3f}  |{bar:<20}|')


if __name__ == '__main__':
    main()
