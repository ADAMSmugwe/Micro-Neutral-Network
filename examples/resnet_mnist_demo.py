"""
Day 22: Residual Networks – Plain CNN vs ResNet comparison on MNIST.

Builds two networks with roughly equal depth:
  Plain CNN  – 6 conv layers with BatchNorm, no skip connections
  ResNet     – same layers organised into 5 residual blocks

Trains both and prints a side-by-side accuracy/loss comparison.
"""
import sys
import os
import time
import struct
import gzip
import urllib.request

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layers import (
    Conv2D, MaxPool2D, Flatten, ReLU,
    BatchNorm, ConvBatchNorm, GlobalAvgPool2D, ResidualBlock, Layer,
)
from src.network import Network

# ── data ────────────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist')
BASE_URL  = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
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
        print(f'  Downloading {filename} ...', end=' ', flush=True)
        urllib.request.urlretrieve(BASE_URL + filename, path)
        print('done')
    return path


def load_mnist():
    def read_images(path):
        with gzip.open(path, 'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), np.uint8).reshape(n, r, c)

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            _, n = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), np.uint8)

    X_train = read_images(_download(FILES['train_images'])).astype(np.float32) / 255.0
    y_train = read_labels(_download(FILES['train_labels']))
    X_test  = read_images(_download(FILES['test_images'])).astype(np.float32) / 255.0
    y_test  = read_labels(_download(FILES['test_labels']))
    return X_train, y_train, X_test, y_test


def one_hot(y, n=10):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def accuracy(net, X, y_int, batch_size=128):
    preds = []
    net.eval_mode()
    for i in range(0, len(X), batch_size):
        preds.append(np.argmax(net.forward(X[i:i + batch_size]), axis=1))
    return float(np.mean(np.concatenate(preds) == y_int))


# ── network builders ─────────────────────────────────────────────────────────

def build_plain_cnn():
    """6 conv layers with ConvBatchNorm + ReLU, no skip connections.

    Input (batch, 28, 28, 1)
      Conv 1→16, 3x3 s1 p1 → BN → ReLU   : (28,28,16)
      Conv 16→16, 3x3 s1 p1 → BN → ReLU  : (28,28,16)
      Conv 16→32, 3x3 s2 p1 → BN → ReLU  : (14,14,32)
      Conv 32→32, 3x3 s1 p1 → BN → ReLU  : (14,14,32)
      Conv 32→64, 3x3 s2 p1 → BN → ReLU  : (7,7,64)
      Conv 64→64, 3x3 s1 p1 → BN → ReLU  : (7,7,64)
      GlobalAvgPool2D                      : (64,)
      Dense 64→10 softmax
    """
    net = Network()

    def cbn_relu(in_c, out_c, stride=1):
        net.add_layer(Conv2D(in_c, out_c, filter_size=3, stride=stride, padding=1))
        net.add_layer(ConvBatchNorm(out_c))
        net.add_layer(ReLU())

    cbn_relu(1, 16)
    cbn_relu(16, 16)
    cbn_relu(16, 32, stride=2)   # 14×14
    cbn_relu(32, 32)
    cbn_relu(32, 64, stride=2)   # 7×7
    cbn_relu(64, 64)

    net.add_layer(GlobalAvgPool2D())
    net.add_layer(Layer(64, 10, 'softmax'))
    net.set_loss('cross_entropy')
    return net


def build_resnet():
    """ResNet with 5 residual blocks (same conv budget as plain CNN).

    Input (batch, 28, 28, 1)
      Conv 1→16, 3x3 s1 p1 → BN → ReLU   : (28,28,16)  [stem]
      ResBlock(16→16, s1)                  : (28,28,16)
      ResBlock(16→32, s2)  [projection]    : (14,14,32)
      ResBlock(32→32, s1)                  : (14,14,32)
      ResBlock(32→64, s2)  [projection]    : (7,7,64)
      ResBlock(64→64, s1)                  : (7,7,64)
      GlobalAvgPool2D                      : (64,)
      Dense 64→10 softmax
    """
    net = Network()

    # stem
    net.add_layer(Conv2D(1, 16, filter_size=3, stride=1, padding=1))
    net.add_layer(ConvBatchNorm(16))
    net.add_layer(ReLU())

    # residual blocks
    net.add_layer(ResidualBlock(16, 16, stride=1))
    net.add_layer(ResidualBlock(16, 32, stride=2))   # projection shortcut
    net.add_layer(ResidualBlock(32, 32, stride=1))
    net.add_layer(ResidualBlock(32, 64, stride=2))   # projection shortcut
    net.add_layer(ResidualBlock(64, 64, stride=1))

    net.add_layer(GlobalAvgPool2D())
    net.add_layer(Layer(64, 10, 'softmax'))
    net.set_loss('cross_entropy')
    return net


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(net, X, Y, y_int, batch_size, lr, momentum):
    net.train_mode()
    perm = np.random.permutation(len(X))
    X_s, Y_s = X[perm], Y[perm]
    batch_losses = []

    for i in range(0, len(X_s), batch_size):
        Xb = X_s[i:i + batch_size]
        Yb = Y_s[i:i + batch_size]
        pred = net.forward(Xb)
        batch_losses.append(float(net.loss(Yb, pred)))
        net.backward(Yb, pred)
        net.update(lr=lr, momentum=momentum)

    return float(np.mean(batch_losses))


def run_experiment(name, net, X_train, Y_train, y_train, X_test, y_test,
                   epochs, batch_size, lr, momentum):
    print(f'\n{"─" * 60}')
    print(f'  {name}')
    print(f'{"─" * 60}')
    train_losses, test_accs = [], []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(net, X_train, Y_train, y_train, batch_size, lr, momentum)
        acc  = accuracy(net, X_test, y_test)
        elapsed = time.time() - t0

        train_losses.append(loss)
        test_accs.append(acc)
        print(f'  Epoch {epoch:2d}/{epochs}  loss={loss:.4f}  test_acc={acc:.4f}  ({elapsed:.1f}s)')

    return train_losses, test_accs


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading MNIST...')
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist()

    np.random.seed(42)
    N_TRAIN, N_TEST = 5000, 1000
    idx = np.random.permutation(len(X_train_full))

    # CNN input: (batch, h, w, channels)
    X_train = X_train_full[idx[:N_TRAIN]].reshape(-1, 28, 28, 1)
    y_train = y_train_full[idx[:N_TRAIN]]
    X_test  = X_test_full[:N_TEST].reshape(-1, 28, 28, 1)
    y_test  = y_test_full[:N_TEST]
    Y_train = one_hot(y_train)

    print(f'Train: {X_train.shape}  Test: {X_test.shape}')

    EPOCHS     = 10
    BATCH_SIZE = 64
    LR         = 0.01
    MOMENTUM   = 0.9

    print(f'\nHyperparameters: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, momentum={MOMENTUM}')

    # ── Plain CNN ────────────────────────────────────────────────
    plain_net = build_plain_cnn()
    plain_losses, plain_accs = run_experiment(
        'Plain CNN  (6 conv layers, no skip connections)',
        plain_net, X_train, Y_train, y_train, X_test, y_test,
        EPOCHS, BATCH_SIZE, LR, MOMENTUM,
    )

    # ── ResNet ───────────────────────────────────────────────────
    res_net = build_resnet()
    res_losses, res_accs = run_experiment(
        'ResNet  (5 residual blocks + stem conv)',
        res_net, X_train, Y_train, y_train, X_test, y_test,
        EPOCHS, BATCH_SIZE, LR, MOMENTUM,
    )

    # ── Summary ──────────────────────────────────────────────────
    print(f'\n{"═" * 60}')
    print('  FINAL COMPARISON')
    print(f'{"═" * 60}')
    print(f'  {"Metric":<30}  {"Plain CNN":>10}  {"ResNet":>10}')
    print(f'  {"-" * 54}')
    print(f'  {"Final train loss":<30}  {plain_losses[-1]:>10.4f}  {res_losses[-1]:>10.4f}')
    print(f'  {"Final test accuracy":<30}  {plain_accs[-1]:>10.4f}  {res_accs[-1]:>10.4f}')
    print(f'  {"Best test accuracy":<30}  {max(plain_accs):>10.4f}  {max(res_accs):>10.4f}')
    print(f'  {"Epoch of best acc":<30}  {plain_accs.index(max(plain_accs))+1:>10d}  {res_accs.index(max(res_accs))+1:>10d}')
    print(f'{"═" * 60}')

    # ── Optional: loss curve (ASCII) ─────────────────────────────
    print('\n  Training loss curve (epoch-by-epoch):')
    print(f'  {"Epoch":<6}  {"Plain CNN loss":>14}  {"ResNet loss":>12}  {"Plain acc":>10}  {"ResNet acc":>10}')
    print(f'  {"-" * 60}')
    for i in range(EPOCHS):
        marker = ' <' if res_accs[i] > plain_accs[i] else ('  >' if plain_accs[i] > res_accs[i] else '')
        print(f'  {i+1:<6}  {plain_losses[i]:>14.4f}  {res_losses[i]:>12.4f}'
              f'  {plain_accs[i]:>10.4f}  {res_accs[i]:>10.4f}{marker}')

    # ── Optional: save plot if matplotlib available ───────────────
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs_range = range(1, EPOCHS + 1)

        ax1.plot(epochs_range, plain_losses, 'b-o', label='Plain CNN', markersize=4)
        ax1.plot(epochs_range, res_losses,   'r-o', label='ResNet',    markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, plain_accs, 'b-o', label='Plain CNN', markersize=4)
        ax2.plot(epochs_range, res_accs,   'r-o', label='ResNet',    markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Test Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'resnet_comparison.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120)
        print(f'\n  Plot saved to {os.path.normpath(out_path)}')
    except ImportError:
        pass

    print('\nDone.')


if __name__ == '__main__':
    main()
