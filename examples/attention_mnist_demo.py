"""
Day 23: Attention Mechanisms – Plain CNN vs ResNet vs ResNet+CBAM on MNIST.

Three networks with the same stem + residual block budget:
  Plain CNN          – 6 conv layers, BatchNorm, no skip connections
  ResNet             – 5 residual blocks + stem
  ResNet + CBAM      – same ResNet with CBAM attention after each ResBlock

After training, spatial attention maps for the last CBAM block are visualised
for a small grid of test images so you can see what the network "looks at".
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
    Conv2D, Flatten, ReLU,
    ConvBatchNorm, GlobalAvgPool2D, ResidualBlock, Layer, CBAM,
)
from src.network import Network


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



def build_plain_cnn():
    """6 conv layers + BatchNorm, no skip connections (baseline).

    Input (batch, 28, 28, 1)
      Conv 1→16, 3x3 s1 p1 → BN → ReLU   : (28,28,16)
      Conv 16→16, 3x3 s1 p1 → BN → ReLU  : (28,28,16)
      Conv 16→32, 3x3 s2 p1 → BN → ReLU  : (14,14,32)
      Conv 32→32, 3x3 s1 p1 → BN → ReLU  : (14,14,32)
      Conv 32→64, 3x3 s2 p1 → BN → ReLU  : (7,7,64)
      Conv 64→64, 3x3 s1 p1 → BN → ReLU  : (7,7,64)
      GlobalAvgPool2D → Dense 64→10 softmax
    """
    net = Network()

    def cbn_relu(in_c, out_c, stride=1):
        net.add_layer(Conv2D(in_c, out_c, filter_size=3, stride=stride, padding=1))
        net.add_layer(ConvBatchNorm(out_c))
        net.add_layer(ReLU())

    cbn_relu(1, 16)
    cbn_relu(16, 16)
    cbn_relu(16, 32, stride=2)
    cbn_relu(32, 32)
    cbn_relu(32, 64, stride=2)
    cbn_relu(64, 64)

    net.add_layer(GlobalAvgPool2D())
    net.add_layer(Layer(64, 10, 'softmax'))
    net.set_loss('cross_entropy')
    return net


def build_resnet():
    """ResNet: 5 residual blocks + stem (same depth as plain CNN).

    Input (batch, 28, 28, 1)
      Conv 1→16, 3x3 s1 p1 → BN → ReLU   : (28,28,16)  [stem]
      ResBlock(16→16, s1)                  : (28,28,16)
      ResBlock(16→32, s2)  [projection]    : (14,14,32)
      ResBlock(32→32, s1)                  : (14,14,32)
      ResBlock(32→64, s2)  [projection]    : (7,7,64)
      ResBlock(64→64, s1)                  : (7,7,64)
      GlobalAvgPool2D → Dense 64→10 softmax
    """
    net = Network()

    net.add_layer(Conv2D(1, 16, filter_size=3, stride=1, padding=1))
    net.add_layer(ConvBatchNorm(16))
    net.add_layer(ReLU())

    net.add_layer(ResidualBlock(16, 16, stride=1))
    net.add_layer(ResidualBlock(16, 32, stride=2))
    net.add_layer(ResidualBlock(32, 32, stride=1))
    net.add_layer(ResidualBlock(32, 64, stride=2))
    net.add_layer(ResidualBlock(64, 64, stride=1))

    net.add_layer(GlobalAvgPool2D())
    net.add_layer(Layer(64, 10, 'softmax'))
    net.set_loss('cross_entropy')
    return net


def build_attention_resnet():
    """ResNet + CBAM attention after each residual block.

    Input (batch, 28, 28, 1)
      Conv 1→16, 3x3 s1 p1 → BN → ReLU      : (28,28,16)  [stem]
      ResBlock(16→16, s1) → CBAM(16)          : (28,28,16)
      ResBlock(16→32, s2) → CBAM(32)          : (14,14,32)
      ResBlock(32→32, s1) → CBAM(32)          : (14,14,32)
      ResBlock(32→64, s2) → CBAM(64)          : (7,7,64)
      ResBlock(64→64, s1) → CBAM(64)          : (7,7,64)
      GlobalAvgPool2D → Dense 64→10 softmax

    Returns (net, cbam_blocks) so callers can inspect attention maps.
    """
    net = Network()
    cbam_blocks = []

    net.add_layer(Conv2D(1, 16, filter_size=3, stride=1, padding=1))
    net.add_layer(ConvBatchNorm(16))
    net.add_layer(ReLU())

    for in_c, out_c, stride in [(16, 16, 1), (16, 32, 2),
                                  (32, 32, 1), (32, 64, 2),
                                  (64, 64, 1)]:
        net.add_layer(ResidualBlock(in_c, out_c, stride=stride))
        cbam = CBAM(out_c, reduction=8)
        net.add_layer(cbam)
        cbam_blocks.append(cbam)

    net.add_layer(GlobalAvgPool2D())
    net.add_layer(Layer(64, 10, 'softmax'))
    net.set_loss('cross_entropy')
    return net, cbam_blocks



def train_one_epoch(net, X, Y, batch_size, lr, momentum):
    net.train_mode()
    perm = np.random.permutation(len(X))
    X_s, Y_s = X[perm], Y[perm]
    batch_losses = []
    for i in range(0, len(X_s), batch_size):
        Xb, Yb = X_s[i:i + batch_size], Y_s[i:i + batch_size]
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
        t0   = time.time()
        loss = train_one_epoch(net, X_train, Y_train, batch_size, lr, momentum)
        acc  = accuracy(net, X_test, y_test)
        elapsed = time.time() - t0
        train_losses.append(loss)
        test_accs.append(acc)
        print(f'  Epoch {epoch:2d}/{epochs}  loss={loss:.4f}  test_acc={acc:.4f}  ({elapsed:.1f}s)')
    return train_losses, test_accs



def visualize_attention(net, cbam_blocks, X_samples, y_samples, out_path):
    """Plot original images alongside the deepest CBAM spatial attention map."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available – skipping attention visualisation')
        return

    net.eval_mode()
    net.forward(X_samples)

    last_cbam   = cbam_blocks[-1]
    attn_maps   = last_cbam.spatial_attention._attn

    n = len(X_samples)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    fig.suptitle('Spatial Attention Maps (last CBAM block)', fontsize=12)

    for idx in range(n):
        orig = X_samples[idx, :, :, 0]
        axes[0, idx].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f'Label: {y_samples[idx]}', fontsize=9)
        axes[0, idx].axis('off')

        attn = attn_maps[idx, :, :, 0]
        im = axes[1, idx].imshow(attn, cmap='hot', vmin=0, vmax=1,
                                  interpolation='nearest')
        axes[1, idx].set_title(f'Attn {attn.shape[0]}×{attn.shape[1]}', fontsize=9)
        axes[1, idx].axis('off')

    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.6, label='Attention weight')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f'\n  Attention map saved to {os.path.normpath(out_path)}')



def main():
    print('Loading MNIST...')
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist()

    np.random.seed(42)
    N_TRAIN, N_TEST = 5000, 1000
    idx = np.random.permutation(len(X_train_full))

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

    print(f'\nHyperparameters: epochs={EPOCHS}, batch={BATCH_SIZE}, '
          f'lr={LR}, momentum={MOMENTUM}')

    plain_net = build_plain_cnn()
    plain_losses, plain_accs = run_experiment(
        'Plain CNN  (6 conv layers, no skip connections)',
        plain_net, X_train, Y_train, y_train, X_test, y_test,
        EPOCHS, BATCH_SIZE, LR, MOMENTUM,
    )

    res_net = build_resnet()
    res_losses, res_accs = run_experiment(
        'ResNet  (5 residual blocks + stem)',
        res_net, X_train, Y_train, y_train, X_test, y_test,
        EPOCHS, BATCH_SIZE, LR, MOMENTUM,
    )

    attn_net, cbam_blocks = build_attention_resnet()
    attn_losses, attn_accs = run_experiment(
        'ResNet + CBAM  (5 residual blocks + CBAM attention)',
        attn_net, X_train, Y_train, y_train, X_test, y_test,
        EPOCHS, BATCH_SIZE, LR, MOMENTUM,
    )

    print(f'\n{"═" * 66}')
    print('  FINAL COMPARISON')
    print(f'{"═" * 66}')
    print(f'  {"Metric":<30}  {"Plain CNN":>10}  {"ResNet":>10}  {"ResNet+CBAM":>11}')
    print(f'  {"-" * 62}')
    print(f'  {"Final train loss":<30}  {plain_losses[-1]:>10.4f}  {res_losses[-1]:>10.4f}  {attn_losses[-1]:>11.4f}')
    print(f'  {"Final test accuracy":<30}  {plain_accs[-1]:>10.4f}  {res_accs[-1]:>10.4f}  {attn_accs[-1]:>11.4f}')
    print(f'  {"Best test accuracy":<30}  {max(plain_accs):>10.4f}  {max(res_accs):>10.4f}  {max(attn_accs):>11.4f}')
    best_plain = plain_accs.index(max(plain_accs)) + 1
    best_res   = res_accs.index(max(res_accs))   + 1
    best_attn  = attn_accs.index(max(attn_accs)) + 1
    print(f'  {"Epoch of best acc":<30}  {best_plain:>10d}  {best_res:>10d}  {best_attn:>11d}')
    print(f'{"═" * 66}')

    print('\n  Per-epoch results:')
    print(f'  {"Ep":>3}  {"PlainLoss":>9}  {"ResLoss":>8}  {"AttnLoss":>9}  '
          f'{"PlainAcc":>9}  {"ResAcc":>7}  {"AttnAcc":>8}')
    print(f'  {"-" * 62}')
    for i in range(EPOCHS):
        best_acc = max(plain_accs[i], res_accs[i], attn_accs[i])
        marker = ' <attn' if attn_accs[i] == best_acc and attn_accs[i] > plain_accs[i] else ''
        print(f'  {i+1:>3}  {plain_losses[i]:>9.4f}  {res_losses[i]:>8.4f}  {attn_losses[i]:>9.4f}  '
              f'{plain_accs[i]:>9.4f}  {res_accs[i]:>7.4f}  {attn_accs[i]:>8.4f}{marker}')

    vis_indices = []
    for digit in range(10):
        idxs = np.where(y_test == digit)[0]
        if len(idxs) > 0:
            vis_indices.append(idxs[0])
    X_vis = X_test[vis_indices]
    y_vis = y_test[vis_indices]

    attn_vis_path = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                                  'attention_maps.png')
    visualize_attention(attn_net, cbam_blocks, X_vis, y_vis, attn_vis_path)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        epochs_range = range(1, EPOCHS + 1)

        axes[0].plot(epochs_range, plain_losses, 'b-o', label='Plain CNN',   markersize=4)
        axes[0].plot(epochs_range, res_losses,   'r-o', label='ResNet',      markersize=4)
        axes[0].plot(epochs_range, attn_losses,  'g-o', label='ResNet+CBAM', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, plain_accs, 'b-o', label='Plain CNN',   markersize=4)
        axes[1].plot(epochs_range, res_accs,   'r-o', label='ResNet',      markersize=4)
        axes[1].plot(epochs_range, attn_accs,  'g-o', label='ResNet+CBAM', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title('Test Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                                 'attention_comparison.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f'  Comparison plot saved to {os.path.normpath(out_path)}')
    except ImportError:
        pass

    print('\nDone.')


if __name__ == '__main__':
    main()
