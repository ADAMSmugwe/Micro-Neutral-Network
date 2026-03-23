"""
Day 18: Data Augmentation Demo
Trains two identical CNNs on a 5000-sample MNIST subset:
  - baseline: no augmentation
  - augmented: random rotation, shift, and zoom applied per batch
Then plots validation loss curves to compare generalization.
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
from src.augmentation import DataAugmentor



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


def accuracy(net, X, y_int, batch_size=256):
    net.eval_mode()
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(np.argmax(net.forward(X[i:i + batch_size]), axis=1))
    return np.mean(np.concatenate(preds) == y_int)


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


def run_training(net, X_train, Y_train, X_val, Y_val, epochs, batch_size, lr, augmentor=None, label=''):
    print(f'\n--- {label} ---')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        t0 = time.time()
        net.train_mode()
        perm = np.random.permutation(len(X_train))
        X_s, Y_s = X_train[perm], Y_train[perm]

        for i in range(0, len(X_train), batch_size):
            Xb = X_s[i:i + batch_size]
            Yb = Y_s[i:i + batch_size]
            if augmentor is not None:
                Xb, Yb = augmentor.apply(Xb, Yb)
            pred = net.forward(Xb)
            net.backward(Yb, pred)
            net.update(lr=lr)

        net.eval_mode()
        train_pred = net.forward(X_train)
        val_pred   = net.forward(X_val)
        t_loss = net.loss(Y_train, train_pred)
        v_loss = net.loss(Y_val,   val_pred)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        train_acc = accuracy(net, X_train, np.argmax(Y_train, axis=1))
        val_acc   = accuracy(net, X_val,   np.argmax(Y_val,   axis=1))
        print(f'  Epoch {epoch + 1}/{epochs}  '
              f'train loss: {t_loss:.4f}  val loss: {v_loss:.4f}  '
              f'train acc: {train_acc:.3f}  val acc: {val_acc:.3f}  '
              f'({time.time() - t0:.1f}s)')

    return train_losses, val_losses



print('Loading MNIST...')
X_train_full, y_train_full, X_test_full, y_test_full = load_mnist()

np.random.seed(42)
N_TRAIN, N_VAL = 5000, 1000

idx = np.random.permutation(len(X_train_full))
X_train = X_train_full[idx[:N_TRAIN]].reshape(-1, 28, 28, 1)
y_train = y_train_full[idx[:N_TRAIN]]
X_val   = X_test_full[:N_VAL].reshape(-1, 28, 28, 1)
y_val   = y_test_full[:N_VAL]

Y_train = one_hot(y_train)
Y_val   = one_hot(y_val)

print(f'Train: {X_train.shape}  Val: {X_val.shape}')

EPOCHS     = 10
BATCH_SIZE = 128
LR         = 0.01

np.random.seed(0)
net_baseline  = build_cnn()
np.random.seed(0)
net_augmented = build_cnn()

augmentor = DataAugmentor(rotation_range=15, shift_range=2, zoom_range=0.1, horizontal_flip=False)

baseline_train, baseline_val = run_training(
    net_baseline, X_train, Y_train, X_val, Y_val,
    EPOCHS, BATCH_SIZE, LR,
    augmentor=None, label='Baseline (no augmentation)'
)

augmented_train, augmented_val = run_training(
    net_augmented, X_train, Y_train, X_val, Y_val,
    EPOCHS, BATCH_SIZE, LR,
    augmentor=augmentor, label='Augmented'
)

print('\n=== Final Results ===')
print(f'  Baseline  — train loss: {baseline_train[-1]:.4f}  val loss: {baseline_val[-1]:.4f}'
      f'  val acc: {accuracy(net_baseline,  X_val, y_val):.3f}')
print(f'  Augmented — train loss: {augmented_train[-1]:.4f}  val loss: {augmented_val[-1]:.4f}'
      f'  val acc: {accuracy(net_augmented, X_val, y_val):.3f}')

try:
    import matplotlib.pyplot as plt

    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, baseline_train,  label='Baseline train')
    axes[0].plot(epochs_range, augmented_train, label='Augmented train')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs_range, baseline_val,  label='Baseline val')
    axes[1].plot(epochs_range, augmented_val, label='Augmented val')
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'augmentation_curves.png')
    plt.savefig(out_path)
    print(f'\nLoss curves saved to {out_path}')
    plt.show()
except ImportError:
    print('\n(matplotlib not available — skipping plot)')

try:
    import matplotlib.pyplot as plt

    n_show = 8
    sample = X_train[:n_show]
    aug_sample, _ = augmentor.apply(sample, np.zeros(n_show))

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.5, 3))
    for j in range(n_show):
        axes[0, j].imshow(sample[j, :, :, 0], cmap='gray')
        axes[0, j].axis('off')
        axes[1, j].imshow(aug_sample[j, :, :, 0], cmap='gray')
        axes[1, j].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=9)
    axes[1, 0].set_ylabel('Augmented', fontsize=9)
    plt.suptitle('Sample augmentations')
    plt.tight_layout()
    vis_path = os.path.join(os.path.dirname(__file__), 'augmentation_samples.png')
    plt.savefig(vis_path)
    print(f'Sample visualisation saved to {vis_path}')
    plt.show()
except ImportError:
    pass
