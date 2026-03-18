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
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(np.argmax(net.forward(X[i:i + batch_size]), axis=1))
    return np.mean(np.concatenate(preds) == y_int)


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

EPOCHS     = 5
BATCH_SIZE = 128
LR         = 0.01

print(f'\nArchitecture: Conv(1→8) → ReLU → Pool → Conv(8→16) → ReLU → Pool → Flatten → Dense(784→128) → Dense(128→10,softmax)')
print(f'Training {N_TRAIN} samples for {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}\n')

for epoch in range(EPOCHS):
    t0 = time.time()
    net.train_mode()
    perm = np.random.permutation(N_TRAIN)
    X_s, Y_s = X_train[perm], Y_train[perm]

    for i in range(0, N_TRAIN, BATCH_SIZE):
        Xb = X_s[i:i + BATCH_SIZE]
        Yb = Y_s[i:i + BATCH_SIZE]
        pred = net.forward(Xb)
        net.backward(Yb, pred)
        net.update(lr=LR)

    net.eval_mode()
    train_acc = accuracy(net, X_train, y_train)
    test_acc  = accuracy(net, X_test,  y_test)
    elapsed   = time.time() - t0
    print(f'Epoch {epoch + 1}/{EPOCHS}  train acc: {train_acc:.3f}  test acc: {test_acc:.3f}  ({elapsed:.1f}s)')

print('\nDone.')
