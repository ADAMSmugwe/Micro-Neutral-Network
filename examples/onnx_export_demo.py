"""
Day 27 – ONNX Export Demo
==========================
Train a small CNN on MNIST, export it to the industry-standard ONNX format,
and verify that ONNX Runtime produces identical predictions.

Requirements
------------
    pip install onnx onnxruntime

Run
---
    python examples/onnx_export_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── dependency check ─────────────────────────────────────────────────────────
try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Missing dependencies.  Install with:\n  pip install onnx onnxruntime")
    sys.exit(1)

import gzip
import struct
import urllib.request
from collections import Counter

import numpy as np

from src.layers import Conv2D, MaxPool2D, Flatten, ReLU, Layer
from src.network import Network
from src.onnx_export import ONNXExporter


# ── data ─────────────────────────────────────────────────────────────────────

CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist')
URLS  = {
    'train_images': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    'test_images':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
}


def _download(key):
    os.makedirs(CACHE, exist_ok=True)
    fname = URLS[key].split('/')[-1]
    path  = os.path.join(CACHE, fname)
    if not os.path.exists(path):
        print(f'  Downloading {fname} ...', end=' ', flush=True)
        urllib.request.urlretrieve(URLS[key], path)
        print('done')
    return path


def load_mnist():
    with gzip.open(_download('train_images'), 'rb') as f:
        _, n, r, c = struct.unpack('>IIII', f.read(16))
        X_tr = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c).astype(np.float32) / 255.0
    with gzip.open(_download('train_labels'), 'rb') as f:
        struct.unpack('>II', f.read(8))
        y_tr = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(_download('test_images'), 'rb') as f:
        _, n, r, c = struct.unpack('>IIII', f.read(16))
        X_te = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c).astype(np.float32) / 255.0
    with gzip.open(_download('test_labels'), 'rb') as f:
        struct.unpack('>II', f.read(8))
        y_te = np.frombuffer(f.read(), dtype=np.uint8)
    return X_tr, y_tr, X_te, y_te


def one_hot(y, n=10):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


# ── model ────────────────────────────────────────────────────────────────────

def build_cnn():
    """Conv(1→8) → Pool → Conv(8→16) → Pool → Flatten → Dense(128) → Dense(10)."""
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


# ── training ─────────────────────────────────────────────────────────────────

def train(net, X_tr, Y_tr, X_te, y_te, epochs=8, lr=0.001, batch=128):
    n = len(X_tr)
    for ep in range(epochs):
        net.train_mode()
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            Xb = X_tr[perm[i:i + batch]]
            Yb = Y_tr[perm[i:i + batch]]
            pred = net.forward(Xb)
            net.backward(Yb, pred)
            net.update(lr=lr, optimizer='adam')
        net.eval_mode()
        acc = np.mean(np.argmax(net.forward(X_te), axis=1) == y_te)
        print(f'  Epoch {ep + 1}/{epochs}  test acc: {acc:.3f}')
    return acc


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    print('=== Day 27: ONNX Export ===\n')

    # ── data ─────────────────────────────────────────────────────────────────
    print('Loading MNIST...')
    X_tr_full, y_tr_full, X_te_full, y_te_full = load_mnist()

    N_TRAIN, N_TEST = 8_000, 1_000
    idx   = np.random.permutation(len(X_tr_full))[:N_TRAIN]
    X_tr  = X_tr_full[idx].reshape(-1, 28, 28, 1)   # NHWC – our model format
    Y_tr  = one_hot(y_tr_full[idx])
    X_te  = X_te_full[:N_TEST].reshape(-1, 28, 28, 1)
    y_te  = y_te_full[:N_TEST]
    print(f'  train: {X_tr.shape}  test: {X_te.shape}')

    # ── train ─────────────────────────────────────────────────────────────────
    print(f'\nTraining CNN ({N_TRAIN} samples, 8 epochs)...')
    net = build_cnn()
    final_acc = train(net, X_tr, Y_tr, X_te, y_te)
    print(f'\nNumPy model test accuracy: {final_acc:.3f}')

    # ── export ───────────────────────────────────────────────────────────────
    out_dir   = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, 'mnist_cnn.onnx')

    print('\nExporting to ONNX...')
    net.eval_mode()
    exporter = ONNXExporter(net, input_shape=(1, 28, 28))   # CHW – ONNX convention
    exporter.export(onnx_path, output_classes=10)

    # ── verify ───────────────────────────────────────────────────────────────
    print('\nVerifying predictions...')

    # ONNX Runtime expects NCHW; our model uses NHWC
    X_te_nchw  = X_te.transpose(0, 3, 1, 2)           # (N, 1, 28, 28)

    numpy_out  = net.forward(X_te)                     # NHWC input
    onnx_out   = ONNXExporter.run(onnx_path, X_te_nchw)  # NCHW input

    max_diff   = float(np.max(np.abs(onnx_out - numpy_out)))
    match      = np.allclose(onnx_out, numpy_out, atol=1e-4)

    print(f'  Max absolute difference : {max_diff:.2e}')
    print(f'  Predictions match (1e-4): {match}')

    numpy_acc  = float(np.mean(np.argmax(numpy_out, axis=1) == y_te))
    onnx_acc   = float(np.mean(np.argmax(onnx_out,  axis=1) == y_te))
    print(f'\n  NumPy model accuracy    : {numpy_acc:.4f}')
    print(f'  ONNX  model accuracy    : {onnx_acc:.4f}')

    if match:
        print('\n  ✓ Export successful – outputs are numerically identical.')
    else:
        print('\n  ✗ Mismatch detected – check layer ordering and weight transpositions.')

    # ── model info ───────────────────────────────────────────────────────────
    print('\n--- ONNX model info ---')
    onnx_model = onnx.load(onnx_path)
    print(f'  IR version   : {onnx_model.ir_version}')
    print(f'  Opset        : {onnx_model.opset_import[0].version}')
    print(f'  Graph nodes  : {len(onnx_model.graph.node)}')
    print(f'  Initializers : {len(onnx_model.graph.initializer)}')

    total_params = sum(
        int(np.prod(list(t.dims))) for t in onnx_model.graph.initializer
    )
    file_kb = os.path.getsize(onnx_path) / 1024
    print(f'  Parameters   : {total_params:,}')
    print(f'  File size    : {file_kb:.1f} KB')

    node_types = Counter(node.op_type for node in onnx_model.graph.node)
    print('\n  Node breakdown:')
    for op, count in sorted(node_types.items()):
        print(f'    {op:<22} x{count}')

    # ── single-sample demo ───────────────────────────────────────────────────
    print('\n--- Single-sample inference ---')
    sample_nhwc  = X_te[:1]                            # (1, 28, 28, 1)
    sample_nchw  = sample_nhwc.transpose(0, 3, 1, 2)   # (1, 1, 28, 28)

    numpy_probs  = net.forward(sample_nhwc)[0]
    onnx_probs   = ONNXExporter.run(onnx_path, sample_nchw)[0]

    true_label   = y_te[0]
    numpy_pred   = int(np.argmax(numpy_probs))
    onnx_pred    = int(np.argmax(onnx_probs))

    print(f'  True label   : {true_label}')
    print(f'  NumPy pred   : {numpy_pred}  (conf {numpy_probs[numpy_pred]:.3f})')
    print(f'  ONNX  pred   : {onnx_pred}  (conf {onnx_probs[onnx_pred]:.3f})')

    print(f'\nTo visualize the graph, run:')
    print(f'  pip install netron')
    print(f'  netron {os.path.normpath(onnx_path)}')
    print(f'\nDone.')


if __name__ == '__main__':
    main()
