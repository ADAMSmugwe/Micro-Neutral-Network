"""
Day 24: Model Quantization – Baseline vs PTQ vs QAT on MNIST.

Three pipelines are compared:

  Baseline  – standard FP32 MLP trained to convergence
  PTQ       – same trained weights converted to INT8 (post-training)
  QAT       – MLP trained with FakeQuantize layers, then converted to INT8

Outputs
-------
  Console   – per-epoch progress + comparison table
  outputs/quantization_comparison.png  – accuracy & quantization-error curves
"""
import sys
import os
import time
import struct
import gzip
import urllib.request

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layers import Layer
from src.network import Network
from src.quantization import (
    Quantizer, FakeQuantize,
    quantize_network, infer,
    model_memory_bytes, quantized_memory_bytes,
)


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

    X_tr = read_images(_download(FILES['train_images'])).astype(np.float32) / 255.0
    y_tr = read_labels(_download(FILES['train_labels']))
    X_te = read_images(_download(FILES['test_images'])).astype(np.float32) / 255.0
    y_te = read_labels(_download(FILES['test_labels']))
    return X_tr, y_tr, X_te, y_te


def one_hot(y, n=10):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out



def net_accuracy(net, X, y_int, batch=256):
    net.eval_mode()
    preds = []
    for i in range(0, len(X), batch):
        preds.append(np.argmax(net.forward(X[i:i + batch]), axis=1))
    return float(np.mean(np.concatenate(preds) == y_int))


def q_accuracy(q_layers, X, y_int, batch=256):
    preds = []
    for i in range(0, len(X), batch):
        preds.append(np.argmax(infer(q_layers, X[i:i + batch]), axis=1))
    return float(np.mean(np.concatenate(preds) == y_int))



def build_baseline():
    """Standard FP32 MLP: 784 → 256 → 128 → 10."""
    net = Network()
    net.add_layer(Layer(784, 256, 'relu'))
    net.add_layer(Layer(256, 128, 'relu'))
    net.add_layer(Layer(128,  10, 'softmax'))
    net.set_loss('cross_entropy')
    return net


def build_qat():
    """Same MLP with FakeQuantize layers before every dense layer.

    The FakeQuantize layers simulate INT8 noise during training so the
    weights learn to be robust to quantization error (STE for gradients).

    Input → FakeQ → Dense → FakeQ → Dense → FakeQ → Dense → output
    """
    net = Network()
    net.add_layer(FakeQuantize())
    net.add_layer(Layer(784, 256, 'relu'))
    net.add_layer(FakeQuantize())
    net.add_layer(Layer(256, 128, 'relu'))
    net.add_layer(FakeQuantize())
    net.add_layer(Layer(128,  10, 'softmax'))
    net.set_loss('cross_entropy')
    return net



def train(net, X_tr, Y_tr, X_te, y_te, epochs, lr, momentum, batch_size, tag):
    print(f'\n{"─" * 56}')
    print(f'  {tag}')
    print(f'{"─" * 56}')
    train_losses, test_accs = [], []

    for epoch in range(1, epochs + 1):
        net.train_mode()
        t0   = time.time()
        perm = np.random.permutation(len(X_tr))
        Xs, Ys = X_tr[perm], Y_tr[perm]
        batch_losses = []

        for i in range(0, len(Xs), batch_size):
            Xb, Yb = Xs[i:i + batch_size], Ys[i:i + batch_size]
            pred = net.forward(Xb)
            batch_losses.append(float(net.loss(Yb, pred)))
            net.backward(Yb, pred)
            net.update(lr=lr, momentum=momentum)

        loss = float(np.mean(batch_losses))
        acc  = net_accuracy(net, X_te, y_te)
        elapsed = time.time() - t0
        train_losses.append(loss)
        test_accs.append(acc)
        print(f'  Epoch {epoch:2d}/{epochs}  loss={loss:.4f}  test_acc={acc:.4f}  ({elapsed:.2f}s)')

    return train_losses, test_accs



def weight_quantization_error(net):
    """Mean absolute reconstruction error across all dense weight matrices."""
    errors = []
    for layer in net.layers:
        if not hasattr(layer, 'weights'):
            continue
        W = layer.weights
        s, zp = Quantizer.get_scale_and_zero_point(W)
        W_q   = Quantizer.quantize(W, s, zp)
        W_rec = Quantizer.dequantize(W_q, s, zp)
        errors.append(float(np.mean(np.abs(W - W_rec))))
    return float(np.mean(errors)) if errors else 0.0



def main():
    print('Loading MNIST...')
    X_tr_full, y_tr_full, X_te_full, y_te_full = load_mnist()

    np.random.seed(42)
    N_TRAIN, N_TEST = 8000, 2000
    idx = np.random.permutation(len(X_tr_full))

    X_tr = X_tr_full[idx[:N_TRAIN]].reshape(-1, 784)
    y_tr = y_tr_full[idx[:N_TRAIN]]
    X_te = X_te_full[:N_TEST].reshape(-1, 784)
    y_te = y_te_full[:N_TEST]
    Y_tr = one_hot(y_tr)

    print(f'Train: {X_tr.shape}  Test: {X_te.shape}')

    EPOCHS     = 15
    BATCH_SIZE = 128
    LR         = 0.01
    MOMENTUM   = 0.9

    print(f'\nHyperparameters: epochs={EPOCHS}, batch={BATCH_SIZE}, '
          f'lr={LR}, momentum={MOMENTUM}')

    baseline_net = build_baseline()
    bl_losses, bl_accs = train(
        baseline_net, X_tr, Y_tr, X_te, y_te,
        EPOCHS, LR, MOMENTUM, BATCH_SIZE,
        'Baseline  (FP32 MLP: 784→256→128→10)',
    )
    baseline_acc = net_accuracy(baseline_net, X_te, y_te)
    baseline_mb  = model_memory_bytes(baseline_net) / 1024 / 1024

    print(f'\n{"─" * 56}')
    print('  Applying PTQ (per-channel INT8) to baseline weights ...')
    print(f'{"─" * 56}')
    ptq_layers  = quantize_network(baseline_net, per_channel=True)
    ptq_acc     = q_accuracy(ptq_layers, X_te, y_te)
    ptq_mb      = quantized_memory_bytes(ptq_layers) / 1024 / 1024
    ptq_drop    = baseline_acc - ptq_acc

    qat_net = build_qat()
    qat_losses, qat_accs = train(
        qat_net, X_tr, Y_tr, X_te, y_te,
        EPOCHS, LR, MOMENTUM, BATCH_SIZE,
        'QAT  (FP32 MLP + FakeQuantize, then INT8)',
    )
    qat_fp32_acc = net_accuracy(qat_net, X_te, y_te)

    print(f'\n{"─" * 56}')
    print('  Applying PTQ to QAT-trained weights ...')
    print(f'{"─" * 56}')
    qat_q_layers = quantize_network(qat_net, per_channel=True)
    qat_q_acc    = q_accuracy(qat_q_layers, X_te, y_te)
    qat_mb       = quantized_memory_bytes(qat_q_layers) / 1024 / 1024
    qat_drop     = qat_fp32_acc - qat_q_acc

    bl_err  = weight_quantization_error(baseline_net)
    qat_err = weight_quantization_error(qat_net)

    print(f'\n{"═" * 68}')
    print('  FINAL COMPARISON')
    print(f'{"═" * 68}')
    print(f'  {"Metric":<34}  {"Baseline":>10}  {"PTQ":>8}  {"QAT+PTQ":>9}')
    print(f'  {"-" * 64}')
    print(f'  {"Test accuracy (FP32)":<34}  {baseline_acc:>10.4f}  {"—":>8}  {qat_fp32_acc:>9.4f}')
    print(f'  {"Test accuracy (after INT8)":<34}  {"—":>10}  {ptq_acc:>8.4f}  {qat_q_acc:>9.4f}')
    print(f'  {"Accuracy drop vs own FP32":<34}  {"0.0000":>10}  {ptq_drop:>8.4f}  {qat_drop:>9.4f}')
    print(f'  {"Model size (MB)":<34}  {baseline_mb:>10.4f}  {ptq_mb:>8.4f}  {qat_mb:>9.4f}')
    print(f'  {"Size reduction":<34}  {"1.00×":>10}  '
          f'{baseline_mb/ptq_mb:>7.2f}×  {baseline_mb/qat_mb:>8.2f}×')
    print(f'  {"Mean weight quant error":<34}  {bl_err:>10.6f}  {"—":>8}  {qat_err:>9.6f}')
    print(f'{"═" * 68}')

    print(f'\n  Key takeaways:')
    print(f'    • PTQ accuracy drop:     {ptq_drop:+.4f}  '
          f'({"within" if abs(ptq_drop) < 0.02 else "exceeds"} 2% target)')
    print(f'    • QAT+PTQ accuracy drop: {qat_drop:+.4f}  '
          f'({"better" if qat_drop < ptq_drop else "similar"} than plain PTQ)')
    print(f'    • Memory saved by INT8:  ~{baseline_mb - ptq_mb:.3f} MB  '
          f'({100*(1 - ptq_mb/baseline_mb):.1f}% reduction)')

    print(f'\n  INT8 Arithmetic Walkthrough')
    print(f'  {"─" * 44}')
    W_sample = baseline_net.layers[0].weights[:5, :5]
    s, zp    = Quantizer.get_scale_and_zero_point(W_sample)
    W_q      = Quantizer.quantize(W_sample, s, zp)
    W_rec    = Quantizer.dequantize(W_q, s, zp)
    mae      = np.mean(np.abs(W_sample - W_rec))
    print(f'  Sample 5×5 weight block (FP32):\n  {W_sample}')
    print(f'\n  Quantized to INT8 (scale={s:.6f}, zero_point={zp}):')
    print(f'  {W_q}')
    print(f'\n  Reconstructed (dequantized):')
    print(f'  {np.round(W_rec, 6)}')
    print(f'\n  Mean absolute error: {mae:.8f}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs_range = range(1, EPOCHS + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs_range, bl_losses,  'b-o', label='Baseline FP32', markersize=4)
        axes[0].plot(epochs_range, qat_losses, 'g-o', label='QAT FP32',      markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, bl_accs,  'b-o', label='Baseline FP32', markersize=4)
        axes[1].plot(epochs_range, qat_accs, 'g-o', label='QAT FP32',      markersize=4)
        axes[1].axhline(ptq_acc,   color='b', linestyle='--', linewidth=1.2,
                        label=f'Baseline PTQ  ({ptq_acc:.4f})')
        axes[1].axhline(qat_q_acc, color='g', linestyle='--', linewidth=1.2,
                        label=f'QAT PTQ  ({qat_q_acc:.4f})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title('Test Accuracy (dashed = after INT8 quantization)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        labels = ['Baseline\n(FP32)', 'PTQ\n(INT8 weights)', 'QAT + PTQ\n(INT8 weights)']
        sizes  = [baseline_mb, ptq_mb, qat_mb]
        colors = ['#4C72B0', '#DD8452', '#55A868']
        bars   = axes[2].bar(labels, sizes, color=colors, width=0.5, edgecolor='white')
        axes[2].set_ylabel('Model size (MB)')
        axes[2].set_title('Parameter Memory Footprint')
        axes[2].grid(True, alpha=0.3, axis='y')
        for bar, size in zip(bars, sizes):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f'{size:.3f} MB', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                                'quantization_comparison.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f'\n  Plot saved → {os.path.normpath(out_path)}')
    except ImportError:
        pass

    print('\nDone.')


if __name__ == '__main__':
    main()
