"""
Day 20 – Hyperparameter Tuning
================================
Systematic search (grid + random) over key CNN hyperparameters on a
small MNIST subset so the search finishes in a reasonable time.

Results are saved to outputs/tuning/.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import itertools
import struct
import gzip
import urllib.request

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.layers import Conv2D, MaxPool2D, Flatten, ReLU, Layer
from src.network import Network
from src.utils import LearningRateScheduler

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'tuning')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MNIST helpers (shared with other examples)
# ---------------------------------------------------------------------------
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
    X_test  = _read_images(_download(FILES['test_images'])).astype(np.float32)  / 255.0
    y_test  = _read_labels(_download(FILES['test_labels']))
    return X_train, y_train, X_test, y_test

def one_hot(y, n=10):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out

# ---------------------------------------------------------------------------
# Network factory
# ---------------------------------------------------------------------------
def build_network(hidden_size=128, dropout=0.0, reg_lambda=0.0):
    """CNN: Conv(1→8)→ReLU→Pool → Conv(8→16)→ReLU→Pool → Dense→Softmax."""
    net = Network(reg_lambda=reg_lambda)
    net.add_layer(Conv2D(in_channels=1,  out_channels=8,  filter_size=3, stride=1, padding=1))
    net.add_layer(ReLU())
    net.add_layer(MaxPool2D(pool_size=2, stride=2))
    net.add_layer(Conv2D(in_channels=8,  out_channels=16, filter_size=3, stride=1, padding=1))
    net.add_layer(ReLU())
    net.add_layer(MaxPool2D(pool_size=2, stride=2))
    net.add_layer(Flatten())
    # After two 2×2 pools on 28×28 input → 7×7×16 = 784 features
    net.add_layer(Layer(7 * 7 * 16, hidden_size, 'relu', dropout_rate=dropout))
    net.add_layer(Layer(hidden_size, 10, 'softmax', dropout_rate=0.0))
    net.set_loss('cross_entropy')
    return net

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def grid_search(X_train, y_train, X_val, y_val, param_grid,
                epochs=20, patience=4, verbose=False):
    """Try every combination in param_grid and return sorted results list."""
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    results = []

    print(f"\n{'='*60}")
    print(f"Grid search: {len(combos)} combinations × up to {epochs} epochs")
    print(f"{'='*60}")

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        print(f"\n[{i}/{len(combos)}] {params}")

        lr          = params.get('lr',          0.001)
        batch_size  = params.get('batch_size',  64)
        hidden_size = params.get('hidden_size', 128)
        dropout     = params.get('dropout',     0.0)
        optimizer   = params.get('optimizer',   'adam')

        net = build_network(hidden_size=hidden_size, dropout=dropout)
        lr_scheduler = LearningRateScheduler(
            initial_lr=lr, decay_type='step', step_size=8, decay_factor=0.5)

        history = net.train_with_history(
            X_train, y_train, X_val, y_val,
            epochs=epochs, lr=lr, lr_scheduler=lr_scheduler,
            batch_size=batch_size, patience=patience,
            optimizer=optimizer, verbose=verbose,
        )

        best_val_acc = max(history['val_acc'])
        best_epoch   = int(np.argmax(history['val_acc'])) + 1
        elapsed      = time.time() - t0

        result = {
            **params,
            'best_val_acc':  best_val_acc,
            'best_epoch':    best_epoch,
            'epochs_run':    len(history['val_acc']),
            'history':       history,
        }
        results.append(result)
        print(f"  → best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}  "
              f"({elapsed:.1f}s, {len(history['val_acc'])} epochs run)")

    results.sort(key=lambda r: r['best_val_acc'], reverse=True)
    return results

# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------
def random_search(n_trials, X_train, y_train, X_val, y_val,
                  epochs=20, patience=4, seed=None, verbose=False):
    """Sample hyperparameters randomly and return sorted results list."""
    if seed is not None:
        np.random.seed(seed)

    results = []
    print(f"\n{'='*60}")
    print(f"Random search: {n_trials} trials × up to {epochs} epochs")
    print(f"{'='*60}")

    for trial in range(1, n_trials + 1):
        lr          = float(10 ** np.random.uniform(-4, -2))   # 0.0001–0.01
        batch_size  = int(np.random.choice([32, 64, 128]))
        hidden_size = int(np.random.choice([64, 128, 256]))
        dropout     = float(np.random.uniform(0.0, 0.5))
        optimizer   = str(np.random.choice(['sgd', 'adam']))

        params = {
            'lr': lr, 'batch_size': batch_size,
            'hidden_size': hidden_size, 'dropout': dropout,
            'optimizer': optimizer,
        }
        t0 = time.time()
        print(f"\n[Trial {trial}/{n_trials}] "
              f"lr={lr:.5f}  batch={batch_size}  "
              f"hidden={hidden_size}  dropout={dropout:.2f}  opt={optimizer}")

        net = build_network(hidden_size=hidden_size, dropout=dropout)
        lr_scheduler = LearningRateScheduler(
            initial_lr=lr, decay_type='step', step_size=8, decay_factor=0.5)

        history = net.train_with_history(
            X_train, y_train, X_val, y_val,
            epochs=epochs, lr=lr, lr_scheduler=lr_scheduler,
            batch_size=batch_size, patience=patience,
            optimizer=optimizer, verbose=verbose,
        )

        best_val_acc = max(history['val_acc'])
        best_epoch   = int(np.argmax(history['val_acc'])) + 1
        elapsed      = time.time() - t0

        result = {
            **params,
            'best_val_acc': best_val_acc,
            'best_epoch':   best_epoch,
            'epochs_run':   len(history['val_acc']),
            'history':      history,
        }
        results.append(result)
        print(f"  → best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}  "
              f"({elapsed:.1f}s)")

    results.sort(key=lambda r: r['best_val_acc'], reverse=True)
    return results

# ---------------------------------------------------------------------------
# Learning-rate finder (LR range test)
# ---------------------------------------------------------------------------
def find_lr(net, X, y, batch_size=64, start_lr=1e-6, end_lr=1.0,
            num_steps=100, beta=0.98):
    """Geometrically increase LR over num_steps mini-batches; track loss.

    Returns (lrs, smoothed_losses) — plot and pick the LR where loss
    falls fastest (steepest downward slope).
    """
    n    = len(X)
    mult = (end_lr / start_lr) ** (1.0 / max(num_steps - 1, 1))
    lr   = start_lr

    avg_loss  = 0.0
    best_loss = float('inf')
    lrs, losses = [], []

    net.train_mode()
    for step in range(num_steps):
        idx  = np.random.choice(n, min(batch_size, n), replace=False)
        Xb, yb = X[idx], y[idx]

        pred = net.forward(Xb)
        raw_loss = net.loss(yb, pred)
        net.backward(yb, pred)
        net.update(lr=lr)

        # Exponential smoothing to stabilise noisy loss
        avg_loss  = beta * avg_loss + (1 - beta) * raw_loss
        smoothed  = avg_loss / (1 - beta ** (step + 1))

        # Stop early if loss explodes
        if step > 5 and smoothed > 4 * best_loss:
            break
        if smoothed < best_loss:
            best_loss = smoothed

        lrs.append(lr)
        losses.append(smoothed)
        lr *= mult

    return lrs, losses

# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------
def print_results_table(results, title='Results'):
    if not results:
        return
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

    # Collect all param keys (exclude internal keys)
    skip = {'best_val_acc', 'best_epoch', 'epochs_run', 'history'}
    param_keys = [k for k in results[0] if k not in skip]

    # Header
    header = f"{'#':>3}  " + "  ".join(f"{k:>12}" for k in param_keys)
    header += f"  {'val_acc':>8}  {'best_ep':>7}  {'ep_run':>6}"
    print(header)
    print('-' * len(header))

    for rank, r in enumerate(results, 1):
        row = f"{rank:>3}  "
        for k in param_keys:
            v = r[k]
            if isinstance(v, float):
                row += f"  {v:>12.5f}"
            else:
                row += f"  {str(v):>12}"
        row += f"  {r['best_val_acc']:>8.4f}  {r['best_epoch']:>7}  {r['epochs_run']:>6}"
        print(row)

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def plot_search_results(results, prefix='grid'):
    """Bar chart of val_acc per trial + history curves of top-3 configs."""
    if not results:
        return

    n = len(results)

    # ---- Figure 1: sorted bar chart ----
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 5))
    accs   = [r['best_val_acc'] for r in results]
    labels = [str(i + 1) for i in range(n)]
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(n)]
    bars = ax.bar(labels, accs, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_ylim(max(0, min(accs) - 0.05), min(1.0, max(accs) + 0.05))
    ax.set_xlabel('Trial (sorted by val acc)')
    ax.set_ylabel('Best validation accuracy')
    ax.set_title(f'{prefix.capitalize()} search — trial comparison')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{prefix}_comparison.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

    # ---- Figure 2: val_acc history for top-3 ----
    top_k = min(3, len(results))
    fig, axes = plt.subplots(1, top_k, figsize=(5 * top_k, 4), squeeze=False)

    for col, r in enumerate(results[:top_k]):
        ax = axes[0][col]
        h  = r['history']
        ep = range(1, len(h['val_acc']) + 1)
        ax.plot(ep, h['train_loss'], label='train loss', color='#e74c3c')
        ax2 = ax.twinx()
        ax2.plot(ep, h['val_acc'],   label='val acc',   color='#2ecc71', linestyle='--')

        # Annotation of best
        best_ep  = int(np.argmax(h['val_acc']))
        best_acc = h['val_acc'][best_ep]
        ax2.axvline(best_ep + 1, color='gray', linestyle=':', linewidth=1)
        ax2.annotate(f'{best_acc:.3f}', xy=(best_ep + 1, best_acc),
                     xytext=(5, -10), textcoords='offset points', fontsize=8,
                     color='#2ecc71')

        skip = {'best_val_acc', 'best_epoch', 'epochs_run', 'history'}
        param_str = ', '.join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r.items() if k not in skip
        )
        ax.set_title(f'Rank {col+1}\n{param_str}', fontsize=7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train loss', color='#e74c3c')
        ax2.set_ylabel('Val accuracy', color='#2ecc71')

    fig.suptitle(f'{prefix.capitalize()} search — top-{top_k} training histories')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{prefix}_top_histories.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


def plot_sensitivity(results, param_key, prefix='grid'):
    """Scatter plot of one hyperparameter vs best val_acc."""
    vals = [r[param_key] for r in results]
    accs = [r['best_val_acc'] for r in results]
    if len(set(vals)) < 2:
        return  # nothing to compare

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(vals, accs, alpha=0.7, edgecolors='k', linewidths=0.5, s=60)
    for v, a in zip(vals, accs):
        ax.annotate(f'{a:.3f}', xy=(v, a), xytext=(4, 4),
                    textcoords='offset points', fontsize=7)

    if param_key == 'lr':
        ax.set_xscale('log')
    ax.set_xlabel(param_key)
    ax.set_ylabel('Best validation accuracy')
    ax.set_title(f'Sensitivity: {param_key} vs val_acc')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{prefix}_sensitivity_{param_key}.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


def plot_lr_finder(lrs, losses):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lrs, losses, color='#3498db', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate (log scale)')
    ax.set_ylabel('Smoothed loss')
    ax.set_title('LR Finder — pick LR where loss falls fastest')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'lr_finder.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # ---- Load a small MNIST subset ----------------------------------------
    print('Loading MNIST...')
    X_full, y_full, X_test_full, y_test_full = load_mnist()

    np.random.seed(42)
    N_TRAIN = 3000   # keep small so search finishes fast
    N_VAL   = 500
    N_TEST  = 1000

    perm = np.random.permutation(len(X_full))
    X_train_raw = X_full[perm[:N_TRAIN]].reshape(-1, 28, 28, 1)
    y_train_int = y_full[perm[:N_TRAIN]]
    X_val_raw   = X_full[perm[N_TRAIN:N_TRAIN + N_VAL]].reshape(-1, 28, 28, 1)
    y_val_int   = y_full[perm[N_TRAIN:N_TRAIN + N_VAL]]
    X_test      = X_test_full[:N_TEST].reshape(-1, 28, 28, 1)
    y_test_int  = y_test_full[:N_TEST]

    Y_train = one_hot(y_train_int)   # one-hot for cross-entropy
    Y_val   = one_hot(y_val_int)
    Y_test  = one_hot(y_test_int)

    print(f'Train: {X_train_raw.shape}  Val: {X_val_raw.shape}  Test: {X_test.shape}')

    # =========================================================================
    # 1. Baseline (no tuning)
    # =========================================================================
    print('\n--- Baseline (lr=0.01, batch=64, hidden=128, no dropout, adam) ---')
    baseline_net = build_network(hidden_size=128, dropout=0.0)
    baseline_sched = LearningRateScheduler(initial_lr=0.01, decay_type='step',
                                            step_size=8, decay_factor=0.5)
    baseline_history = baseline_net.train_with_history(
        X_train_raw, Y_train, X_val_raw, Y_val,
        epochs=20, lr=0.01, lr_scheduler=baseline_sched,
        batch_size=64, patience=5, optimizer='adam', verbose=True,
    )
    baseline_acc = max(baseline_history['val_acc'])
    print(f'\nBaseline best val_acc: {baseline_acc:.4f}')

    # =========================================================================
    # 2. Grid search
    # =========================================================================
    param_grid = {
        'lr':          [0.01, 0.001],
        'batch_size':  [64, 128],
        'hidden_size': [64, 128],
        'dropout':     [0.0, 0.2],
    }
    # 2×2×2×2 = 16 combinations. verbose=False so only summaries print.
    grid_results = grid_search(
        X_train_raw, Y_train, X_val_raw, Y_val,
        param_grid=param_grid, epochs=20, patience=4, verbose=False,
    )
    print_results_table(grid_results, title='Grid search results (sorted by val_acc)')
    plot_search_results(grid_results, prefix='grid')
    for key in param_grid:
        plot_sensitivity(grid_results, key, prefix='grid')

    best_grid = grid_results[0]
    print(f'\nBest grid config: val_acc={best_grid["best_val_acc"]:.4f}')
    print(f'  Params: lr={best_grid["lr"]}, batch={best_grid["batch_size"]}, '
          f'hidden={best_grid["hidden_size"]}, dropout={best_grid["dropout"]}')

    # =========================================================================
    # 3. Random search
    # =========================================================================
    rand_results = random_search(
        n_trials=8,
        X_train=X_train_raw, y_train=Y_train,
        X_val=X_val_raw,   y_val=Y_val,
        epochs=20, patience=4, seed=0, verbose=False,
    )
    print_results_table(rand_results, title='Random search results (sorted by val_acc)')
    plot_search_results(rand_results, prefix='random')
    plot_sensitivity(rand_results, 'lr', prefix='random')

    best_rand = rand_results[0]
    print(f'\nBest random config: val_acc={best_rand["best_val_acc"]:.4f}')

    # =========================================================================
    # 4. LR Finder (on a fresh network with best grid settings)
    # =========================================================================
    print('\n--- LR Finder ---')
    lr_net = build_network(hidden_size=best_grid['hidden_size'],
                           dropout=best_grid['dropout'])
    lrs, losses = find_lr(lr_net, X_train_raw, Y_train,
                          batch_size=64, start_lr=1e-5, end_lr=1.0,
                          num_steps=80)
    plot_lr_finder(lrs, losses)

    # =========================================================================
    # 5. Train best config fully and evaluate on test set
    # =========================================================================
    print('\n--- Training best config on full train+val, evaluating on test ---')
    # Combine train + val for final training
    X_all = np.concatenate([X_train_raw, X_val_raw], axis=0)
    Y_all = np.concatenate([Y_train, Y_val], axis=0)

    final_net   = build_network(hidden_size=best_grid['hidden_size'],
                                dropout=best_grid['dropout'])
    final_sched = LearningRateScheduler(
        initial_lr=best_grid['lr'], decay_type='step',
        step_size=8, decay_factor=0.5)

    # Use val set as a dummy for early stopping even during "final" training
    final_history = final_net.train_with_history(
        X_all, Y_all, X_test, Y_test,
        epochs=30, lr=best_grid['lr'], lr_scheduler=final_sched,
        batch_size=best_grid['batch_size'], patience=5,
        optimizer=best_grid.get('optimizer', 'adam'), verbose=True,
    )

    final_net.eval_mode()
    test_pred = final_net.forward(X_test)
    test_acc  = final_net.accuracy(Y_test, test_pred)

    print(f'\n{"="*60}')
    print(f' SUMMARY')
    print(f'{"="*60}')
    print(f'  Baseline val_acc :  {baseline_acc:.4f}')
    print(f'  Best grid val_acc:  {best_grid["best_val_acc"]:.4f}')
    print(f'  Best rand val_acc:  {best_rand["best_val_acc"]:.4f}')
    print(f'  Final test_acc   :  {test_acc:.4f}')
    print(f'\nAll plots saved to {os.path.abspath(OUT_DIR)}')
