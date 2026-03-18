import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.layers import Conv2D

np.random.seed(42)

print("=" * 60)
print("1. Shape verification")
print("=" * 60)

batch_size, H, W, in_c = 2, 8, 8, 3
out_c, f = 4, 3

X = np.random.randn(batch_size, H, W, in_c)
conv = Conv2D(in_channels=in_c, out_channels=out_c, filter_size=f, stride=1, padding=0)

out_im2col = conv.forward(X, use_im2col=True)
out_naive  = conv.forward(X, use_im2col=False)

expected_h = (H - f) // 1 + 1
expected_w = (W - f) // 1 + 1

print(f"Input shape      : {X.shape}")
print(f"Filter shape     : {conv.filters.shape}")
print(f"Output shape     : {out_im2col.shape}  (expected {(batch_size, expected_h, expected_w, out_c)})")
assert out_im2col.shape == (batch_size, expected_h, expected_w, out_c), "Shape mismatch!"
print("Shape check      : PASSED")

print()
print("=" * 60)
print("2. Naive vs im2col numerical agreement")
print("=" * 60)

max_diff = np.max(np.abs(out_im2col - out_naive))
print(f"Max absolute diff: {max_diff:.2e}  (should be < 1e-10)")
assert max_diff < 1e-10, f"Mismatch between naive and im2col: {max_diff}"
print("Agreement check  : PASSED")

print()
print("=" * 60)
print("3. Stride = 2")
print("=" * 60)

conv_s2 = Conv2D(in_channels=in_c, out_channels=out_c, filter_size=f, stride=2, padding=0)
out_s2 = conv_s2.forward(X)
exp_h2 = (H - f) // 2 + 1
exp_w2 = (W - f) // 2 + 1
print(f"Output shape (stride=2): {out_s2.shape}  (expected {(batch_size, exp_h2, exp_w2, out_c)})")
assert out_s2.shape == (batch_size, exp_h2, exp_w2, out_c), "Stride shape mismatch!"
print("Stride check     : PASSED")

print()
print("=" * 60)
print("4. Padding = 1  (output same spatial size as input)")
print("=" * 60)

conv_p = Conv2D(in_channels=in_c, out_channels=out_c, filter_size=f, stride=1, padding=1)
out_p = conv_p.forward(X)
print(f"Output shape (pad=1): {out_p.shape}  (expected {(batch_size, H, W, out_c)})")
assert out_p.shape == (batch_size, H, W, out_c), "Padding shape mismatch!"
print("Padding check    : PASSED")

print()
print("=" * 60)
print("5. Backward pass – gradient shape check")
print("=" * 60)

conv_bwd = Conv2D(in_channels=in_c, out_channels=out_c, filter_size=f, stride=1, padding=0)
out_bwd = conv_bwd.forward(X, use_im2col=False)
d_out = np.random.randn(*out_bwd.shape)
d_X = conv_bwd.backward(d_out)

print(f"d_out shape   : {d_out.shape}")
print(f"d_filters shape: {conv_bwd.d_filters.shape}  (expected {conv_bwd.filters.shape})")
print(f"d_biases shape : {conv_bwd.d_biases.shape}")
print(f"d_X shape     : {d_X.shape}  (expected {X.shape})")
assert conv_bwd.d_filters.shape == conv_bwd.filters.shape, "d_filters shape mismatch!"
assert d_X.shape == X.shape, "d_X shape mismatch!"
print("Backward check : PASSED")

print()
print("=" * 60)
print("6. Numerical gradient check (d_filters)")
print("=" * 60)

eps = 1e-5
conv_gc = Conv2D(in_channels=1, out_channels=1, filter_size=2, stride=1, padding=0)
X_small = np.random.randn(1, 4, 4, 1)

def fwd_sum(conv, X):
    return np.sum(conv.forward(X, use_im2col=False))

out_gc = conv_gc.forward(X_small, use_im2col=False)
d_out_gc = np.ones_like(out_gc)
conv_gc.backward(d_out_gc)
analytic_grad = conv_gc.d_filters.copy()

num_grad = np.zeros_like(conv_gc.filters)
for idx in np.ndindex(conv_gc.filters.shape):
    conv_gc.filters[idx] += eps
    fp = fwd_sum(conv_gc, X_small)
    conv_gc.filters[idx] -= 2 * eps
    fm = fwd_sum(conv_gc, X_small)
    conv_gc.filters[idx] += eps
    num_grad[idx] = (fp - fm) / (2 * eps)

rel_error = np.max(np.abs(analytic_grad - num_grad) /
                   (np.abs(analytic_grad) + np.abs(num_grad) + 1e-8))
print(f"Max relative error: {rel_error:.2e}  (should be < 1e-4)")
assert rel_error < 1e-4, f"Gradient check failed: {rel_error}"
print("Gradient check   : PASSED")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print()
    print("=" * 60)
    print("7. Visualising filters & feature maps")
    print("=" * 60)

    conv_vis = Conv2D(in_channels=1, out_channels=4, filter_size=3, stride=1, padding=1)
    img = np.zeros((1, 16, 16, 1))
    for i in range(16):
        img[0, i, i, 0] = 1.0
        if i > 0:
            img[0, i, i - 1, 0] = 0.5

    feature_maps = conv_vis.forward(img)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Day 15 – Conv2D: Filters (top) & Feature Maps (bottom)", fontsize=13)

    for k in range(4):
        filt = conv_vis.filters[:, :, 0, k]
        axes[0, k].imshow(filt, cmap='RdBu_r', vmin=-filt.max(), vmax=filt.max())
        axes[0, k].set_title(f"Filter {k}")
        axes[0, k].axis('off')

        fm = feature_maps[0, :, :, k]
        axes[1, k].imshow(fm, cmap='viridis')
        axes[1, k].set_title(f"Feature Map {k}")
        axes[1, k].axis('off')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'conv2d_filters.png')
    plt.savefig(out_path, dpi=100)
    print(f"Saved visualisation → {os.path.abspath(out_path)}")

except ImportError:
    print("matplotlib not available – skipping visualisation.")

print()
print("=" * 60)
print("All checks passed!  Conv2D layer is working correctly.")
print("=" * 60)
