import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.layers import Conv2D, Layer

np.random.seed(0)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def numerical_grad(loss_fn, param_array, eps=1e-5):
    grad = np.zeros_like(param_array)
    for idx in np.ndindex(param_array.shape):
        param_array[idx] += eps
        lp = loss_fn()
        param_array[idx] -= 2 * eps
        lm = loss_fn()
        param_array[idx] += eps
        grad[idx] = (lp - lm) / (2 * eps)
    return grad

def rel_error(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))

def check(name, analytical, numerical, tol=1e-4):
    err = rel_error(analytical, numerical)
    status = "PASSED" if err < tol else "FAILED"
    print(f"  {name:30s}: rel err = {err:.2e}  [{status}]")
    assert err < tol, f"Gradient check failed for {name}: {err:.2e}"


print("=" * 60)
print("1. Gradient check – stride=1, padding=0")
print("=" * 60)

X  = np.random.randn(2, 6, 6, 3)
tgt = np.random.randn(2, 4, 4, 4)
conv = Conv2D(in_channels=3, out_channels=4, filter_size=3, stride=1, padding=0)

def loss_filters():
    return mse(tgt, conv.forward(X))

out = conv.forward(X)
d_out = 2 * (out - tgt) / tgt.size
conv.backward(d_out)

check("d_filters (im2col bwd)", conv.d_filters, numerical_grad(loss_filters, conv.filters))

conv_n = Conv2D(in_channels=3, out_channels=4, filter_size=3, stride=1, padding=0)
conv_n.filters = conv.filters.copy()
conv_n.biases  = conv.biases.copy()
conv_n.forward(X, use_im2col=False)
d_X_naive = conv_n._backward_naive(d_out)
d_X_im2col = conv.backward(d_out)

check("d_filters naive vs im2col", conv_n.d_filters, conv.d_filters)
check("d_X     naive vs im2col",   d_X_naive,        d_X_im2col)

def loss_biases():
    return mse(tgt, conv.forward(X))

conv.backward(d_out)
check("d_biases", conv.d_biases, numerical_grad(loss_biases, conv.biases))

def loss_X():
    return mse(tgt, conv.forward(X))

check("d_X", d_X_im2col, numerical_grad(loss_X, X))

print()
print("=" * 60)
print("2. Gradient check – stride=1, padding=1  (same-size output)")
print("=" * 60)

X2  = np.random.randn(2, 6, 6, 2)
tgt2 = np.random.randn(2, 6, 6, 3)
conv2 = Conv2D(in_channels=2, out_channels=3, filter_size=3, stride=1, padding=1)

def loss2_filters():
    return mse(tgt2, conv2.forward(X2))

def loss2_X():
    return mse(tgt2, conv2.forward(X2))

out2  = conv2.forward(X2)
d_out2 = 2 * (out2 - tgt2) / tgt2.size
conv2.backward(d_out2)

check("d_filters (pad=1)", conv2.d_filters, numerical_grad(loss2_filters, conv2.filters))
check("d_X      (pad=1)", conv2.backward(d_out2), numerical_grad(loss2_X, X2))

print()
print("=" * 60)
print("3. Gradient check – stride=2, padding=0")
print("=" * 60)

X3   = np.random.randn(2, 8, 8, 2)
tgt3 = np.random.randn(2, 3, 3, 2)
conv3 = Conv2D(in_channels=2, out_channels=2, filter_size=3, stride=2, padding=0)

def loss3_filters():
    return mse(tgt3, conv3.forward(X3))

def loss3_X():
    return mse(tgt3, conv3.forward(X3))

out3   = conv3.forward(X3)
d_out3 = 2 * (out3 - tgt3) / tgt3.size
conv3.backward(d_out3)

check("d_filters (stride=2)", conv3.d_filters, numerical_grad(loss3_filters, conv3.filters))
check("d_X      (stride=2)", conv3.backward(d_out3), numerical_grad(loss3_X, X3))

print()
print("=" * 60)
print("4. Mini CNN  (Conv → flatten → Dense)  end-to-end gradient check")
print("=" * 60)

cnn  = Conv2D(in_channels=1, out_channels=2, filter_size=3, stride=1, padding=0)
dense = Layer(n_inputs=2*4*4, n_neurons=3, activation='sigmoid')

X4   = np.random.randn(2, 6, 6, 1)
y4   = np.random.randn(2, 3)

def cnn_forward(X):
    feat = cnn.forward(X)
    flat = feat.reshape(feat.shape[0], -1)
    return dense.forward(flat)

def cnn_loss(X):
    return mse(y4, cnn_forward(X))

pred = cnn_forward(X4)
dL   = 2 * (pred - y4) / y4.size

dA_flat = dense.backward(dL)
dA_conv = dA_flat.reshape(2, 4, 4, 2)
cnn.backward(dA_conv)

def loss_cnn_filters():
    return cnn_loss(X4)

def loss_dense_W():
    return cnn_loss(X4)

def loss_cnn_X():
    return cnn_loss(X4)

check("CNN  d_filters",  cnn.d_filters,  numerical_grad(loss_cnn_filters, cnn.filters))
check("Dense dW",        dense.dW,        numerical_grad(loss_dense_W,    dense.weights))
check("CNN  d_X",        cnn.backward(dA_conv), numerical_grad(loss_cnn_X, X4))

print()
print("=" * 60)
print("5. Naive vs im2col agreement – padding=1, stride=2")
print("=" * 60)

X5 = np.random.randn(3, 8, 8, 2)
c_a = Conv2D(in_channels=2, out_channels=3, filter_size=3, stride=2, padding=1)
c_b = Conv2D(in_channels=2, out_channels=3, filter_size=3, stride=2, padding=1)
c_b.filters = c_a.filters.copy()
c_b.biases  = c_a.biases.copy()

out5  = c_a.forward(X5, use_im2col=True)
tgt5  = np.random.randn(*out5.shape)
d5    = 2 * (out5 - tgt5) / tgt5.size

c_b.forward(X5, use_im2col=False)

dX_im2col = c_a.backward(d5, use_im2col=True)
dX_naive  = c_b._backward_naive(d5)

check("d_filters im2col vs naive", c_a.d_filters, c_b.d_filters)
check("d_X      im2col vs naive",  dX_im2col,     dX_naive)

print()
print("=" * 60)
print("All Day 16 checks passed!")
print("=" * 60)
