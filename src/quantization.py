"""
quantization.py – Post-training quantization (PTQ) and
                  Quantization-Aware Training (QAT) utilities.

Public API
----------
Quantizer           – static helpers: scale/zp, quantize, dequantize
FakeQuantize        – QAT layer: quantize-then-dequantize with STE gradient
QuantizedLayer      – PTQ wrapper around a trained dense Layer
QuantizedConv2D     – PTQ wrapper around a trained Conv2D layer
quantize_network()  – convert every weight layer in a Network to PTQ
model_memory_bytes()        – FP32 parameter bytes for a Network
quantized_memory_bytes()    – INT8 parameter bytes for a PTQ layer list
"""

import numpy as np


# ── quantization math ────────────────────────────────────────────────────────

class Quantizer:
    """Pure-static helpers for INT8 affine quantization.

    Affine (asymmetric) mapping:
        quantized  = clip(round(x / scale + zero_point), INT8_MIN, INT8_MAX)
        x_approx   = scale * (quantized - zero_point)

    scale     – float, maps one INT8 step to one FP32 unit
    zero_point – int8, the quantized value that represents FP32 zero
    """

    INT8_MIN: int = -128
    INT8_MAX: int =  127

    # ── per-tensor ────────────────────────────────────────────────────────────
    @staticmethod
    def get_scale_and_zero_point(tensor: np.ndarray):
        """Compute per-tensor scale and zero_point from a float32 tensor."""
        mn = float(np.min(tensor))
        mx = float(np.max(tensor))
        if mx == mn:
            return 1.0, 0
        scale = (mx - mn) / (Quantizer.INT8_MAX - Quantizer.INT8_MIN)
        zp    = int(round(Quantizer.INT8_MIN - mn / scale))
        zp    = int(np.clip(zp, Quantizer.INT8_MIN, Quantizer.INT8_MAX))
        return scale, zp

    @staticmethod
    def quantize(tensor: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Map FP32 tensor → INT8 tensor."""
        q = np.round(tensor / scale + zero_point)
        return np.clip(q, Quantizer.INT8_MIN, Quantizer.INT8_MAX).astype(np.int8)

    @staticmethod
    def dequantize(tensor: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Map INT8 tensor → FP32 approximation."""
        return scale * (tensor.astype(np.float32) - zero_point)

    # ── per-channel (one scale/zp per output column) ──────────────────────────
    @staticmethod
    def get_per_channel_params(weights: np.ndarray):
        """Per-column scale/zp for a (n_in, n_out) weight matrix.

        Returns
        -------
        scales      : ndarray shape (n_out,)
        zero_points : ndarray shape (n_out,)
        """
        n_out = weights.shape[1]
        scales = np.empty(n_out, dtype=np.float64)
        zps    = np.empty(n_out, dtype=np.int32)
        for c in range(n_out):
            scales[c], zps[c] = Quantizer.get_scale_and_zero_point(weights[:, c])
        return scales, zps

    @staticmethod
    def quantize_per_channel(weights: np.ndarray, scales, zero_points) -> np.ndarray:
        """Quantize a (n_in, n_out) weight matrix column-by-column."""
        q = np.empty_like(weights, dtype=np.int8)
        for c in range(weights.shape[1]):
            q[:, c] = Quantizer.quantize(weights[:, c], scales[c], zero_points[c])
        return q

    @staticmethod
    def dequantize_per_channel(q_weights: np.ndarray, scales, zero_points) -> np.ndarray:
        """Reconstruct FP32 weights from per-channel INT8 representation."""
        out = np.empty(q_weights.shape, dtype=np.float32)
        for c in range(q_weights.shape[1]):
            out[:, c] = Quantizer.dequantize(q_weights[:, c], scales[c], zero_points[c])
        return out


# ── fake quantization (for QAT) ──────────────────────────────────────────────

class FakeQuantize:
    """Quantization-aware training layer (straight-through estimator).

    On the forward pass, tensors are quantized then immediately dequantized
    (so they stay FP32 but contain the discretisation noise of INT8).  On
    the backward pass, gradients flow through unchanged – the straight-through
    estimator (STE) approximation that makes QAT tractable.

    Use as a drop-in layer inside a Network:

        net.add_layer(FakeQuantize())
        net.add_layer(Layer(784, 256, 'relu'))
        net.add_layer(FakeQuantize())
        ...
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._in_range = (float(np.min(X)), float(np.max(X)))
        scale, zp = Quantizer.get_scale_and_zero_point(X)
        q  = Quantizer.quantize(X, scale, zp)
        return Quantizer.dequantize(q, scale, zp)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Straight-through estimator: pass gradient unchanged.
        return dout


# ── post-training quantization wrappers ──────────────────────────────────────

class QuantizedLayer:
    """Inference-only wrapper that replaces FP32 weights with INT8 storage.

    Supports both per-tensor and per-channel quantization; per-channel
    generally gives lower accuracy drop with the same 4× memory saving.

    Parameters
    ----------
    original_layer : Layer
        A trained dense Layer instance whose weights will be quantized.
    per_channel : bool
        If True, use a separate scale/zp per output neuron (column).
    """

    def __init__(self, original_layer, per_channel: bool = False):
        self.layer       = original_layer
        self.per_channel = per_channel
        self._quantized  = False

        # populated by quantize_weights()
        self.q_weights  = None
        self.scales     = None     # scalar or (n_out,) array
        self.zps        = None     # scalar int or (n_out,) array

    # ── weight quantization ───────────────────────────────────────────────────
    def quantize_weights(self):
        """Convert FP32 weights to INT8 and report the size change."""
        W = self.layer.weights

        if self.per_channel:
            self.scales, self.zps = Quantizer.get_per_channel_params(W)
            self.q_weights        = Quantizer.quantize_per_channel(W, self.scales, self.zps)
        else:
            s, zp              = Quantizer.get_scale_and_zero_point(W)
            self.scales, self.zps = s, zp
            self.q_weights     = Quantizer.quantize(W, s, zp)

        self._quantized = True
        label = "per-channel" if self.per_channel else "per-tensor"
        saved = W.nbytes - self.q_weights.nbytes
        ratio = W.nbytes / self.q_weights.nbytes
        print(f"  QuantizedLayer  [{label}]  "
              f"{W.nbytes:,} → {self.q_weights.nbytes:,} bytes  "
              f"({ratio:.1f}× smaller,  saved {saved:,} B)")

    # ── inference forward ─────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        if self._quantized:
            if self.per_channel:
                W = Quantizer.dequantize_per_channel(self.q_weights, self.scales, self.zps)
            else:
                W = Quantizer.dequantize(self.q_weights, self.scales, self.zps)
        else:
            W = self.layer.weights
        z = np.dot(X, W) + self.layer.biases
        return self.layer._activate(z)   # biases stay FP32

    # ── size reporting ────────────────────────────────────────────────────────
    @property
    def memory_bytes(self) -> int:
        """Bytes used by this layer (INT8 weights + FP32 biases)."""
        if self._quantized:
            return int(self.q_weights.nbytes + self.layer.biases.nbytes)
        return int(self.layer.weights.nbytes + self.layer.biases.nbytes)

    @property
    def original_bytes(self) -> int:
        """FP32 bytes that this layer replaces."""
        return int(self.layer.weights.nbytes + self.layer.biases.nbytes)


class QuantizedConv2D:
    """Inference-only wrapper that quantizes Conv2D filter weights to INT8.

    Per-tensor quantization is used (one scale/zp for the full filter bank).
    Biases stay in FP32 for numerical stability.

    Parameters
    ----------
    original_conv : Conv2D
        A trained Conv2D instance.
    """

    def __init__(self, original_conv):
        self.conv       = original_conv
        self._quantized = False
        self.q_filters  = None
        self.f_scale    = None
        self.f_zp       = None

    def quantize_weights(self):
        F = self.conv.filters
        self.f_scale, self.f_zp = Quantizer.get_scale_and_zero_point(F)
        self.q_filters           = Quantizer.quantize(F, self.f_scale, self.f_zp)
        self._quantized = True
        ratio = F.nbytes / self.q_filters.nbytes
        saved = F.nbytes - self.q_filters.nbytes
        print(f"  QuantizedConv2D  [per-tensor]  "
              f"{F.nbytes:,} → {self.q_filters.nbytes:,} bytes  "
              f"({ratio:.1f}× smaller,  saved {saved:,} B)")

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self._quantized:
            # swap in dequantized filters for the duration of this forward pass
            orig_filters        = self.conv.filters
            self.conv.filters   = Quantizer.dequantize(self.q_filters, self.f_scale, self.f_zp)
            out                 = self.conv.forward(X)
            self.conv.filters   = orig_filters
        else:
            out = self.conv.forward(X)
        return out

    @property
    def memory_bytes(self) -> int:
        if self._quantized:
            return int(self.q_filters.nbytes + self.conv.biases.nbytes)
        return int(self.conv.filters.nbytes + self.conv.biases.nbytes)

    @property
    def original_bytes(self) -> int:
        return int(self.conv.filters.nbytes + self.conv.biases.nbytes)


# ── network-level helpers ─────────────────────────────────────────────────────

def quantize_network(net, per_channel: bool = True):
    """Apply PTQ to every weight layer in *net*.

    Returns a list of layers suitable for inference (no training possible).
    Dense layers → QuantizedLayer, Conv2D layers → QuantizedConv2D,
    all others (ReLU, BN, pooling, etc.) pass through unchanged.

    Parameters
    ----------
    net          : Network
    per_channel  : bool  Use per-channel quantization for dense layers.
    """
    from src.layers import Conv2D  # local import to avoid circular dependency

    q_layers = []
    for layer in net.layers:
        if hasattr(layer, 'weights') and not hasattr(layer, 'filters'):
            # Dense Layer
            ql = QuantizedLayer(layer, per_channel=per_channel)
            ql.quantize_weights()
            q_layers.append(ql)
        elif hasattr(layer, 'filters'):
            # Conv2D
            qc = QuantizedConv2D(layer)
            qc.quantize_weights()
            q_layers.append(qc)
        else:
            # ReLU, BatchNorm, pooling, Flatten, GlobalAvgPool2D, CBAM, etc.
            q_layers.append(layer)

    return q_layers


def infer(q_layers: list, X: np.ndarray) -> np.ndarray:
    """Run a forward pass through a list of (possibly quantized) layers."""
    for layer in q_layers:
        X = layer.forward(X)
    return X


# ── size utilities ────────────────────────────────────────────────────────────

def model_memory_bytes(net) -> int:
    """Total FP32 parameter bytes in a Network."""
    total = 0
    for layer in net.layers:
        if hasattr(layer, 'weights'):
            total += layer.weights.nbytes
        if hasattr(layer, 'biases') and not hasattr(layer, 'weights'):
            # biases on conv layers
            total += layer.biases.nbytes
        if hasattr(layer, 'filters'):
            total += layer.filters.nbytes
            total += layer.biases.nbytes
        if hasattr(layer, 'gamma'):    # BatchNorm
            total += layer.gamma.nbytes + layer.beta.nbytes
    return total


def quantized_memory_bytes(q_layers: list) -> int:
    """Total parameter bytes for a list of (possibly quantized) layers."""
    total = 0
    for layer in q_layers:
        if hasattr(layer, 'memory_bytes'):
            total += layer.memory_bytes
        elif hasattr(layer, 'weights'):
            total += layer.weights.nbytes + layer.biases.nbytes
        elif hasattr(layer, 'filters'):
            total += layer.filters.nbytes + layer.biases.nbytes
        elif hasattr(layer, 'gamma'):
            total += layer.gamma.nbytes + layer.beta.nbytes
    return total
