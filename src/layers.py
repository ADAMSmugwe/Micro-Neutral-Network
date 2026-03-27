import numpy as np

class Layer:
    def get_config(self):
        return {
            'type': 'dense',
            'n_inputs': self.weights.shape[0],
            'n_neurons': self.weights.shape[1],
            'activation': self.activation,
            'dropout_rate': getattr(self, 'dropout_rate', 0.0)
        }

    def get_parameters(self):
        params = {
            'weights': self.weights,
            'biases': self.biases
        }
        if hasattr(self, 'm_weights'):
            params.update({
                'm_weights': self.m_weights,
                'v_weights': getattr(self, 'v_weights', None),
                'm_biases': self.m_biases,
                'v_biases': getattr(self, 'v_biases', None),
                't': getattr(self, 't', 0)
            })
        return params

    def set_parameters(self, params):
        self.weights = params['weights']
        self.biases = params['biases']
        if 'm_weights' in params and params['m_weights'] is not None:
            self.m_weights = params['m_weights']
        if 'v_weights' in params and params['v_weights'] is not None:
            self.v_weights = params['v_weights']
        if 'm_biases' in params and params['m_biases'] is not None:
            self.m_biases = params['m_biases']
        if 'v_biases' in params and params['v_biases'] is not None:
            self.v_biases = params['v_biases']
        if 't' in params:
            self.t = params['t']

    def __init__(self, n_inputs, n_neurons, activation='relu', dropout_rate=0.0, init_method='auto'):
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.training = True

        if init_method == 'auto':
            if activation in ['relu', 'leaky_relu']:
                init_method = 'he'
            elif activation in ['tanh', 'sigmoid']:
                init_method = 'xavier'
            else:
                init_method = 'he'

        if init_method == 'xavier':
            limit = np.sqrt(6.0 / (n_inputs + n_neurons))
            self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        elif init_method == 'xavier_normal':
            std = np.sqrt(2.0 / (n_inputs + n_neurons))
            self.weights = np.random.randn(n_inputs, n_neurons) * std
        elif init_method == 'he':
            std = np.sqrt(2.0 / n_inputs)
            self.weights = np.random.randn(n_inputs, n_neurons) * std
        elif init_method == 'he_uniform':
            limit = np.sqrt(6.0 / n_inputs)
            self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        elif init_method == 'random':
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        self.biases = np.zeros((1, n_neurons))
        self.inputs = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None
        self.v_weights = np.zeros_like(self.weights)
        self.v_biases = np.zeros_like(self.biases)
        self.m_weights = np.zeros_like(self.weights)
        self.vw_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.vw_biases = np.zeros_like(self.biases)
        self.t = 0
        self.dropout_mask = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self._activate(self.z)

        if self.training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.a.shape)
            return self.a * self.dropout_mask / (1 - self.dropout_rate)

        return self.a

    def _activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'softmax':
            e = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)
        return z

    def backward(self, dA, divisor=None):
        if self.training and self.dropout_rate > 0:
            dA *= self.dropout_mask / (1 - self.dropout_rate)

        if self.activation == 'relu':
            dZ = dA * (self.a > 0)
        elif self.activation == 'sigmoid':
            dZ = dA * (self.a * (1 - self.a))
        elif self.activation == 'tanh':
            dZ = dA * (1 - self.a**2)
        elif self.activation == 'softmax':
            dZ = dA
        else:
            dZ = dA

        if divisor is None:
            divisor = 1
        self.dW = np.dot(self.inputs.T, dZ) / divisor
        self.db = np.sum(dZ, axis=0, keepdims=True) / divisor

        dA_prev = np.dot(dZ, self.weights.T)
        return dA_prev


class Conv2D:
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        fan_in = filter_size * filter_size * in_channels
        self.filters = np.random.randn(filter_size, filter_size, in_channels, out_channels) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros((1, 1, 1, out_channels))

        self.d_filters = None
        self.d_biases = None
        self._X_pad = None
        self._col   = None
        self.v_filters = np.zeros_like(self.filters)
        self.v_biases  = np.zeros_like(self.biases)

    def forward(self, X, use_im2col=True):
        self._X_orig = X
        batch_size, h, w, in_c = X.shape
        f, stride, pad = self.filter_size, self.stride, self.padding

        if pad > 0:
            X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
        else:
            X_pad = X
        self._X_pad = X_pad

        out_h = (h + 2 * pad - f) // stride + 1
        out_w = (w + 2 * pad - f) // stride + 1

        if use_im2col:
            return self._forward_im2col(X_pad, batch_size, out_h, out_w)
        else:
            return self._forward_naive(X_pad, batch_size, out_h, out_w)

    def _forward_naive(self, X_pad, batch_size, out_h, out_w):
        f, stride = self.filter_size, self.stride
        out = np.zeros((batch_size, out_h, out_w, self.out_channels))

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    patch = X_pad[b, h_start:h_start + f, w_start:w_start + f, :]
                    for k in range(self.out_channels):
                        out[b, i, j, k] = np.sum(patch * self.filters[:, :, :, k]) + self.biases[0, 0, 0, k]
        return out

    def _forward_im2col(self, X_pad, batch_size, out_h, out_w):
        f, stride = self.filter_size, self.stride

        col = np.zeros((batch_size, out_h, out_w, f, f, self.in_channels))
        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * stride, j * stride
                col[:, i, j, :, :, :] = X_pad[:, h_s:h_s + f, w_s:w_s + f, :]

        col = col.reshape(batch_size, out_h, out_w, -1)
        self._col = col
        filters_col = self.filters.reshape(-1, self.out_channels)
        out = col @ filters_col + self.biases
        return out

    def _backward_naive(self, d_out):
        X_pad = self._X_pad
        batch_size, out_h, out_w, _ = d_out.shape
        f, stride, pad = self.filter_size, self.stride, self.padding

        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.sum(d_out, axis=(0, 1, 2), keepdims=True)
        d_X_pad = np.zeros_like(X_pad)

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_s, w_s = i * stride, j * stride
                    patch = X_pad[b, h_s:h_s + f, w_s:w_s + f, :]
                    self.d_filters += patch[:, :, :, np.newaxis] * d_out[b, i, j, np.newaxis, np.newaxis, np.newaxis, :]
                    d_X_pad[b, h_s:h_s + f, w_s:w_s + f, :] += np.tensordot(
                        self.filters, d_out[b, i, j, :], axes=([3], [0])
                    )

        if pad > 0:
            return d_X_pad[:, pad:-pad, pad:-pad, :]
        return d_X_pad

    def _backward_im2col(self, d_out):
        batch_size, out_h, out_w, out_c = d_out.shape
        f, stride, pad = self.filter_size, self.stride, self.padding

        self.d_biases = np.sum(d_out, axis=(0, 1, 2), keepdims=True)

        d_out_flat  = d_out.reshape(-1, out_c)
        col_flat    = self._col.reshape(-1, f * f * self.in_channels)
        filters_col = self.filters.reshape(-1, out_c)

        self.d_filters = (col_flat.T @ d_out_flat).reshape(self.filters.shape)

        d_col = (d_out_flat @ filters_col.T).reshape(
            batch_size, out_h, out_w, f, f, self.in_channels
        )

        d_X_pad = np.zeros_like(self._X_pad)
        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * stride, j * stride
                d_X_pad[:, h_s:h_s + f, w_s:w_s + f, :] += d_col[:, i, j, :, :, :]

        if pad > 0:
            return d_X_pad[:, pad:-pad, pad:-pad, :]
        return d_X_pad

    def backward(self, d_out, use_im2col=True):
        if use_im2col and self._col is not None:
            return self._backward_im2col(d_out)
        return self._backward_naive(d_out)


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self._mask = None
        self._input_shape = None

    def forward(self, X):
        batch, h, w, c = X.shape
        p, s = self.pool_size, self.stride
        out_h = (h - p) // s + 1
        out_w = (w - p) // s + 1
        self._input_shape = X.shape
        self._mask = np.zeros_like(X)
        out = np.zeros((batch, out_h, out_w, c))
        b_idx = np.arange(batch)[:, None]
        c_idx = np.arange(c)[None, :]
        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * s, j * s
                window = X[:, h_s:h_s + p, w_s:w_s + p, :]
                flat = window.reshape(batch, p * p, c)
                idx = np.argmax(flat, axis=1)
                out[:, i, j, :] = flat[b_idx, idx, c_idx]
                self._mask[b_idx, h_s + idx // p, w_s + idx % p, c_idx] = 1
        return out

    def backward(self, d_out):
        batch, out_h, out_w, c = d_out.shape
        p, s = self.pool_size, self.stride
        d_X = np.zeros(self._input_shape)
        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * s, j * s
                d_X[:, h_s:h_s + p, w_s:w_s + p, :] += (
                    self._mask[:, h_s:h_s + p, w_s:w_s + p, :] * d_out[:, i:i + 1, j:j + 1, :]
                )
        return d_X


class Flatten:
    def forward(self, X):
        self._original_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self._original_shape)


class ReLU:
    def forward(self, X):
        self._mask = X > 0
        return np.maximum(0, X)

    def backward(self, d_out):
        return d_out * self._mask


class BatchNorm:
    def get_config(self):
        return {
            'type': 'batchnorm',
            'n_features': self.gamma.shape[1],
            'eps': self.eps,
            'momentum': self.momentum
        }

    def get_parameters(self):
        return {
            'gamma': self.gamma,
            'beta': self.beta,
            'running_mean': self.running_mean,
            'running_var': self.running_var
        }

    def set_parameters(self, params):
        self.gamma = params['gamma']
        self.beta = params['beta']
        self.running_mean = params['running_mean']
        self.running_var = params['running_var']

    def __init__(self, n_features, eps=1e-8, momentum=0.9):
        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))
        self.eps = eps
        self.momentum = momentum

        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))

        self.training = True

        self.x_norm = None
        self.x_centered = None
        self.std = None
        self.var = None
        self.mean = None
        self.batch_size = None

        self.dgamma = None
        self.dbeta = None
        self.v_gamma = np.zeros_like(self.gamma)
        self.v_beta = np.zeros_like(self.beta)

    def forward(self, Z):
        if self.training:
            self.batch_size = Z.shape[0]
            self.mean = np.mean(Z, axis=0, keepdims=True)
            self.var = np.var(Z, axis=0, keepdims=True)

            self.x_centered = Z - self.mean
            self.std = np.sqrt(self.var + self.eps)
            self.x_norm = self.x_centered / self.std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            out = self.gamma * self.x_norm + self.beta
        else:
            x_norm = (Z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        if self.training:
            self.dgamma = np.sum(dout * self.x_norm, axis=0, keepdims=True)
            self.dbeta = np.sum(dout, axis=0, keepdims=True)

            dx_norm = dout * self.gamma

            dvar = np.sum(dx_norm * self.x_centered * -0.5 * self.std ** (-3), axis=0, keepdims=True)
            dmean = np.sum(dx_norm * -1 / self.std, axis=0, keepdims=True) + dvar * np.mean(-2 * self.x_centered, axis=0, keepdims=True)

            dZ = (dx_norm / self.std) + (dvar * 2 * self.x_centered / self.batch_size) + (dmean / self.batch_size)

        else:
            dZ = dout * self.gamma / np.sqrt(self.running_var + self.eps)

        return dZ


class ConvBatchNorm:
    """Batch normalization for conv layers: normalizes over (batch, H, W), one scale/shift per channel."""

    def __init__(self, n_channels, eps=1e-8, momentum=0.9):
        self.n_channels = n_channels
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones((1, 1, 1, n_channels))
        self.beta = np.zeros((1, 1, 1, n_channels))
        self.running_mean = np.zeros((1, 1, 1, n_channels))
        self.running_var = np.ones((1, 1, 1, n_channels))

        self.training = True

        self.x_norm = None
        self.x_centered = None
        self.std = None
        self._n = None

        self.dgamma = None
        self.dbeta = None
        self.v_gamma = np.zeros_like(self.gamma)
        self.v_beta = np.zeros_like(self.beta)

    def forward(self, Z):
        if self.training:
            self._n = Z.shape[0] * Z.shape[1] * Z.shape[2]
            mean = np.mean(Z, axis=(0, 1, 2), keepdims=True)
            var  = np.var(Z,  axis=(0, 1, 2), keepdims=True)
            self.std = np.sqrt(var + self.eps)
            self.x_centered = Z - mean
            self.x_norm = self.x_centered / self.std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var

            return self.gamma * self.x_norm + self.beta
        else:
            x_norm = (Z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta

    def backward(self, dout):
        self.dgamma = np.sum(dout * self.x_norm,  axis=(0, 1, 2), keepdims=True)
        self.dbeta  = np.sum(dout,                axis=(0, 1, 2), keepdims=True)

        dx_norm = dout * self.gamma
        dvar    = np.sum(dx_norm * self.x_centered * -0.5 * self.std ** (-3),
                         axis=(0, 1, 2), keepdims=True)
        dmean   = (np.sum(dx_norm * -1.0 / self.std, axis=(0, 1, 2), keepdims=True)
                   + dvar * np.mean(-2.0 * self.x_centered, axis=(0, 1, 2), keepdims=True))

        dZ = ((dx_norm / self.std)
              + (dvar * 2.0 * self.x_centered / self._n)
              + (dmean / self._n))
        return dZ


class GlobalAvgPool2D:
    """Average-pool the spatial dimensions, outputting (batch, channels)."""

    def forward(self, X):
        self._input_shape = X.shape
        return np.mean(X, axis=(1, 2))

    def backward(self, dout):
        batch, h, w, c = self._input_shape
        return (dout[:, np.newaxis, np.newaxis, :] / (h * w)) * np.ones((batch, h, w, c))


class ResidualBlock:
    """Two 3x3 conv layers with a skip connection (He et al. 2016).

    When stride > 1 or channels change, a 1x1 projection convolution
    is used on the shortcut path so shapes match before addition.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = Conv2D(in_channels, out_channels, filter_size=3, stride=stride, padding=1)
        self.bn1   = ConvBatchNorm(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(out_channels, out_channels, filter_size=3, stride=1, padding=1)
        self.bn2   = ConvBatchNorm(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut    = Conv2D(in_channels, out_channels, filter_size=1, stride=stride, padding=0)
            self.shortcut_bn = ConvBatchNorm(out_channels)
        else:
            self.shortcut    = None
            self.shortcut_bn = None

        self.relu_out = ReLU()
        self._training = True

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        self._training = value
        for layer in self._all_sublayers():
            if hasattr(layer, 'training'):
                layer.training = value

    def _all_sublayers(self):
        layers = [self.conv1, self.bn1, self.relu1,
                  self.conv2, self.bn2, self.relu_out]
        if self.shortcut is not None:
            layers += [self.shortcut, self.shortcut_bn]
        return layers

    def forward(self, X):
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.shortcut is not None:
            identity = self.shortcut.forward(X)
            identity = self.shortcut_bn.forward(identity)
        else:
            identity = X

        out = out + identity
        out = self.relu_out.forward(out)
        return out

    def backward(self, dout):
        dout = self.relu_out.backward(dout)

        d_main     = dout.copy()
        d_identity = dout.copy()

        d_main = self.bn2.backward(d_main)
        d_main = self.conv2.backward(d_main)
        d_main = self.relu1.backward(d_main)
        d_main = self.bn1.backward(d_main)
        d_main = self.conv1.backward(d_main)

        if self.shortcut is not None:
            d_identity = self.shortcut_bn.backward(d_identity)
            d_identity = self.shortcut.backward(d_identity)

        return d_main + d_identity

    def update(self, lr=0.01, momentum=0.0, optimizer='sgd', beta1=0.9, beta2=0.999, eps=1e-8):
        conv_layers = [self.conv1, self.conv2]
        bn_layers   = [self.bn1, self.bn2]
        if self.shortcut is not None:
            conv_layers.append(self.shortcut)
            bn_layers.append(self.shortcut_bn)

        for layer in conv_layers:
            if layer.d_filters is None:
                continue
            layer.v_filters = momentum * layer.v_filters - lr * layer.d_filters
            layer.v_biases  = momentum * layer.v_biases  - lr * layer.d_biases
            layer.filters  += layer.v_filters
            layer.biases   += layer.v_biases

        for layer in bn_layers:
            if layer.dgamma is None:
                continue
            layer.v_gamma = momentum * layer.v_gamma - lr * layer.dgamma
            layer.v_beta  = momentum * layer.v_beta  - lr * layer.dbeta
            layer.gamma  += layer.v_gamma
            layer.beta   += layer.v_beta


class ChannelAttention:
    """Squeeze-and-Excitation channel attention (Hu et al. 2018).

    Squeezes spatial information via global average pooling, then learns
    per-channel importance weights via two FC layers.  The weights rescale
    every channel of the feature map.

    Args:
        channels:  Number of input/output channels.
        reduction: Bottleneck ratio for the two FC layers (default 16).
    """

    def __init__(self, channels, reduction=16):
        self.channels = channels
        reduced = max(channels // reduction, 1)
        self.gap = GlobalAvgPool2D()
        self.fc1 = Layer(channels, reduced,  'relu')
        self.fc2 = Layer(reduced,  channels, 'sigmoid')
        self._training = True

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        self._training = value
        self.fc1.training = value
        self.fc2.training = value

    def forward(self, X):
        self._X = X
        squeezed   = self.gap.forward(X)
        excitation = self.fc1.forward(squeezed)
        excitation = self.fc2.forward(excitation)
        self._excitation = excitation[:, np.newaxis, np.newaxis, :]
        return X * self._excitation

    def backward(self, dout):
        dX           = dout * self._excitation
        d_excitation = np.sum(dout * self._X, axis=(1, 2))
        d2   = self.fc2.backward(d_excitation)
        d1   = self.fc1.backward(d2)
        d_gap = self.gap.backward(d1)
        return dX + d_gap

    def update(self, lr=0.01, momentum=0.0, optimizer='sgd',
               beta1=0.9, beta2=0.999, eps=1e-8):
        for layer in [self.fc1, self.fc2]:
            if layer.dW is None:
                continue
            if optimizer == 'adam':
                layer.t += 1
                g = layer.dW
                layer.m_weights  = beta1 * layer.m_weights  + (1 - beta1) * g
                layer.vw_weights = beta2 * layer.vw_weights + (1 - beta2) * (g ** 2)
                m_hat = layer.m_weights  / (1 - beta1 ** layer.t)
                v_hat = layer.vw_weights / (1 - beta2 ** layer.t)
                layer.weights -= lr * m_hat / (np.sqrt(v_hat) + eps)
                g_b = layer.db
                layer.m_biases  = beta1 * layer.m_biases  + (1 - beta1) * g_b
                layer.vw_biases = beta2 * layer.vw_biases + (1 - beta2) * (g_b ** 2)
                m_hat_b = layer.m_biases  / (1 - beta1 ** layer.t)
                v_hat_b = layer.vw_biases / (1 - beta2 ** layer.t)
                layer.biases -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
            else:
                layer.v_weights = momentum * layer.v_weights - lr * layer.dW
                layer.v_biases  = momentum * layer.v_biases  - lr * layer.db
                layer.weights  += layer.v_weights
                layer.biases   += layer.v_biases


class SpatialAttention:
    """Spatial attention via two 1×1 convolutions (from CBAM, Woo et al. 2018).

    Projects channels to a small bottleneck with ReLU, then reduces to a
    single-channel attention map with sigmoid.  Every channel of the input is
    scaled by the learned spatial weight at that position.

    Args:
        channels:  Number of input/output channels.
        reduction: Bottleneck ratio (default 16).
    """

    def __init__(self, channels, reduction=16):
        reduced = max(channels // reduction, 1)
        self.conv1 = Conv2D(channels, reduced, filter_size=1, padding=0)
        self.conv2 = Conv2D(reduced,  1,       filter_size=1, padding=0)

    def forward(self, X):
        self._X = X
        a1 = self.conv1.forward(X)
        self._pre_relu = a1
        a1 = np.maximum(0, a1)
        a2 = self.conv2.forward(a1)
        self._attn = 1.0 / (1.0 + np.exp(-a2))
        return X * self._attn

    def backward(self, dout):
        dX     = dout * self._attn
        d_attn = np.sum(dout * self._X, axis=-1, keepdims=True)
        d_pre_sigmoid = d_attn * self._attn * (1.0 - self._attn)
        d_relu  = self.conv2.backward(d_pre_sigmoid)
        d_relu  = d_relu * (self._pre_relu > 0)
        d_conv1 = self.conv1.backward(d_relu)
        return dX + d_conv1

    def update(self, lr=0.01, momentum=0.0, optimizer='sgd',
               beta1=0.9, beta2=0.999, eps=1e-8):
        for conv in [self.conv1, self.conv2]:
            if conv.d_filters is None:
                continue
            conv.v_filters = momentum * conv.v_filters - lr * conv.d_filters
            conv.v_biases  = momentum * conv.v_biases  - lr * conv.d_biases
            conv.filters  += conv.v_filters
            conv.biases   += conv.v_biases


class CBAM:
    """Convolutional Block Attention Module (Woo et al. 2018).

    Applies channel attention (which channels matter?) followed by spatial
    attention (where matters?).  Drop-in after any conv layer or ResBlock.

    Args:
        channels:  Number of feature channels.
        reduction: Bottleneck reduction ratio for both sub-modules (default 16).
    """

    def __init__(self, channels, reduction=16):
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(channels, reduction)
        self._training = True

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        self._training = value
        self.channel_attention.training = value

    def forward(self, X):
        out = self.channel_attention.forward(X)
        out = self.spatial_attention.forward(out)
        return out

    def backward(self, dout):
        dout = self.spatial_attention.backward(dout)
        dout = self.channel_attention.backward(dout)
        return dout

    def update(self, lr=0.01, momentum=0.0, optimizer='sgd',
               beta1=0.9, beta2=0.999, eps=1e-8):
        self.channel_attention.update(lr, momentum, optimizer, beta1, beta2, eps)
        self.spatial_attention.update(lr, momentum, optimizer, beta1, beta2, eps)
