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
