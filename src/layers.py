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
    # Adam optimizer state
    self.m_weights = np.zeros_like(self.weights)
    self.vw_weights = np.zeros_like(self.weights)
    self.m_biases = np.zeros_like(self.biases)
    self.vw_biases = np.zeros_like(self.biases)
    self.t = 0  # timestep for Adam
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

    def backward(self, dA):
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

        m = dA.shape[0]
        self.dW = np.dot(self.inputs.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        dA_prev = np.dot(dZ, self.weights.T)
        return dA_prev


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
