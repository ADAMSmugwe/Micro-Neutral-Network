import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, activation='relu', dropout_rate=0.0):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.training = True
        self.inputs = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None
        self.v_weights = np.zeros_like(self.weights)
        self.v_biases = np.zeros_like(self.biases)
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
