import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.inputs = None
        self.z = None
        self.a = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self._activate(self.z)
        return self.a

    def _activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        return z
