import numpy as np
from .loss import mse_loss, mse_derivative, cross_entropy_loss, cross_entropy_derivative

class Network:
    def __init__(self):
        self.layers = []
        self.loss_name = None
        self.loss = None
        self.loss_derivative = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_name):
        self.loss_name = loss_name
        if loss_name == 'mse':
            self.loss = mse_loss
            self.loss_derivative = mse_derivative
        elif loss_name == 'cross_entropy':
            self.loss = cross_entropy_loss
            self.loss_derivative = cross_entropy_derivative
        else:
            raise ValueError('Unsupported loss function')

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred):
        dA = self.loss_derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self, lr=0.01):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= lr * layer.dW
                layer.biases -= lr * layer.db

    def train(self, X, y, epochs=1000, lr=0.1, verbose=True, print_every=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss_val = self.loss(y, y_pred)
            self.backward(y, y_pred)
            self.update(lr)
            if verbose and epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss_val:.6f}")
        return self.loss(y, self.forward(X))
