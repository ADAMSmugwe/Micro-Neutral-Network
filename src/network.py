# src/network.py
import numpy as np
from .loss import mse_loss, mse_derivative, cross_entropy_loss, cross_entropy_derivative

class network:
    def __init__(self, layers=None, reg_lambda=0.0):
        self.layers = layers if layers is not None else []
        self.loss_name = None
        self.loss_func = None
        self.loss_derivative = None
        self.reg_lambda = reg_lambda

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_name):
        self.loss_name = loss_name
        if loss_name == 'mse':
            self.loss_func = mse_loss
            self.loss_derivative = mse_derivative
        elif loss_name == 'cross_entropy':
            self.loss_func = cross_entropy_loss
            self.loss_derivative = cross_entropy_derivative
        else:
            raise ValueError('Unsupported loss function')

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, y_true, y_pred):
        data_loss = self.loss_func(y_true, y_pred)
        reg_loss = 0
        if self.reg_lambda > 0:
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    reg_loss += np.sum(np.square(layer.weights))
        return data_loss + (self.reg_lambda / 2) * reg_loss

    def backward(self, y_true, y_pred):
        dA = self.loss_derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        
        if self.reg_lambda > 0:
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    layer.dW += self.reg_lambda * layer.weights

    def update(self, lr=0.01):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= lr * layer.dW
                layer.biases -= lr * layer.db

    def train(self, X, y, epochs=1000, lr=0.01, batch_size=32, verbose=True, print_every=100):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.update(lr)
            
            if verbose and epoch % print_every == 0:
                full_pred = self.forward(X)
                current_loss = self.loss(y, full_pred)
                print(f"Epoch {epoch}, Loss: {current_loss:.6f}")
        
        return self.loss(y, self.forward(X))