# src/network.py
import numpy as np
from .loss import mse_loss, mse_derivative, cross_entropy_loss, cross_entropy_derivative

class Network:
    def __init__(self, layers=None, reg_lambda=0.0):
        self.layers = layers if layers is not None else []
        self.loss_name = None
        self.loss_func = None
        self.loss_derivative = None
        self.reg_lambda = reg_lambda

    def train_mode(self):
        for layer in self.layers:
            layer.training = True

    def eval_mode(self):
        for layer in self.layers:
            layer.training = False

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

    def update(self, lr=0.01, momentum=0.0, optimizer='sgd', beta1=0.9, beta2=0.999, eps=1e-8):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                if optimizer == 'adam':
                    # Adam optimizer
                    layer.t += 1
                    # Weights
                    g = layer.dW
                    layer.m_weights = beta1 * layer.m_weights + (1 - beta1) * g
                    layer.vw_weights = beta2 * layer.vw_weights + (1 - beta2) * (g ** 2)
                    m_hat = layer.m_weights / (1 - beta1 ** layer.t)
                    v_hat = layer.vw_weights / (1 - beta2 ** layer.t)
                    layer.weights -= lr * m_hat / (np.sqrt(v_hat) + eps)
                    # Biases
                    g_b = layer.db
                    layer.m_biases = beta1 * layer.m_biases + (1 - beta1) * g_b
                    layer.vw_biases = beta2 * layer.vw_biases + (1 - beta2) * (g_b ** 2)
                    m_hat_b = layer.m_biases / (1 - beta1 ** layer.t)
                    v_hat_b = layer.vw_biases / (1 - beta2 ** layer.t)
                    layer.biases -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
                else:
                    # SGD or Momentum
                    layer.v_weights = momentum * layer.v_weights - lr * layer.dW
                    layer.v_biases = momentum * layer.v_biases - lr * layer.db
                    layer.weights += layer.v_weights
                    layer.biases += layer.v_biases
            if hasattr(layer, 'gamma'):
                # BatchNorm params (still use momentum/SGD)
                layer.v_gamma = momentum * layer.v_gamma - lr * layer.dgamma
                layer.v_beta = momentum * layer.v_beta - lr * layer.dbeta
                layer.gamma += layer.v_gamma
                layer.beta += layer.v_beta

    def train(self, X, y, epochs=1000, lr=0.01, momentum=0.0, batch_size=32, verbose=True, print_every=100, lr_scheduler=None):
        self.train_mode()
        n_samples = X.shape[0]
        history = []

        for epoch in range(epochs):
            if lr_scheduler is not None:
                lr = lr_scheduler.get_lr(epoch)
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.update(lr, momentum)
            
            if verbose and epoch % print_every == 0:
                full_pred = self.forward(X)
                current_loss = self.loss(y, full_pred)
                print(f"Epoch {epoch}, Loss: {current_loss:.6f}, LR: {lr:.6f}")
                history.append(current_loss)
        
        return history