import numpy as np
from .loss import mse_loss, mse_derivative, cross_entropy_loss, cross_entropy_derivative

class Network:
    def clip_gradients_value(self, clip_value=1.0):
        for layer in self.layers:
            if hasattr(layer, 'dW') and layer.dW is not None:
                layer.dW = np.clip(layer.dW, -clip_value, clip_value)
            if hasattr(layer, 'db') and layer.db is not None:
                layer.db = np.clip(layer.db, -clip_value, clip_value)

    def clip_gradients_norm(self, max_norm=1.0):
        for layer in self.layers:
            if hasattr(layer, 'dW') and layer.dW is not None:
                norm_w = np.linalg.norm(layer.dW)
                if norm_w > max_norm:
                    layer.dW *= max_norm / norm_w
            if hasattr(layer, 'db') and layer.db is not None:
                norm_b = np.linalg.norm(layer.db)
                if norm_b > max_norm:
                    layer.db *= max_norm / norm_b
    def save(self, filepath):
        import pickle
        config = [layer.get_config() for layer in self.layers]
        params = [layer.get_parameters() for layer in self.layers]
        data = {
            'config': config,
            'params': params,
            'loss': self.loss_name
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        import pickle
        from src.layers import Layer, BatchNorm
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        network = cls()
        network.set_loss(data.get('loss', 'mse'))
        for config in data['config']:
            if config['type'] == 'dense':
                layer = Layer(
                    config['n_inputs'],
                    config['n_neurons'],
                    config['activation'],
                    dropout_rate=config.get('dropout_rate', 0.0)
                )
            elif config['type'] == 'batchnorm':
                layer = BatchNorm(
                    config['n_features'],
                    config['eps'],
                    config['momentum']
                )
            else:
                raise ValueError(f"Unknown layer type: {config['type']}")
            network.layers.append(layer)
        for layer, params in zip(network.layers, data['params']):
            layer.set_parameters(params)
        print(f"Model loaded from {filepath}")
        return network
    def __init__(self, layers=None, reg_lambda=0.0):
        self.layers = layers if layers is not None else []
        self.loss_name = None
        self.loss_func = None
        self.loss_derivative = None
        self.reg_lambda = reg_lambda

    def train_mode(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval_mode(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
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
                if hasattr(layer, 'weights') and layer.dW is not None:
                    layer.dW += self.reg_lambda * layer.weights

    def update(self, lr=0.01, momentum=0.0, optimizer='sgd', beta1=0.9, beta2=0.999, eps=1e-8):
        for layer in self.layers:
            # Composite layers (e.g. ResidualBlock) manage their own sub-layer updates
            if hasattr(layer, 'update') and callable(getattr(layer, 'update')):
                layer.update(lr, momentum, optimizer, beta1, beta2, eps)
                continue
            if hasattr(layer, 'weights'):
                if optimizer == 'adam':
                    layer.t += 1
                    g = layer.dW
                    layer.m_weights = beta1 * layer.m_weights + (1 - beta1) * g
                    layer.vw_weights = beta2 * layer.vw_weights + (1 - beta2) * (g ** 2)
                    m_hat = layer.m_weights / (1 - beta1 ** layer.t)
                    v_hat = layer.vw_weights / (1 - beta2 ** layer.t)
                    layer.weights -= lr * m_hat / (np.sqrt(v_hat) + eps)
                    g_b = layer.db
                    layer.m_biases = beta1 * layer.m_biases + (1 - beta1) * g_b
                    layer.vw_biases = beta2 * layer.vw_biases + (1 - beta2) * (g_b ** 2)
                    m_hat_b = layer.m_biases / (1 - beta1 ** layer.t)
                    v_hat_b = layer.vw_biases / (1 - beta2 ** layer.t)
                    layer.biases -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
                else:
                    layer.v_weights = momentum * layer.v_weights - lr * layer.dW
                    layer.v_biases = momentum * layer.v_biases - lr * layer.db
                    layer.weights += layer.v_weights
                    layer.biases += layer.v_biases
            if hasattr(layer, 'filters') and layer.d_filters is not None:
                layer.v_filters = momentum * layer.v_filters - lr * layer.d_filters
                layer.v_biases  = momentum * layer.v_biases  - lr * layer.d_biases
                layer.filters  += layer.v_filters
                layer.biases   += layer.v_biases
            if hasattr(layer, 'gamma'):
                layer.v_gamma = momentum * layer.v_gamma - lr * layer.dgamma
                layer.v_beta = momentum * layer.v_beta - lr * layer.dbeta
                layer.gamma += layer.v_gamma
                layer.beta += layer.v_beta

    def train(self, X, y, epochs=1000, lr=0.01, momentum=0.0, optimizer='sgd', beta1=0.9, beta2=0.999, eps=1e-8, batch_size=32, verbose=True, print_every=100, lr_scheduler=None, clip_type=None, clip_value=1.0, augmentor=None, val_data=None):
        self.train_mode()
        n_samples = X.shape[0]
        history = []
        val_history = []

        for epoch in range(epochs):
            if lr_scheduler is not None:
                lr = lr_scheduler.get_lr(epoch)
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                if augmentor is not None:
                    X_batch, y_batch = augmentor.apply(X_batch, y_batch)

                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                if clip_type == 'value':
                    self.clip_gradients_value(clip_value)
                elif clip_type == 'norm':
                    self.clip_gradients_norm(clip_value)
                self.update(lr, momentum, optimizer=optimizer, beta1=beta1, beta2=beta2, eps=eps)

            full_pred = self.forward(X)
            current_loss = self.loss(y, full_pred)
            history.append(current_loss)

            if val_data is not None:
                self.eval_mode()
                X_val, y_val = val_data
                val_pred = self.forward(X_val)
                val_loss = self.loss(y_val, val_pred)
                val_history.append(val_loss)
                self.train_mode()
            else:
                val_loss = None

            if verbose and epoch % print_every == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch}, Loss: {current_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {lr:.6f}")
                else:
                    print(f"Epoch {epoch}, Loss: {current_loss:.6f}, LR: {lr:.6f}")

        if val_data is not None:
            return history, val_history
        return history

    def accuracy(self, y_true, y_pred):
        preds = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true.astype(int)
        return float(np.mean(preds == labels))

    def train_with_history(self, X_train, y_train, X_val, y_val,
                            epochs=50, lr=0.01, lr_scheduler=None,
                            batch_size=32, patience=5,
                            optimizer='sgd', momentum=0.0,
                            verbose=True):
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        strikes = 0
        n = X_train.shape[0]

        self.train_mode()
        for epoch in range(epochs):
            if lr_scheduler is not None:
                lr = lr_scheduler.get_lr(epoch)

            idx = np.random.permutation(n)
            X_s, y_s = X_train[idx], y_train[idx]

            batch_losses = []
            for i in range(0, n, batch_size):
                Xb, yb = X_s[i:i+batch_size], y_s[i:i+batch_size]
                pred = self.forward(Xb)
                batch_losses.append(float(self.loss(yb, pred)))
                self.backward(yb, pred)
                self.update(lr, momentum, optimizer=optimizer)

            train_loss = float(np.mean(batch_losses))

            self.eval_mode()
            val_pred = self.forward(X_val)
            val_loss = float(self.loss(y_val, val_pred))
            val_acc = self.accuracy(y_val, val_pred)
            self.train_mode()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1} | loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                strikes = 0
            else:
                strikes += 1
                if strikes >= patience:
                    if verbose:
                        print(f"early stopping at epoch {epoch+1}")
                    break

        return history