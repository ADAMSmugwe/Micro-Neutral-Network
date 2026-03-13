import numpy as np
from src.network import Network
from src.layers import Layer

# Dummy training data (replace with real MNIST data for real use)
X_train = np.random.randn(100, 784)
y_train = np.eye(10)[np.random.choice(10, 100)]

# Build a simple network
net = Network()
net.add_layer(Layer(784, 128, 'relu', dropout_rate=0.2))
net.add_layer(Layer(128, 64, 'relu', dropout_rate=0.2))
net.add_layer(Layer(64, 10, 'softmax'))
net.set_loss('cross_entropy')

# Train for a few epochs
net.train(X_train, y_train, epochs=5, lr=0.001, batch_size=16, verbose=True)

# Save the trained model
net.save('mnist_model.pkl')
print('Model trained and saved as mnist_model.pkl')
