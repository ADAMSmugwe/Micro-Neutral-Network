import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layers import Layer

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

if __name__ == "__main__":
    layer = Layer(3, 5, 'relu')
    X = np.random.randn(4, 3)
    output = layer.forward(X)
    print("Input shape:", X.shape)
    print("Output shape:", output.shape)

    net = Network()
    net.add_layer(Layer(3, 5, 'relu'))
    net.add_layer(Layer(5, 2, 'sigmoid'))
    out = net.forward(X)
    print("Network output shape:", out.shape)
