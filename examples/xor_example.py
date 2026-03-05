import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.network import Network
from src.layers import Layer
from src.utils import generate_xor_data

if __name__ == "__main__":
    X, y = generate_xor_data()

    net = Network()
    net.add_layer(Layer(2, 4, 'tanh'))
    net.add_layer(Layer(4, 1, 'sigmoid'))
    net.set_loss('mse')

    net.train(X, y, epochs=5000, lr=1.0, print_every=500)
    
    final_loss = net.loss(y, net.forward(X))
    print(f"\nFinal loss: {final_loss:.6f}")

    preds = net.forward(X)
    print("Predictions:\n", preds)
    print("Binary Predictions:\n", (preds > 0.5).astype(int))
