# examples/xor_example.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.network import Network
from src.layers import Layer
from src.utils import generate_xor_data

if __name__ == "__main__":
    X, y = generate_xor_data()

    print("Training with SGD...")
    net_sgd = Network(
        layers=[Layer(2, 4, 'tanh'), Layer(4, 1, 'sigmoid')],
        reg_lambda=0.001
    )
    net_sgd.set_loss('mse')
    sgd_history = net_sgd.train(X, y, epochs=3000, lr=0.1, momentum=0.0, batch_size=4, print_every=1000)

    print("\nTraining with Momentum...")
    net_momentum = Network(
        layers=[Layer(2, 4, 'tanh'), Layer(4, 1, 'sigmoid')],
        reg_lambda=0.001
    )
    net_momentum.set_loss('mse')
    momentum_history = net_momentum.train(X, y, epochs=3000, lr=0.1, momentum=0.9, batch_size=4, print_every=1000)

    plt.figure(figsize=(10, 6))
    plt.plot(sgd_history, label='SGD Loss')
    plt.plot(momentum_history, label='Momentum Loss')
    plt.title('Loss Curve Comparison: SGD vs. Momentum')
    plt.xlabel('Epoch (x1000)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nFinal Predictions (Momentum):")
    preds = net_momentum.forward(X)
    print(preds)
    print("Binary Predictions:\n", (preds > 0.5).astype(int))