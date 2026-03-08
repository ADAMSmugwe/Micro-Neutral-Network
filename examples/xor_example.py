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

    net = Network(
        layers=[
            Layer(2, 4, 'tanh', dropout_rate=0.0),
            Layer(4, 1, 'sigmoid', dropout_rate=0.0)
        ],
        reg_lambda=0.0
    )
    net.set_loss("mse")

    history = net.train(
        X,
        y,
        epochs=5000,
        lr=1.0,
        momentum=0.9,
        batch_size=4,
        verbose=True,
        print_every=500,
    )

    plt.plot(history)
    plt.title("XOR training loss (no dropout)")
    plt.xlabel("Checkpoint")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    net.eval_mode()
    preds = net.forward(X)
    print("Final loss:", net.loss(y, preds))
    print("Preds:\n", preds)
    print("Binary:\n", (preds > 0.5).astype(int))