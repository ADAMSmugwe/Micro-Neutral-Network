import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.network import Network
from src.layers import Layer
from src.utils import generate_xor_data

def create_network(init_method, activation='relu'):
    layers = [
        Layer(2, 16, activation=activation, init_method=init_method),
        Layer(16, 32, activation=activation, init_method=init_method),
        Layer(32, 16, activation=activation, init_method=init_method),
        Layer(16, 1, activation='sigmoid', init_method=init_method)
    ]
    net = Network(layers=layers, reg_lambda=0.0)
    net.set_loss("mse")
    return net

def train_and_record(net, X, y, epochs=2000, lr=0.1):
    n_samples = X.shape[0]
    losses = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, 4):
            X_batch = X_shuffled[i:i+4]
            y_batch = y_shuffled[i:i+4]
            
            net.train_mode()
            y_pred = net.forward(X_batch)
            net.backward(y_batch, y_pred)
            net.update(lr, momentum=0.9)
        
        if epoch % 10 == 0:
            net.eval_mode()
            full_pred = net.forward(X)
            current_loss = net.loss(y, full_pred)
            losses.append(current_loss)
    
    return losses

if __name__ == "__main__":
    X, y = generate_xor_data()
    
    print("=" * 60)
    print("Weight Initialization Comparison on Deep Network")
    print("=" * 60)
    print("Architecture: 2 -> 16 -> 32 -> 16 -> 1")
    print("Activation: ReLU (hidden), Sigmoid (output)")
    print("Training: 2000 epochs, momentum=0.9, lr=0.1")
    print()
    
    init_methods = {
        'random': 'Naive Random (std=0.01)',
        'xavier': 'Xavier Initialization',
        'he': 'He Initialization (optimal for ReLU)'
    }
    
    results = {}
    
    for method, label in init_methods.items():
        print(f"Training with {label}...")
        net = create_network(method, activation='relu')
        losses = train_and_record(net, X, y, epochs=2000, lr=0.1)
        results[label] = losses
        
        net.eval_mode()
        preds = net.forward(X)
        final_loss = net.loss(y, preds)
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Predictions: {preds.flatten()}")
        print()
    
    plt.figure(figsize=(12, 6))
    
    for label, losses in results.items():
        epochs = np.arange(0, 2000, 10)
        plt.plot(epochs, losses, label=label, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Weight Initialization Comparison on Deep Network (4 layers)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('init_comparison.png', dpi=150)
    print("Plot saved as 'init_comparison.png'")
    plt.show()
