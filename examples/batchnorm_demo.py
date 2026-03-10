import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.network import Network
from src.layers import Layer, BatchNorm
from src.utils import generate_xor_data

if __name__ == "__main__":
    X, y = generate_xor_data()
    
    print("=" * 60)
    print("Testing Deep Network WITHOUT Batch Normalization")
    print("=" * 60)
    
    net_no_bn = Network(reg_lambda=0.0)
    net_no_bn.add_layer(Layer(2, 16, 'tanh'))
    net_no_bn.add_layer(Layer(16, 16, 'tanh'))
    net_no_bn.add_layer(Layer(16, 16, 'tanh'))
    net_no_bn.add_layer(Layer(16, 1, 'sigmoid'))
    net_no_bn.set_loss("mse")
    
    history_no_bn = net_no_bn.train(
        X, y,
        epochs=2000,
        lr=0.5,
        momentum=0.9,
        batch_size=4,
        verbose=True,
        print_every=400
    )
    
    print("\n" + "=" * 60)
    print("Testing Deep Network WITH Batch Normalization")
    print("=" * 60)
    
    net_with_bn = Network(reg_lambda=0.0)
    net_with_bn.add_layer(Layer(2, 16, 'tanh'))
    net_with_bn.add_layer(BatchNorm(16))
    net_with_bn.add_layer(Layer(16, 16, 'tanh'))
    net_with_bn.add_layer(BatchNorm(16))
    net_with_bn.add_layer(Layer(16, 16, 'tanh'))
    net_with_bn.add_layer(BatchNorm(16))
    net_with_bn.add_layer(Layer(16, 1, 'sigmoid'))
    net_with_bn.set_loss("mse")
    
    history_with_bn = net_with_bn.train(
        X, y,
        epochs=2000,
        lr=0.5,
        momentum=0.9,
        batch_size=4,
        verbose=True,
        print_every=400
    )
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_no_bn, label='Without BatchNorm', linewidth=2)
    plt.plot(history_with_bn, label='With BatchNorm', linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Checkpoint')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    
    net_no_bn.eval_mode()
    preds_no_bn = net_no_bn.forward(X)
    
    net_with_bn.eval_mode()
    preds_with_bn = net_with_bn.forward(X)
    
    x_vals = np.arange(4)
    width = 0.35
    
    plt.bar(x_vals - width/2, preds_no_bn.ravel(), width, label='Without BatchNorm', alpha=0.8)
    plt.bar(x_vals + width/2, preds_with_bn.ravel(), width, label='With BatchNorm', alpha=0.8)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary')
    plt.xticks(x_vals, ['[0,0]', '[0,1]', '[1,0]', '[1,1]'])
    plt.ylabel('Predicted Output')
    plt.title('Final Predictions')
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Final loss without BatchNorm: {history_no_bn[-1]:.6f}")
    print(f"Final loss with BatchNorm: {history_with_bn[-1]:.6f}")
    print(f"\nPredictions without BatchNorm:\n{preds_no_bn}")
    print(f"Binary: {(preds_no_bn > 0.5).astype(int).T}")
    print(f"\nPredictions with BatchNorm:\n{preds_with_bn}")
    print(f"Binary: {(preds_with_bn > 0.5).astype(int).T}")
