import numpy as np
import matplotlib.pyplot as plt
from src.network import Network
from src.layers import Layer

def make_deep_mlp(input_dim=784, hidden_dim=256, depth=5, output_dim=10):
    layers = [Layer(input_dim, hidden_dim, 'relu')]
    for _ in range(depth-1):
        layers.append(Layer(hidden_dim, hidden_dim, 'relu'))
    layers.append(Layer(hidden_dim, output_dim, 'softmax'))
    return layers

# Dummy MNIST-like data (replace with real MNIST for real use)
X = np.random.randn(512, 784)
y = np.eye(10)[np.random.choice(10, 512)]

results = {}
for clip_type, clip_value in [(None, None), ('norm', 1.0), ('value', 1.0)]:
    net = Network()
    for layer in make_deep_mlp():
        net.add_layer(layer)
    net.set_loss('cross_entropy')
    label = f"no_clip" if clip_type is None else f"{clip_type}_{clip_value}"
    print(f"Training with {label}...")
    history = net.train(X, y, epochs=30, lr=0.01, batch_size=64, verbose=False, clip_type=clip_type, clip_value=clip_value or 1.0)
    results[label] = history
    if not history:
        print(f"  [WARNING] No loss history recorded for {label}. Training may have failed or no epochs were run.")
    else:
        print(f"  Final loss: {history[-1]:.4f}")

plt.figure(figsize=(8,5))
for label, hist in results.items():
    plt.plot(hist, label=label)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Gradient Clipping Effect on Deep MLP (Dummy Data)')
plt.legend()
plt.tight_layout()
plt.show()
