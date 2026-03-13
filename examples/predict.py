import numpy as np
from src.network import Network

def main():
    # Load the saved model
    net = Network.load('mnist_model.pkl')
    # Dummy test data (replace with real test data as needed)
    X_new = np.random.randn(10, 784)
    preds = net.forward(X_new)
    print(preds)

if __name__ == '__main__':
    main()
