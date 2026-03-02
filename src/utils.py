import numpy as np

def generate_xor_data():
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([[0], [1], [1], [0]])
    return features, labels
