#!/usr/bin/env python3
"""
Command-line digit prediction tool.

Usage:
    python predict.py <image_path> [model_path]

Examples:
    python predict.py my_digit.png
    python predict.py my_digit.png model/best_model.pkl
"""
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.network import Network
from PIL import Image


def preprocess_image(image_path):
    """Load and preprocess image to (1, 784) float32 array."""
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Invert if white-background image (model trained on white-on-black)
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    return img_array.reshape(1, 784)


def predict_image(image_path, model_path='model/best_model.pkl'):
    model = Network.load(model_path)
    model.eval_mode()

    img_array = preprocess_image(image_path)
    probs = model.forward(img_array)[0]
    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted]) * 100

    print(f"\nImage:      {image_path}")
    print(f"Prediction: {predicted}  ({confidence:.1f}% confidence)\n")
    print("All probabilities:")
    for i, p in enumerate(probs):
        bar = '█' * int(p * 30)
        marker = ' <-- predicted' if i == predicted else ''
        print(f"  {i}: {bar:<30} {p * 100:5.1f}%{marker}")

    return predicted


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'model', 'best_model.pkl')

    if not os.path.exists(image_path):
        print(f"Error: image file not found: {image_path}")
        sys.exit(1)

    predict_image(image_path, model_path)
