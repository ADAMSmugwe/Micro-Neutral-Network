from flask import Flask, request, jsonify, render_template
import numpy as np
import sys
import os
import base64
from PIL import Image
import io

# Add project root to path so we can import the src package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.network import Network

app = Flask(__name__)

# Load the trained model once at startup
print("Loading model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pkl')
model = Network.load(MODEL_PATH)
model.eval_mode()  # disable dropout, use batch norm running stats
print("Model loaded and ready.")


def preprocess_image(image_bytes):
    """Convert uploaded image bytes to (1, 784) array matching training format."""
    img = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    img = img.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img, dtype=np.float32) / 255.0

    # Invert if needed: model was trained on white digits on black background
    # Canvas draws white on black, so we match that directly.
    # If uploading an image that has black digits on white, invert it:
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    return img_array.reshape(1, 784)  # MLP expects flat input


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    image_bytes = None

    if request.files and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file selected'}), 400
        image_bytes = file.read()

    elif request.is_json and request.json and 'image' in request.json:
        # Base64-encoded data URL from canvas: "data:image/png;base64,<data>"
        raw = request.json['image']
        if ',' in raw:
            raw = raw.split(',', 1)[1]
        image_bytes = base64.b64decode(raw)

    else:
        return jsonify({'error': 'No image provided. Send a file or base64 image.'}), 400

    try:
        img_array = preprocess_image(image_bytes)
        predictions = model.forward(img_array)         # (1, 10)
        probs = predictions[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'probabilities': {str(i): round(float(p) * 100, 2) for i, p in enumerate(probs)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Accept multiple uploaded images and return predictions for all."""
    if not request.files:
        return jsonify({'error': 'No files provided'}), 400

    results = []
    for key, file in request.files.items():
        try:
            img_array = preprocess_image(file.read())
            probs = model.forward(img_array)[0]
            pred = int(np.argmax(probs))
            results.append({
                'file': file.filename,
                'prediction': pred,
                'confidence': round(float(probs[pred]) * 100, 2)
            })
        except Exception as e:
            results.append({'file': file.filename, 'error': str(e)})

    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
