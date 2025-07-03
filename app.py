from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import io
import tensorflow_hub as hub

app = Flask(__name__)
CORS(app)

# Load models
chonk_model = tf.keras.models.load_model(
    'chonk-chart-model.h5'
)

inst_model = tf.keras.models.load_model(
    "inst-classification-clean.keras",
    custom_objects={"KerasLayer": hub.KerasLayer}
)

# Static data
all_breeds = [
    "Abyssinian", "American Shorthair", "Balinese", "Bengal", "Birman",
    "British Shorthair", "Burmese", "Chartreux", "Cornish Rex", "Devon Rex",
    "Egyptian Mau", "Exotic Shorthair", "Himalayan", "Maine Coon", "Manx",
    "Munchkin", "Norwegian Forest", "Ocicat", "Oriental", "Persian", "Ragdoll",
    "Russian Blue", "Savannah", "Scottish Fold", "Siamese", "Siberian",
    "Singapura", "Sphynx", "Tonkinese", "Turkish Angora"
]
all_sexes = ["Female", "Male"]
class_labels = [
    "A fine boi", "A heckin chonker", "He chomnk", "HEFTYCHONK", "MEGACHONKER", "OH LAWD HE COMIN", "Too smol"
]
instrument_labels = ["Electric guitar", "Acoustic guitar", "drums", "keyboard"]
IMG_SIZE = (224, 224)

@app.route('/predict', methods=['POST'])
def predict_chonk():
    data = request.get_json()
    weight = int(data['weight'])
    breed = data['breed']
    sex = data['sex']

    breed_encoded = [1 if b == breed else 0 for b in all_breeds]
    sex_encoded = [1 if s == sex else 0 for s in all_sexes]

    features = [weight] + breed_encoded + sex_encoded
    input_tensor = np.array([features], dtype=np.float32)

    prediction = chonk_model.predict(input_tensor)[0]
    class_index = int(np.argmax(prediction))
    class_label = class_labels[class_index]

    return jsonify({"prediction": class_label})


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.pad(img, IMG_SIZE, color=(255, 255, 255))
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/predict2', methods=['POST'])
def predict_instrument():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image_data = file.read()
    processed_image = preprocess_image(image_data)

    prediction = inst_model.predict(processed_image)[0]
    class_index = int(np.argmax(prediction))
    class_name = instrument_labels[class_index]
    confidence = float(prediction[class_index])

    return jsonify({"prediction": class_name, "confidence": confidence})


if __name__ == '__main__':
    app.run(debug=True)
