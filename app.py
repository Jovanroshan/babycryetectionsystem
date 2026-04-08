import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp"   # helps reduce memory issues on Render

from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np
import tempfile

app = Flask(__name__)

# Load model and encoder
model = joblib.load("baby_cry_rf_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Audio settings
SAMPLING_RATE = 16000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLING_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc = mfcc.flatten()
    return mfcc.reshape(1, -1)

@app.route("/")
def home():
    return "Baby Cry Detection API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use key name: file"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    file.save(temp_path)

    try:
        features = extract_mfcc(temp_path)

        prediction = model.predict(features)
        predicted_label = encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True)
