from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)


model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("manual_pretrained_model.h5")
    return model


WINDOW = 40
CHANNELS = 8

@app.get("/")
def home():
    return {"status": "running", "service": "EMG CNN-LSTM API"}

@app.post("/predict-csv")
def predict_csv():
    file = request.files["file"]
    df = pd.read_csv(file, header=None)
    if df.shape == (8, 40):
        df = df.T  
  
    if df.shape != (40, 8):
        return jsonify({
            "error": "invalid file shape"
        }), 400
    x = df.values.astype("float32")
    x = np.expand_dims(x, axis=0)
    model = get_model()
    pred = model.predict(x, verbose=0)

    cls = int(np.argmax(pred, axis=1)[0]) + 1

    return jsonify({
        "class": cls
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  
    app.run(host="0.0.0.0", port=port)
