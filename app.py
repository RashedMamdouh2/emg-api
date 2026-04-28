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

    # لازم 8 صفوف و 40 عمود
    if df.shape != (8, 40):
        return jsonify({
            "error": "CSV must be exactly 8 rows x 40 columns"
        }), 400

    x = df.values.astype("float32")

    # (8,40) -> (1,8,40)
    x = np.expand_dims(x, axis=0)
    x = x.T
    model=get_model()
    pred = model.predict(x, verbose=0)

    cls = int(np.argmax(pred, axis=1)[0]) + 1

    return jsonify({
        "class": cls,
        "probabilities": pred[0].tolist()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  
    app.run(host="0.0.0.0", port=port)
