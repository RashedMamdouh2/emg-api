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
    try:
        file = request.files["file"]

        if not file:
            return jsonify({"error":"No file uploaded"}),400

        df = pd.read_csv(file, header=None)

        values = df.values.flatten()

        if len(values) != 40:
            return jsonify({
                "error":"CSV must contain exactly 40 values"
            }),400

        x = np.array(values, dtype=np.float32).reshape(1,40,1)

        pred = model.predict(x, verbose=0)

        cls = int(np.argmax(pred, axis=1)[0]) + 1

        return jsonify({
            "class": cls,
            "raw_prediction": pred[0].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}),500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  
    app.run(host="0.0.0.0", port=port)
