from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("manual_pretrained_model.h5")

WINDOW = 40
CHANNELS = 8

@app.get("/")
def home():
    return {"status":"running","service":"EMG CNN-LSTM API"}

@app.post("/predict-csv")
def predict_csv():
    try:
        file = request.files["file"]

        if not file:
            return jsonify({"error":"No file uploaded"}),400

        df = pd.read_csv(file)

        if df.shape[1] != CHANNELS:
            return jsonify({"error":"CSV must contain 8 columns"}),400

        data = df.values.astype("float32")

        if len(data) < WINDOW:
            return jsonify({"error":"Need at least 40 rows"}),400

        windows = []
        step = 10

        for i in range(0, len(data)-WINDOW+1, step):
            windows.append(data[i:i+WINDOW])

        X = np.array(windows)

        preds = model.predict(X, verbose=0)

        classes = np.argmax(preds, axis=1) + 1

        final_class = int(np.bincount(classes).argmax())

        return jsonify({
            "class": final_class,
            "windows_used": len(windows),
            "predictions": classes.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}),500


if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
