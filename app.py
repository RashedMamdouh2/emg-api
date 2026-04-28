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

    # قراءة الملف بدون عناوين (header=None)
    df = pd.read_csv(file, header=None)

    # التحقق إذا كانت البيانات 8 صفوف و 40 عموداً، ثم قلبها (Transpose)
    if df.shape == (8, 40):
        df = df.T  # هنا يتم تحويلها من (8, 40) إلى (40, 8)
    
    # التأكد من أن النتيجة النهائية هي 40 صفاً و 8 أعمدة
    if df.shape != (40, 8):
        return jsonify({
            "error": "الملف يجب أن يحتوي على (8 صفوف × 40 عمود) أو (40 صف × 8 أعمدة)"
        }), 400

    # تحويل البيانات إلى مصفوفة numpy ونوع البيانات float32
    x = df.values.astype("float32")

    # إضافة بُعد الـ Batch ليصبح الشكل (1, 40, 8)
    x = np.expand_dims(x, axis=0)

    # استدعاء المودل والتنبؤ
    model = get_model()
    pred = model.predict(x, verbose=0)

    # الحصول على رقم الفئة (من 1 إلى 7)
    cls = int(np.argmax(pred, axis=1)[0]) + 1

    return jsonify({
        "class": cls,
        "probabilities": pred[0].tolist()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  
    app.run(host="0.0.0.0", port=port)
