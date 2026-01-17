from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("app/model/pan_detector_model.h5")

UPLOAD_FOLDER = "app/static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "✅ Authentic PAN Card" if pred < 0.5 else "❌ Tampered PAN Card"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["pan_image"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    result = predict_image(file_path)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
