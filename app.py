import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "lung_cancer_secret"

# Load trained model
MODEL_PATH = os.path.join("models", "lung_cancer_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file!", "warning")
            return redirect(request.url)

        file_path = os.path.join("static", "images", file.filename)
        file.save(file_path)  # Save uploaded file
        
        img = preprocess_image(file_path)
        prediction = model.predict(img)[0][0]
        
        result = "No Cancer" if prediction > 0.5 else "Cancer Detected"
        confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

        return render_template("predict.html", file_path=file_path, result=result, confidence=confidence)

    return render_template("predict.html")

# Recommendations Page
@app.route("/recommendations")
def recommendations():
    return render_template("recommendations.html")

if __name__ == "__main__":
    app.run(debug=True)


