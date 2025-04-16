# 🯡 Lung Cancer Detection System using Deep Learning

This project is a **deep learning-based web application** that detects lung cancer from chest X-ray images using **Convolutional Neural Networks (CNN)**. It is designed for healthcare professionals and researchers to assist in early cancer detection, improving the chances of timely treatment.


---

## 🚀 Features

- 🧠 Trained CNN model for binary classification (Cancer / No Cancer)
- 📈 Model performance evaluation using Accuracy, Precision, Recall, and F1-Score
- 🖼️ Real-time image upload and prediction via web interface
- 🌐 Interactive Flask-based web application

---

## 📾 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/lung-cancer-detection.git
   cd lung-cancer-detection
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   python app.py
   ```

---

## 📁 Project Structure

```
Lung-Cancer-Detection-System/
├── app.py
├── requirements.txt
├── models/
│   └── lung_cancer_model.h5
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── dataset_augmented/ (optional - or link externally)
├── static/
│   ├── css/
│   ├── images/
│   └── heatmap/
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── recommendations.html
├── README.md

```

---

## 📷 Model Architecture

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])
```

---

## 📊 Evaluation Metrics

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.98  |
| Precision | 0.96  |
| Recall    | 1.00  |
| F1 Score  | 0.98  |

---

## ⚠️ Limitations

- The dataset used is limited in size and variety.
- Predictions are based on static 2D X-ray images; CT scans could improve accuracy.
- Not intended for clinical diagnosis without expert validation.

---

## 🔮 Future Work

- Integrate CT scan support.
- Improve model with larger, more diverse datasets.
- Deploy to cloud (Heroku, Render, etc.).
- Add multilingual support for global use.

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle - Lung Cancer X-ray Dataset](https://www.kaggle.com/datasets)
- **Libraries**: TensorFlow, Keras, OpenCV, Flask, NumPy, Matplotlib
- Special thanks to all contributors and mentors for their support.

---


