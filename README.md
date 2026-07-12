# 📦 VisionDNA AI Package Inspector

An AI-powered **Industrial Package Damage Detection System** built using **Computer Vision**, **TensorFlow**, **EfficientNetB0**, and **Streamlit**.

The application automatically classifies package images as **Damaged** or **Intact**, helping automate quality inspection in warehouses, logistics, and manufacturing industries.

---

## 🚀 Live Demo

🌐 **Streamlit App**

YOUR_STREAMLIT_APP_URL

---

## 📷 Project Preview

> *(Add screenshots after deployment)*

- Home Page
- Upload Image
- Prediction Result
- Confidence Score

---

# 📌 Problem Statement

Manual package inspection is time-consuming, inconsistent, and prone to human error.

Industries require an automated solution capable of detecting damaged packages accurately using Computer Vision.

---

# 🎯 Objective

Develop an AI-powered inspection system capable of:

- Detecting damaged packages
- Classifying packages as Damaged or Intact
- Providing confidence scores
- Assisting warehouse quality inspection
- Reducing manual inspection effort

---

# ✨ Features

- 📦 Upload package images
- 🤖 AI-powered damage detection
- 📊 Confidence percentage
- 📈 Real-time prediction
- 🖥 Interactive Streamlit interface
- ⚡ Fast inference using EfficientNetB0
- 📱 Responsive UI

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Streamlit
- EfficientNetB0
- NumPy
- OpenCV
- Pillow
- Pandas
- Scikit-Learn

---

# 🧠 Deep Learning Model

- Transfer Learning
- EfficientNetB0
- Binary Classification
- Optimizer: Adam
- Loss Function: Binary Crossentropy

---

# 📂 Dataset

**Industrial Quality Control of Packages**

Kaggle Dataset:

https://www.kaggle.com/datasets/christianvorhemus/industrial-quality-control-of-packages

Dataset Structure:

```
dataset/

damaged/
    top/
    side/

intact/
    top/
    side/
```

---

# 📁 Project Structure

```
VisionDNA/

│── app.py
│── train.py
│── prepare_dataset.py
│── requirements.txt
│── README.md

├── models/
│      package_classifier.keras

├── dataset/

├── dataset_split/

```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Sunmathi-A77/VisionDNA-AI-Package-Inspector.git
```

Go to the project folder

```bash
cd VisionDNA-AI-Package-Inspector
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run the Application

Launch Streamlit

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 🏋 Train the Model

Prepare dataset

```bash
python prepare_dataset.py
```

Train model

```bash
python train.py
```

The trained model will be saved in

```
models/package_classifier.keras
```

---

# 📊 Model Output

The application predicts:

- ✅ Intact Package

or

- ❌ Damaged Package

along with

- Confidence Score
- Recommendation

---

# 💼 Applications

- Warehouse Inspection
- Logistics Industry
- Manufacturing
- Packaging Industry
- Supply Chain Quality Control
- Smart Warehouses

---

# 📈 Future Enhancements

- Grad-CAM Explainable AI
- Webcam Inspection
- Video-based Inspection
- Multi-class Damage Detection
- Mobile Application
- Cloud Deployment
- Barcode Integration
- OCR-based Label Verification

---

# 🙏 Acknowledgements

- TensorFlow
- Streamlit
- Kaggle
- EfficientNetB0
- OpenCV

---

# 🌟 If you like this project

Please ⭐ the repository.
