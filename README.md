# 🩻 Chest X-Ray Pneumonia Detection

A deep learning-based image classification project for detecting **Pneumonia** from chest X-ray images using Convolutional Neural Networks and Transfer Learning.

The project compares a custom CNN with **MobileNetV2** and **ResNet50**, followed by fine-tuning of the pretrained models. **Grad-CAM** is also used to improve model interpretability.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool.

---

🌐 Live Demo

🚀 Try the Streamlit Application:



---

## 📌 Project Overview

Pneumonia is a respiratory infection that can be identified from abnormalities visible in chest X-ray images. This project explores how deep learning can be used to automatically classify chest X-rays into two categories:

* **Normal**
* **Pneumonia**

The main objective is to build a model with strong classification performance while giving particular importance to **recall**, since missing a pneumonia case can be more serious than producing a false positive.

---

## 🎯 Objectives

* Load and preprocess chest X-ray images.
* Analyze class distribution and image properties.
* Apply image normalization and training augmentation.
* Build a custom CNN model.
* Apply transfer learning using MobileNetV2 and ResNet50.
* Fine-tune the pretrained models.
* Handle class imbalance using class weighting.
* Evaluate models using multiple classification metrics.
* Compare model performance.
* Use Grad-CAM for model interpretability.
* Deploy the final model using Streamlit.

---

## 📂 Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

Dataset source:

**Kaggle — Chest X-Ray Images (Pneumonia)**
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains chest X-ray images categorized into:

* Normal
* Pneumonia

The original dataset contains separate training, validation, and testing folders. Because the provided validation set was very small, part of the training data was used to create a more useful validation split.

---

## 🔬 Methodology

### 1. Data Exploration

The dataset was analyzed for:

* Class distribution
* Image dimensions
* Image channels
* Corrupted images
* Sample image visualization

### 2. Preprocessing

The images were:

* Resized to **224 × 224**
* Normalized appropriately
* Prepared using model-specific preprocessing where required

Training augmentation included:

* Horizontal flipping
* Random rotation
* Random zoom
* Random brightness adjustment

### 3. Class Imbalance

The dataset contains more Pneumonia images than Normal images.

To reduce the effect of this imbalance, **class weights** were applied during model training.

---

## 🧠 Models

### Custom CNN

A custom Convolutional Neural Network was developed using multiple convolutional and pooling blocks followed by fully connected layers.

### MobileNetV2

MobileNetV2 pretrained on ImageNet was used for transfer learning.

The process involved:

1. Freezing the pretrained base.
2. Training a new classification head.
3. Unfreezing the top layers.
4. Fine-tuning with a low learning rate.

### ResNet50

ResNet50 pretrained on ImageNet was similarly used for transfer learning and fine-tuning.

---

## 📊 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Recall was given particular importance because the application focuses on detecting pneumonia cases.

---

## 🏆 Model Comparison

| Model                  |   Accuracy | Precision |     Recall |   F1 Score |    ROC-AUC |
| ---------------------- | ---------: | --------: | ---------: | ---------: | ---------: |
| CNN                    |     77.72% |    74.46% |     97.95% |     84.61% |     91.15% |
| MobileNetV2            |     86.54% |    82.97% |     98.72% |     90.16% |     96.29% |
| MobileNetV2 Fine-tuned | **86.54%** |    82.83% | **98.97%** | **90.19%** | **96.46%** |
| ResNet50               |     81.89% |    78.21% |     98.46% |     87.17% |     95.44% |
| ResNet50 Fine-tuned    |     83.49% |    80.21% |     97.69% |     88.09% |     95.71% |

### Best Model

The **fine-tuned MobileNetV2** was selected as the final model.

It achieved:

* **Accuracy:** 86.54%
* **Precision:** 82.83%
* **Recall:** 98.97%
* **F1 Score:** 90.19%
* **ROC-AUC:** 96.46%

Its high recall makes it particularly suitable for the project's objective of minimizing missed pneumonia cases.

---

## 🔎 Grad-CAM Interpretability

Grad-CAM was used to visualize the image regions that contributed to the model's predictions.

This provides a more interpretable view of the model and helps assess whether predictions are influenced by meaningful regions of the chest X-ray.

---

## 🌐 Streamlit Application

The trained MobileNetV2 model is integrated into a Streamlit web application.

The application allows users to:

1. Upload a chest X-ray image.
2. Process the image automatically.
3. Generate a prediction.
4. Display the predicted class.
5. Display the model's confidence.

### Application Flow

```text
Upload Chest X-Ray
        ↓
Resize to 224 × 224
        ↓
MobileNetV2 Preprocessing
        ↓
Fine-tuned MobileNetV2
        ↓
Prediction
        ↓
Normal / Pneumonia
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pillow
* Streamlit

---

## 📁 Project Structure

```text
Chest-XRay-Pneumonia-Detection/
│
├── app.py
├── best_model.keras
├── requirements.txt
├── README.md
├── .gitignore
└── chest_xray_pneumonia.ipynb
```

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Open the project folder

```bash
cd Chest-XRay-Pneumonia-Detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📌 Future Improvements

Possible future improvements include:

* Training with a larger and more diverse dataset.
* Additional hyperparameter tuning.
* Threshold optimization based on clinical objectives.
* More extensive Grad-CAM analysis.
* External validation on an independent dataset.
* Improved deployment and monitoring.

---

Author: Supriya Raj

---

## ⚠️ Disclaimer

This application is **not intended to replace medical professionals or clinical diagnosis**.

Predictions generated by the model should not be used for medical decisions.
