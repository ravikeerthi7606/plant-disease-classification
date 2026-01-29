# 🌿 Plant Disease Classification using Deep Learning

This project is a **Plant Disease Classification system** built using **TensorFlow** and deployed on **Hugging Face Spaces** using **Gradio**.  
It allows users to upload an image of a plant leaf and instantly predicts the disease along with confidence.

---

## 🚀 Live Demo
🔗 **Hugging Face Space:**  
https://ravi-7606-plant-disease-classification-demo.hf.space/?__theme=system&deep_link=-f0aSKQy_44


## 🧠 Model Overview

- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input Size:** 224 × 224 RGB image
- **Output:** Plant disease class + confidence score
- **Dataset:** PlantVillage Dataset

---

## 🧪 How It Works

1. Upload a plant leaf image  
2. Image is resized and normalized  
3. Trained model predicts disease class  
4. Confidence score is displayed  

---

## 🖼️ Supported Input

- Leaf images (JPG / PNG)
- Single leaf per image
- Clear lighting recommended for best accuracy

---

## 📁 Project Structure

plant-disease-classifier/
│
├── app.py # Gradio application
├── model.keras # Trained TensorFlow model
├── class_names.json # Class labels
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## 📦 Requirements

The project uses the following libraries:

```txt
tensorflow
gradio
numpy
Pillow

---
title: Plant Disease Classification DEMO
emoji: 🔥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
short_description: ML model that detect the leaf whether it is infected or not
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
