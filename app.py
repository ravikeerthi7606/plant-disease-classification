import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json

# Load saved model
model = tf.keras.models.load_model("model.keras",compile=False)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    idx = np.argmax(predictions[0])
    confidence = predictions[0][idx]

    return class_names[idx], confidence

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path).resize((300,300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    label, conf = predict_image(file_path)
    result_label.config(text=f"Disease: {label}\nConfidence: {conf*100:.2f}%")

# UI
root = tk.Tk()
root.title("Plant Disease Classifier")
root.geometry("450x550")

title = tk.Label(root, text="Plant Disease Detector", font=("Arial", 16))
title.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()