import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("model.keras", compile=False)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

def predict(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    predicted_class = class_names[np.argmax(preds)]

    return {
        predicted_class: confidence
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="🌿 Plant Disease Classification",
    description="Upload a plant leaf image to detect disease using deep learning"
)

demo.launch()
