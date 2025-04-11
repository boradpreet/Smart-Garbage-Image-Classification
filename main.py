# main.py
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model("Garbage.keras")

# Class names (adjust if needed)
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Page config
st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️", layout="centered")

# Sidebar
st.sidebar.title("📋 About")
st.sidebar.info(
    """
    This is an AI-powered Garbage Image Classifier app.  
    Upload any garbage image and get an instant classification.  
    Model: `Garbage.keras`  
    Developed by Preet Borad 🚀
    """
)

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: green;'>🧠 Smart Garbage Classification</h1>
    <h4 style='text-align: center;'>Upload an image, and let AI tell you what kind of waste it is!</h4>
    <br>
    """,
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a garbage image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))  # Match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.markdown("---")
    st.markdown(
        f"""
        <div style='padding: 20px; border-radius: 10px; background-color: #f0f8ff; text-align: center;'>
            <h2>🔍 Prediction Result</h2>
            <h1 style='color: #1f77b4;'>🗑️ {predicted_class}</h1>
            <p style='font-size: 18px;'>Confidence: <b>{confidence:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("👈 Upload a garbage image to get started.")
