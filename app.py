import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Set a global random seed for reproducibility
tf.random.set_seed(42)

# Paths to the model and weights
MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\MNIST Model\mnist_digit_recognition_model.h5"

# Load the pre-trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Function to preprocess an image
def preprocess_image(image):
    """
    Preprocess the input image for digit recognition:
    - Convert to grayscale if necessary.
    - Resize to 28x28 pixels.
    - Normalize pixel values to [0, 1].
    - Reshape for model input.
    """
    try:
        if image.ndim == 3:  # Convert color images to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Function to predict a digit from an image
def predict_digit(image):
    """
    Predict the digit from a preprocessed image using the pre-trained model.
    """
    try:
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Streamlit App
st.title("MNIST Digit Recognition")
st.markdown("""
This app predicts handwritten digits (0-9) using a pre-trained Convolutional Neural Network.
Upload an image of a handwritten digit to see the prediction.
""")

# File uploader for digit image
st.subheader("Upload a Digit Image")
uploaded_file = st.file_uploader("Choose an image file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Read and display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        preprocessed_image = preprocess_image(uploaded_image)
        if preprocessed_image is not None:
            with st.spinner("Predicting..."):
                predicted_digit, confidence = predict_digit(preprocessed_image)

            if predicted_digit is not None:
                st.success(f"Predicted Digit: {predicted_digit}")
                st.info(f"Confidence: {confidence * 100:.2f}%")
            else:
                st.error("Failed to predict the digit.")
        else:
            st.error("Failed to preprocess the image. Please try again with a valid image.")
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")

# Footer
st.markdown("""
---
**Project Contributor:**  
- Gunja Shah
""")
