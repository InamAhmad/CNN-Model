
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define the path where the saved model is located (in .h5 format)
model_save_path_h5 = '/content/drive/MyDrive/archive/cnn_model.h5'

# Load the trained CNN model
try:
    model = tf.keras.models.load_model(model_save_path_h5)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

# Define image size
img_height = 128
img_width = 128

# Define class names
# Assuming the model predicts 0 for 'cats' and 1 for 'dogs' based on previous training output
class_names = ['cat', 'dog']

# Set up the Streamlit application title and header
st.title("Cat and Dog Image Classifier")
st.header("Upload an image of a cat or dog for classification")

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the uploaded image
    # Resize the image
    image = image.resize((img_height, img_width))
    # Convert the image to a NumPy array
    img_array = np.array(image)
    # Normalize the pixel values to the range [0, 1]
    img_array = img_array / 255.0
    # Expand the dimensions of the array to match the model's input shape (add a batch dimension)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)

    # Interpret the prediction result
    # For a single output unit with sigmoid activation, the output is a probability
    # If probability > 0.5, it's likely the positive class (dogs)
    score = predictions[0][0]
    if score > 0.5:
        predicted_class = class_names[1] # 'dog'
        confidence = score
    else:
        predicted_class = class_names[0] # 'cat'
        confidence = 1 - score # Confidence in 'cat'

    # Display the prediction result
    st.write(f"Prediction: This image most likely belongs to a **{predicted_class}** with a confidence of **{confidence:.2f}**.")

elif uploaded_file is None and model is not None:
    st.write("Please upload an image to classify.")
elif model is None:
    st.write("Model could not be loaded. Please check the model path and ensure it's in .h5 format.")
