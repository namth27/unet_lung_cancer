import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os

# Define the IoU metric
def iou(y_true, y_pred, smooth=1):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) - intersection + smooth)

# Preprocess the image
def preprocess_single_image(image_path, resize_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError("Invalid image at path: {}".format(image_path))
    resized_image = cv2.resize(image, resize_size)
    _, binary_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)
    inverted_binary_image = cv2.bitwise_not(binary_image)
    return inverted_binary_image

def overlay_mask_on_image(image, mask, alpha=0.5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask_colored = np.zeros_like(image_rgb)
    mask_colored[:, :, 0] = mask_resized  # Set the mask to red channel
    # Combine the original image with the mask
    overlay_image = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)

    return overlay_image

# Streamlit app
st.title("Lung Cancer Cell Segmentation")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', use_column_width=True, width=100)
    st.write("")

    if st.button("Submit"):
        st.write("Processing...")

        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_image_path = temp_file.name
            image.save(temp_image_path)

        # Load the pre-trained model
        model_path = r'D:\Desktop\Code\u-net-lung-cancer\best_04-08-2024.keras'
        get_custom_objects().update({"iou": iou})
        model = load_model(model_path, custom_objects={'iou': iou})
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou])

        # Preprocess the image
        preprocessed_image = preprocess_single_image(temp_image_path, resize_size=(256, 256))
        preprocessed_image = np.expand_dims(preprocessed_image, axis=(0, -1))

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_mask = prediction.squeeze()

        # Convert the original image to a NumPy array for overlay
        original_image_np = np.array(image)

        # Create the overlay image
        overlay_image = overlay_mask_on_image(original_image_np, predicted_mask * 255)

        # Display the results
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Original Image")
        ax[1].imshow(predicted_mask, cmap='gray')
        ax[1].set_title("Predicted Mask")
        ax[2].imshow(overlay_image)
        ax[2].set_title("Overlay Image")
        st.pyplot(fig)

        # Clean up temporary file
        os.remove(temp_image_path)