import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🩻",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Notebook/best_model.keras")


model = load_model()


# --------------------------------------------------
# Title and Introduction
# --------------------------------------------------

st.title("🩻 Chest X-Ray Pneumonia Detection")

st.write(
    "Upload a chest X-ray image to classify it as "
    "**Normal** or **Pneumonia** using a fine-tuned "
    "MobileNetV2 deep learning model."
)

st.info(
    "This application is intended for educational and "
    "demonstration purposes only. It is not a medical "
    "diagnostic tool."
)


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    # Resize image
    image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(image).astype("float32")

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Display result
    if prediction >= 0.5:

        confidence = prediction * 100

        st.error(
            f"🔴 Prediction: Pneumonia"
        )

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )

    else:

        confidence = (1 - prediction) * 100

        st.success(
            f"🟢 Prediction: Normal"
        )

        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Deep Learning Project | CNN + MobileNetV2 + ResNet50 | "
    "Chest X-Ray Pneumonia Detection"
)