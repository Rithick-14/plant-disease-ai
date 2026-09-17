import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "plant_disease_model.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False,
        safe_mode=False
    )

    return model

@st.cache_data
def load_class_names():

    with open("class_names.json", "r") as f:
        return json.load(f)


model = load_model()
class_names = load_class_names()


# ============================================================
# DISEASE NAME FORMATTER
# ============================================================

def format_disease_name(name):

    name = name.replace("___", " ")
    name = name.replace("__", " ")
    name = name.replace("_", " ")

    return name


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to MobileNetV2 input size
    image = image.resize((224, 224))

    # Convert to NumPy
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # IMPORTANT:
    # Do NOT use preprocess_input here.
    # The trained model already contains preprocessing.

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    # Top prediction
    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(prediction[predicted_index])

    # Top 5 predictions
    top_indices = np.argsort(prediction)[::-1][:5]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({
            "class": format_disease_name(class_names[index]),
            "confidence": float(prediction[index])
        })

    return predicted_class, confidence, top_predictions


# ============================================================
# HEADER
# ============================================================

st.title("🌱 Plant Disease AI")

st.markdown(
    """
    ### AI-powered plant disease detection
    Upload a leaf image and let the trained MobileNetV2 model
    analyze it across **15 plant-health classes**.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write("**Architecture:** MobileNetV2")

    st.write("**Classes:** 15")

    st.write("**Validation Accuracy:** 95.82%")

    st.divider()

    st.info(
        "For best results, use a clear image where the "
        "leaf is visible and reasonably well lit."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📷 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader("📸 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with col2:

        st.subheader("🤖 AI Analysis")

        with st.spinner("Analyzing leaf..."):

            predicted_class, confidence, top_predictions = (
                predict_image(image)
            )

        display_name = format_disease_name(
            predicted_class
        )

        confidence_percent = confidence * 100

        st.success(
            f"Prediction: {display_name}"
        )

        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )

        # Confidence warning
        if confidence >= 0.80:

            st.success(
                "🟢 High-confidence prediction"
            )

        elif confidence >= 0.60:

            st.warning(
                "🟡 Moderate-confidence prediction"
            )

        else:

            st.error(
                "🔴 Low-confidence prediction — "
                "try another clearer image."
            )


    # ========================================================
    # TOP 5 PREDICTIONS
    # ========================================================

    st.divider()

    st.subheader("📊 Top 5 Predictions")

    for item in top_predictions:

        name = item["class"]

        score = item["confidence"]

        st.write(
            f"**{name}** — {score * 100:.2f}%"
        )

        st.progress(
            min(score, 1.0)
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ This AI model is intended for educational and "
        "demonstration purposes. Predictions should not be "
        "treated as a definitive agricultural diagnosis."
    )

else:

    st.info(
        "👆 Upload a leaf image above to start the analysis."
    )