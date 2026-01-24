import os
import random
import pickle
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 224
IMAGES_DIR = "images"

st.set_page_config(page_title="AI Fashion Recommender", layout="centered")
st.title("👗 AI Fashion Recommendation System")
st.write("Upload a face image to get clothing recommendations based on skin tone.")

# -------------------------------
# LOAD CLOTHES MAPPING (.pkl)
# -------------------------------
@st.cache_resource
def load_clothes_mapping():
    with open("clothes_mapping.pkl", "rb") as f:
        return pickle.load(f)

clothes_mapping = load_clothes_mapping()

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    gender_model = tf.keras.models.load_model("gender_model.h5", compile=False)
    skin_model = tf.keras.models.load_model("skin_model.h5", compile=False)
    return gender_model, skin_model

gender_model, skin_model = load_models()

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

skin_labels = {0: "Light", 1: "Medium", 2: "Dark"}

# -------------------------------
# STREAMLIT UI
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing image..."):
            img_input = preprocess_image(image)

            # -------- Gender Prediction --------
            gender_pred = gender_model.predict(img_input, verbose=0)
            gender_class = int(np.argmax(gender_pred))

            if gender_class != 1:
                st.warning("⚠️ This demo currently supports **female users only**.")
            else:
                # -------- Skin Tone Prediction --------
                skin_pred = skin_model.predict(img_input, verbose=0)
                skin_class = int(np.argmax(skin_pred))

                st.success(f"🎨 Detected Skin Tone: **{skin_labels[skin_class]}**")
                st.subheader("👗 Recommended Clothes")

                recommendations = clothes_mapping.get(skin_class, [])

                if not recommendations:
                    st.error("No clothing recommendations found.")
                else:
                    # Select 3–4 random clothes
                    num_to_show = min(4, len(recommendations))
                    selected_clothes = random.sample(recommendations, num_to_show)

                    cols = st.columns(1)

                    for i, cloth in enumerate(selected_clothes):
                        img_path = os.path.join(IMAGES_DIR, cloth["img"])

                        with cols[0]:
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)

                                # Download button (Android & iPhone friendly)
                                with open(img_path, "rb") as f:
                                    img_bytes = f.read()

                                st.download_button(
                                    label="⬇️ Download",
                                    data=img_bytes,
                                    file_name=cloth["img"],
                                    mime="image/jpeg"
                                )
                            else:
                                st.error(f"Image not found: {cloth['img']}")

                            st.markdown(f"**{cloth['name']}**")
