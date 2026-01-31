import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import os
import random

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = 224
IMAGES_DIR = "images"

st.set_page_config(
    page_title="Women Fashion AI",
    layout="wide"
)

st.title("👩 Women Clothing Recommendation System")
st.write("AI-based fashion recommendation using skin tone detection")

# -------------------------------
# LOAD MODEL & DATA
# -------------------------------
@st.cache_resource
def load_skin_model():
    return tf.keras.models.load_model("skin_model.h5")

@st.cache_data
def load_clothes_mapping():
    with open("clothes_mapping.pkl", "rb") as f:
        return pickle.load(f)

skin_model = load_skin_model()
clothes_mapping = load_clothes_mapping()

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------------
# LABELS & EXPLANATION
# -------------------------------
skin_labels = {
    0: "Light Skin",
    1: "Medium Skin",
    2: "Dark Skin"
}

skin_explanations = {
    0: "✨ Light skin tones look best in pastel, floral and soft colors.",
    1: "✨ Medium skin tones are enhanced by rich, earthy and royal colors.",
    2: "✨ Dark skin tones shine in bold, vibrant and high-contrast colors."
}

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "📄 Project Description",
    "🎯 AI Recommendation",
    "👗 All Women Clothes"
    ])

# =====================================================
# TAB 1 – PROJECT DESCRIPTION
# =====================================================
with tab1:
    st.subheader("📄 Women Clothing Recommendation System")

    st.markdown("""
    ### 🔍 Project Overview
    The **Women Clothing Recommendation System** is an AI-based web application
    designed to recommend suitable clothing for women based on their **skin tone**.
    The system uses **deep learning techniques** to analyze a facial image uploaded
    by the user and predicts the skin tone category.

    ---
    ### ⚙️ How the System Works
    1. User uploads a face image
    2. Image is resized and normalized
    3. CNN model predicts skin tone
    4. Clothing data is filtered based on prediction
    5. Recommendations are displayed with images
    6. Category-based filtering improves personalization

    ---
    ### 🎯 Objectives
    - Provide personalized fashion recommendations
    - Apply AI in real-world fashion scenarios
    - Enhance user confidence in clothing selection
    - Build a user-friendly AI web application

    ---
    ### 🛠 Technologies Used
    - Python  
    - Streamlit  
    - TensorFlow / Keras  
    - NumPy  
    - PIL (Pillow)  
    - Pickle  

    ---
    ### 🚀 Applications
    - Online fashion platforms
    - AI shopping assistants
    - E-commerce recommendation engines
    - Smart fashion advisory systems

    ---
    ### 📌 Conclusion
    This project demonstrates the practical use of **Artificial Intelligence**
    in the fashion industry, offering smart and personalized clothing
    recommendations through a simple and interactive interface.
    """)

# =====================================================
# TAB 2 – AI RECOMMENDATION
# =====================================================
with tab2:
    st.subheader("📷 Upload Face Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=250)

        img_input = preprocess_image(image)
        prediction = skin_model.predict(img_input)
        skin_tone = int(np.argmax(prediction))

        st.success(f"Detected Skin Tone: {skin_labels[skin_tone]}")
        st.info(skin_explanations[skin_tone])

        # CATEGORY FILTER
        st.subheader("🎯 Filter by Category")

        all_categories = sorted({
            item.get("category", "Other")
            for items in clothes_mapping.values()
            for item in items
        })

        selected_category = st.selectbox(
            "Select Category",
            ["All"] + all_categories
        )

        st.subheader("👗 Recommended Clothes")

        filtered_items = [
            item for item in clothes_mapping.get(skin_tone, [])
            if selected_category == "All"
            or item.get("category", "Other") == selected_category
        ]

        random.shuffle(filtered_items)
        filtered_items = filtered_items[:4]

        cols = st.columns(1)
        for i, item in enumerate(filtered_items):
            with cols[0]:
                img_name = item.get("img")
                img_path = os.path.join(IMAGES_DIR, img_name) if img_name else None

                if img_path and os.path.exists(img_path):
                    st.image(
                        img_path,
                        caption=item.get("name", "Clothing"),
                        use_container_width=True
                    )
                else:
                    st.warning(item.get("name", "Image not available"))

# =====================================================
# TAB 3 – ALL WOMEN CLOTHES
# =====================================================
with tab3:
    st.subheader("👗 Complete Women Clothing Collection")

    for tone, items in clothes_mapping.items():
        st.markdown(f"### {skin_labels.get(tone, 'Skin Tone')}")

        cols = st.columns(4)
        for i, item in enumerate(items):
            with cols[i % 4]:
                img_name = item.get("img")
                img_path = os.path.join(IMAGES_DIR, img_name) if img_name else None

                if img_path and os.path.exists(img_path):
                    st.image(
                        img_path,
                        caption=item.get("name", "Clothing"),
                        use_container_width=True
                    )
                else:
                    st.warning(item.get("name", "Image not available"))


