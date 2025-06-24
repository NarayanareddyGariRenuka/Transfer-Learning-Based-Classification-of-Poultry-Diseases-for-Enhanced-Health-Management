import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('poultry_disease_classification_model.keras')

# Define class labels
class_labels = ['healthy', 'coccidiosis', 'fowl_pox']

# Set page title
st.title("🐥 Poultry Disease Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload a poultry image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.markdown(f"### 🧪 Predicted Disease: **{predicted_class.capitalize()}**")
