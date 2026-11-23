from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).parent / "pneumonia_best_model.keras"

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🫁 Pneumonia Detection")
st.write("Analyse pédagogique de radiographies thoraciques.")

st.warning(
    "Cet outil est expérimental et pédagogique. "
    "Il ne remplace pas l'avis ni le diagnostic d'un professionnel de santé."
)

uploaded_file = st.file_uploader(
    "Importer une radiographie thoracique",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Radiographie importée", use_container_width=True)

    image_resized = image.resize((64, 64))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Analyser l'image"):
        prediction = float(model.predict(image_array, verbose=0)[0][0])

        st.subheader("Résultat de la prédiction")

        if prediction >= 0.5:
            st.write("Classe prédite : PNEUMONIA")
        else:
            st.write("Classe prédite : NORMAL")

        st.write(f"Score du modèle : {prediction:.2%}")
