import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
import io

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_nanas_mobilenet_final.keras")

model = load_model()

# Kelas target
class_names = ['healthy', 'root rot', 'fruit rot', 'mealybug wilt', 'leaf blight']
CONFIDENCE_THRESHOLD = 0.7

st.title("🍍 Deteksi Penyakit Daun Nanas")
st.write("Upload gambar daun nanas, dan sistem akan memprediksi jenis penyakit atau memberi peringatan jika tidak dikenali.")

uploaded_file = st.file_uploader("Upload gambar daun nanas...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    st.image(img, caption='Gambar yang diupload', use_column_width=True)

    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_idx = np.argmax(prediction)

    if confidence >= CONFIDENCE_THRESHOLD:
        st.success(f"✅ Prediksi: *{class_names[predicted_idx]}* ({confidence*100:.2f}% yakin)")
    else:
        st.error("❌ Maaf, gambar tidak dikenali dengan cukup yakin.")
        st.info("Silakan upload ulang gambar yang lebih jelas atau dengan sudut pandang berbeda.")
