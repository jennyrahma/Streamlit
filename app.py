import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
@st.cache_resource
def load_model_nanas():
    return load_model("model_nanas_mobilenet_final.keras")

model = load_model_nanas()

# Daftar label (urutan harus sesuai dengan model training)
class_labels = ['fruit rot', 'healthy', 'leaf blight', 'mealybug wilt', 'root rot']

st.title("🍍 Deteksi Penyakit Nanas dengan MobileNet")

st.write("Unggah gambar daun atau buah nanas, dan aplikasi akan memprediksi jenis penyakitnya.")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption='Gambar yang Diunggah', use_column_width=True)

        # Preprocessing sesuai input model
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]

        # Threshold minimal confidence (misal 60%)
        if confidence < 0.6:
            st.warning("🔍 Tidak dapat mendeteksi penyakit dengan cukup yakin. Silakan unggah ulang gambar yang lebih jelas.")
        else:
            predicted_class = class_labels[predicted_index]
            st.success(f"✅ Terdeteksi: **{predicted_class}** dengan keyakinan {confidence:.2f}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")
else:
    st.info("📤 Silakan unggah gambar terlebih dahulu.")
