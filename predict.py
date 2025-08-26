import tensorflow as tf
import json
from keras.layers import TFSMLayer
import streamlit as st  # pastikan streamlit di-import

# Load vectorizer & model (format SavedModel)
vectorizer = TFSMLayer("models/vectorizer_savedmodel", call_endpoint="serving_default")
model = TFSMLayer("models/bilstm_attention_savedmodel", call_endpoint="serving_default")

# Load label mapping
with open("models/label_map.json") as f:
    inv_label_map = json.load(f)

# Buat ulang embedding layer (HARUS SAMA dengan saat training)
embedding_layer = tf.keras.layers.Embedding(
    input_dim=21691,     # jumlah total token unik (harus disamakan)
    output_dim=128,      # dimensi embedding saat training
    mask_zero=True
)

def predict_label(text):
    # Step 1: Format input
    text_tensor = tf.constant([text])
    text_tensor = tf.reshape(text_tensor, (-1, 1))  # shape: (1, 1)

    # Step 2: Vectorize → Embed
    X = vectorizer(text_tensor)["text_vectorization"]
    X_embed = embedding_layer(X)

    # Step 3: Predict
    y_probs = model(X_embed)["output_0"]
    y_pred = tf.argmax(y_probs, axis=1).numpy()

    # Step 4: Convert to label
    label = inv_label_map[str(y_pred[0])]
    return label

def get_vision_mission(company_type):
    if company_type == 'small':
        return {
            "vision": "To become a flexible and personalized service provider for niche and local client needs.",
            "mission": [
                "Deliver innovative solutions with a compact and agile team.",
                "Build strong, long-term relationships with clients.",
                "Focus on specialized services tailored to unique market demands."
            ]
        }
    elif company_type == 'medium':
        return {
            "vision": "To balance customer satisfaction with efficient internal systems and scalable operations.",
            "mission": [
                "Develop high-quality services and maintain operational excellence.",
                "Provide structured workflows while remaining adaptable.",
                "Grow steadily while fostering innovation and reliability."
            ]
        }
    elif company_type == 'large':
        return {
            "vision": "To be a global leader in innovation, technology, and social impact.",
            "mission": [
                "Deliver world-class solutions at scale with cutting-edge technology.",
                "Invest in research and development for sustainable growth.",
                "Embrace social responsibility and drive positive global change."
            ]
        }
    else:
        return {
            "vision": "Not available.",
            "mission": ["Not available."]
        }

def show_prediction_page():
    st.title("Prediksi Ukuran Perusahaan")
    st.write("Masukkan deskripsi perusahaan, dan sistem akan memprediksi apakah perusahaan tersebut termasuk kecil, menengah, atau besar.")

    user_input = st.text_area("Deskripsi Perusahaan", height=150)

    if st.button("Prediksi"):
        if user_input.strip():
            label = predict_label(user_input)
            st.success(f"Predicted company size: **{label.upper()}**")

            # Show Vision & Mission
            vm = get_vision_mission(label)
            st.markdown("### Visi & Misi Perusahaan")
            st.markdown(f"**Vision:** {vm['vision']}")
            st.markdown("**Misi:**")
            for point in vm['mission']:
                st.markdown(f"- {point}")
        else:
            st.warning("Silahkan masukkan deskripsi perusahaan.")
