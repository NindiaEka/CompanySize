import streamlit as st
import eda
import predict

# Set up the page configuration
st.set_page_config(page_title="Company Classification App", layout="wide")

# Sidebar for navigation
with st.sidebar:
    st.write("# Page Navigation")

    # Toggle for selecting the page
    page = st.selectbox("Pilih Halaman", ("EDA", "Prediksi"))

    # Display selected page
    st.write(f"Halaman yang dituju: **{page}**")

    st.write("## About")
    st.markdown('''
    Aplikasi ini melakukan analisis deskriptif dan klasifikasi ukuran perusahaan berdasarkan deskripsi LinkedIn menggunakan pendekatan NLP dan deep learning.
    ''')

# Main content based on the selected page
if page == 'EDA':
    eda.show_eda_page()  # ganti dengan run_eda() jika kamu ubah fungsinya
else:
    predict.show_prediction_page()  # ganti dengan run_prediction() jika kamu ubah fungsinya
