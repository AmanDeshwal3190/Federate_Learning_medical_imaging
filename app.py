import streamlit as st
from PIL import Image

st.set_page_config(page_title="Federated Learning", layout="wide")

st.title("Federated Learning for Medical Imaging")

st.write("AI-powered medical imaging analysis system.")

uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.success("Image uploaded successfully")

    st.subheader("Prediction")
    st.write("Sample Prediction: Normal Scan")

st.header("Project Features")

st.markdown("""
- Federated Learning
- Medical Image Processing
- Deep Learning Integration
- Distributed Training
""")
