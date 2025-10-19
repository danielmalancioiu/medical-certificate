import streamlit as st
from ocr_utils import extract_fields
import cv2
from PIL import Image

st.set_page_config(page_title="Medical Certificate OCR", layout="wide")

st.title("📄 CNAS Medical Certificate OCR")
st.write("Upload a scanned CNAS medical certificate (JPG/PNG) to extract structured fields.")

uploaded_file = st.file_uploader("Drag & drop or browse", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("Processing... ⏳"):
        img_preview, _ = extract_fields(uploaded_file, preview=True)

        st.image(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB),
                 caption="Preview ROIs (OCR zones)", use_container_width=True)

        uploaded_file.seek(0)
        extracted = extract_fields(uploaded_file)

        st.success("✅ Extraction complete!")
        st.json(extracted)
