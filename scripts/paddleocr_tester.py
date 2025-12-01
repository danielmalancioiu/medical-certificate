# Create an in-memory bytes buffer
from string import digits

import numpy as np
import streamlit as st
from paddleocr import TextRecognition
from PIL import Image
import io
import json
import os

@st.cache_resource
def load_model():
    return TextRecognition(model_name="en_PP-OCRv5_mobile_rec")

model = load_model()

st.title("PaddleOCR TextRecognition Demo")
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if st.button("Run OCR"):
    with st.spinner("Running OCR..."):
        # PaddleOCR TextRecognition can accept a path or numpy array; here we use bytes->PIL->np
        # Convert to numpy array
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        # Run prediction
        output = model.predict(input=img_np)

    st.success("OCR complete!")

    # Prepare output directory
    os.makedirs("output", exist_ok=True)

    # Collect all results for JSON download
    all_results = []
    for idx, res in enumerate(output):
        # Print to app
        st.subheader(f"Result {idx + 1}")
        st.text(res)

        # Save result image and JSON like in your original script
        img_save_path = os.path.join("output", f"res_{idx + 1}.png")
        json_save_path = os.path.join("output", f"res_{idx + 1}.json")

        res.save_to_img(save_path=img_save_path)
        res.save_to_json(save_path=json_save_path)

        # For in-app preview of result image
        try:
            res_img = Image.open(img_save_path)
            st.image(res_img, caption=f"OCR result {idx + 1}", use_column_width=True)
        except Exception:
            pass

        # Add to combined JSON result
        all_results.append(res.to_dict() if hasattr(res, "to_dict") else str(res))

    # Button to download combined JSON
    json_bytes = io.BytesIO()
    json_bytes.write(json.dumps(all_results, indent=2).encode("utf-8"))
    json_bytes.seek(0)

    st.download_button(
        label="Download OCR JSON results",
        data=json_bytes,
        file_name="ocr_results.json",
        mime="application/json",
    )
