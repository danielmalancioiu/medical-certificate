import cv2
import streamlit as st

from ocr_utils import extract_fields

st.set_page_config(page_title="CNAS Medical-Certificates DDE", layout="wide")

LANGUAGE_OPTIONS = {
    "English": {
        "title": "CNAS Medical-Certificates DDE",
        "description": "Upload a scanned CNAS medical certificate (JPG/PNG) to extract structured fields.",
        "uploader_label": "Drag and drop or select a file",
        "processing": "Processing certificate...",
        "preview_caption": "Detected OCR zones",
        "success": "Extraction complete.",
        "results_header": "OCR Results",
    },
    "Romana": {
        "title": "CNAS Medical-Certificates DDE",
        "description": "Incarca un certificat medical CNAS scanat (JPG/PNG) pentru a extrage campuri structurate.",
        "uploader_label": "Trage si plaseaza sau selecteaza un fisier",
        "processing": "Procesez certificatul...",
        "preview_caption": "Zone OCR identificate",
        "success": "Extractie finalizata.",
        "results_header": "Rezultate OCR",
    },
}

language = st.sidebar.radio("Language / Limba", list(LANGUAGE_OPTIONS.keys()), index=0)
texts = LANGUAGE_OPTIONS[language]

st.title(texts["title"])
st.write(texts["description"])

uploaded_file = st.file_uploader(texts["uploader_label"], type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner(texts["processing"]):
        preview_image, extracted = extract_fields(uploaded_file, preview=True)

    st.image(
        cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB),
        caption=texts["preview_caption"],
        use_container_width=True,
    )

    st.success(texts["success"])
    st.subheader(texts["results_header"])
    st.json(extracted)
