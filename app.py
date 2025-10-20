import hashlib
import io
from datetime import datetime

import cv2
import pandas as pd
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
        "download_label": "Download results as Excel",
        "history_header": "History",
        "history_empty": "No certificates processed yet.",
        "history_download_original": "Download original image",
        "history_download_excel": "Download results (Excel)",
        "history_preview_caption": "Preview",
        "history_clear": "Clear history",
        "history_cleared": "History cleared.",
        "history_preview_unavailable": "Preview unavailable.",
    },
    "Romana": {
        "title": "CNAS Medical-Certificates DDE",
        "description": "Incarca un certificat medical CNAS scanat (JPG/PNG) pentru a extrage campuri structurate.",
        "uploader_label": "Trage si plaseaza sau selecteaza un fisier",
        "processing": "Procesez certificatul...",
        "preview_caption": "Zone OCR identificate",
        "success": "Extractie finalizata.",
        "results_header": "Rezultate OCR",
        "download_label": "Descarca rezultatele in Excel",
        "history_header": "Istoric",
        "history_empty": "Nu exista certificate procesate inca.",
        "history_download_original": "Descarca imaginea originala",
        "history_download_excel": "Descarca rezultatele (Excel)",
        "history_preview_caption": "Previzualizare",
        "history_clear": "Sterge istoricul",
        "history_cleared": "Istoricul a fost sters.",
        "history_preview_unavailable": "Previzualizarea nu este disponibila.",
    },
}

MAX_HISTORY_ENTRIES = 15

language = st.sidebar.radio("Language / Limba", list(LANGUAGE_OPTIONS.keys()), index=0)
texts = LANGUAGE_OPTIONS[language]

if "history" not in st.session_state:
    st.session_state["history"] = []

if st.sidebar.button(texts["history_clear"]):
    st.session_state["history"] = []
    st.sidebar.success(texts["history_cleared"])

st.title(texts["title"])
st.write(texts["description"])

uploaded_file = st.file_uploader(texts["uploader_label"], type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with st.spinner(texts["processing"]):
        preview_image, extracted = extract_fields(io.BytesIO(file_bytes), preview=True)

    results_df = pd.DataFrame([extracted])

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="OCR")
    excel_bytes = excel_buffer.getvalue()

    preview_rgb = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
    preview_col, results_col = st.columns((3, 2))
    with preview_col:
        st.image(preview_rgb, caption=texts["preview_caption"], width="stretch")
    with results_col:
        st.success(texts["success"])
        st.subheader(texts["results_header"])
        st.json(extracted)
        st.dataframe(results_df, width="stretch")
        st.download_button(
            label=texts["download_label"],
            data=excel_bytes,
            file_name="ocr_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_current_excel",
        )

    history = st.session_state["history"]
    existing_entry = history[0] if history and history[0].get("file_hash") == file_hash else None

    success, preview_encoded = cv2.imencode(".png", preview_image)
    preview_bytes = preview_encoded.tobytes() if success else None

    timestamp = (
        existing_entry["timestamp"]
        if existing_entry and existing_entry.get("timestamp")
        else datetime.now().isoformat(timespec="seconds")
    )

    history_entry = {
        "file_hash": file_hash,
        "name": uploaded_file.name or "certificate.png",
        "timestamp": timestamp,
        "results": dict(extracted),
        "preview_png": preview_bytes,
        "excel_bytes": excel_bytes,
        "original_bytes": file_bytes,
        "mime": uploaded_file.type or "application/octet-stream",
    }

    if existing_entry:
        history[0] = history_entry
    else:
        history.insert(0, history_entry)
    del history[MAX_HISTORY_ENTRIES:]

st.divider()
st.subheader(texts["history_header"])

history = st.session_state["history"]
if not history:
    st.info(texts["history_empty"])
else:
    for idx, entry in enumerate(history):
        entry_title = f"{entry['timestamp']} - {entry['name']}"
        with st.expander(entry_title, expanded=(idx == 0 and uploaded_file is not None)):
            cols = st.columns((3, 2))
            with cols[0]:
                if entry.get("preview_png"):
                    st.image(
                        entry["preview_png"],
                        caption=texts["history_preview_caption"],
                        width="stretch",
                    )
                else:
                    st.caption(texts["history_preview_caption"])
                    st.info(texts["history_preview_unavailable"])
            with cols[1]:
                st.json(entry["results"])

            download_cols = st.columns(2)
            safe_timestamp = entry["timestamp"].replace(":", "-")
            excel_name = f"ocr_results_{safe_timestamp}.xlsx"
            with download_cols[0]:
                st.download_button(
                    label=texts["history_download_excel"],
                    data=entry["excel_bytes"],
                    file_name=excel_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"history_excel_{idx}",
                )
            with download_cols[1]:
                original_name = entry["name"] or f"certificate_{safe_timestamp}.png"
                st.download_button(
                    label=texts["history_download_original"],
                    data=entry["original_bytes"],
                    file_name=original_name,
                    mime=entry.get("mime", "application/octet-stream"),
                    key=f"history_original_{idx}",
                )
