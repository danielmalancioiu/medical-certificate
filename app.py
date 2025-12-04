import hashlib
import io
import mimetypes
from datetime import datetime
from pathlib import Path

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
        "debug_label": "Show ROI debug layers",
        "debug_section": "ROI debug layers",
        "debug_empty": "No debug layers captured.",
        "values_tab": "Values",
        "review_tab": "Confidences & Review",
        "review_header": "Confidences & Review",
        "review_empty": "No review info available.",
        "picker_label": "Or pick a sample from /certificate",
        "picker_placeholder": "No sample selected",
        "picker_error": "Selected sample could not be read.",
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
        "debug_label": "Afiseaza etapele ROI (debug)",
        "debug_section": "Etape ROI debug",
        "debug_empty": "Nu exista etape capturate pentru debug.",
        "values_tab": "Valori",
        "review_tab": "Confidenta si revizuire",
        "review_header": "Confidente si verificare",
        "review_empty": "Nu exista informatii de verificare.",
        "picker_label": "Sau alege un fisier din /certificate",
        "picker_placeholder": "Niciun fisier selectat",
        "picker_error": "Fisierul selectat nu a putut fi citit.",
    },
}

MAX_HISTORY_ENTRIES = 15
CERTIFICATE_DIR = Path(__file__).parent / "certificate"

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
available_samples = (
    sorted(
        [
            path
            for path in CERTIFICATE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=lambda p: p.name,
    )
    if CERTIFICATE_DIR.exists()
    else []
)
sample_lookup = {path.name: path for path in available_samples}
selected_sample_name = st.selectbox(
    texts["picker_label"],
    options=[""] + list(sample_lookup.keys()),
    format_func=lambda name: texts["picker_placeholder"] if name == "" else name,
    key="certificate_sample_select",
)
selected_sample_path = sample_lookup.get(selected_sample_name) if selected_sample_name else None

debug_mode = st.sidebar.checkbox(texts["debug_label"], value=False)

file_bytes = None
input_name = None
input_mime = None
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    input_name = uploaded_file.name or "certificate.png"
    input_mime = uploaded_file.type or "application/octet-stream"
elif selected_sample_path:
    try:
        file_bytes = selected_sample_path.read_bytes()
        input_name = selected_sample_path.name
        guessed_mime, _ = mimetypes.guess_type(selected_sample_path.name)
        input_mime = guessed_mime or "image/png"
    except Exception as exc:
        st.error(f"{texts['picker_error']} ({exc})")

if file_bytes:
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    progress_status = st.empty()
    progress_bar_container = st.empty()
    progress_bar = progress_bar_container.progress(0)
    progress_status.text(texts["processing"])

    def report_progress(fraction: float, message: str) -> None:
        progress_value = max(0, min(100, int(round(fraction * 100))))
        progress_bar.progress(progress_value)
        progress_status.text(message)

    try:
        preview_image, values, confidences, review_info, debug_layers = extract_fields(
            io.BytesIO(file_bytes),
            preview=True,
            progress_callback=report_progress,
            debug=debug_mode,
        )
    except Exception as exc:
        progress_bar_container.empty()
        progress_status.empty()
        st.error(
            "OCR failed to run. "
            "If you are offline, download PaddleOCR model in advance or allow temporary network access."
            f"Details: {exc}"
        )
        st.stop()

    sorted_names = sorted(values.keys())
    values = {name: values[name] for name in sorted_names}
    confidences = {name: confidences.get(name, 0.0) for name in sorted_names}
    review_info = {name: review_info.get(name, {}) for name in sorted_names}

    progress_bar_container.empty()
    progress_status.empty()

    results_df = pd.DataFrame([values])
    review_rows = [
        {
            "field": name,
            "value": values.get(name, ""),
            "confidence": confidences.get(name, 0.0),
            "needs_review": review_info.get(name, {}).get("needs_review", False),
            "reason": review_info.get(name, {}).get("reason", ""),
            "kind": review_info.get(name, {}).get("kind"),
            "description": review_info.get(name, {}).get("description"),
        }
        for name in sorted_names
    ]
    review_df = pd.DataFrame(review_rows)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="OCR")
    excel_bytes = excel_buffer.getvalue()

    preview_rgb = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
    preview_col, results_col = st.columns((3, 2))
    with preview_col:
        st.image(preview_rgb, caption=texts["preview_caption"], width="stretch")
        if debug_mode:
            st.divider()
            st.subheader(texts["debug_section"])
            if debug_layers:
                for roi_name, layers in debug_layers.items():
                    with st.expander(roi_name, expanded=False):
                        for layer in layers:
                            caption = layer.get("label", "")
                            if layer.get("note"):
                                caption = f"{caption} ({layer['note']})" if caption else layer["note"]
                            conf = layer.get("confidence")
                            if conf is not None:
                                if caption:
                                    caption = f"{caption} [conf={conf:.12f}]"
                                else:
                                    caption = f"conf={conf:.12f}"
                            st.image(layer.get("image"), caption=caption, width='stretch')
            else:
                st.info(texts["debug_empty"])
    with results_col:
        st.success(texts["success"])
        tab_values, tab_review = st.tabs([texts["values_tab"], texts["review_tab"]])
        with tab_values:
            st.subheader(texts["results_header"])
            st.json(values)
            st.download_button(
                label=texts["download_label"],
                data=excel_bytes,
                file_name="ocr_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_current_excel",
            )
            st.download_button(
                label="Download results as JSON",
                data=pd.Series(values).to_json(orient="index", indent=2),
                file_name="ocr_results.json",
                mime="application/json",
                key="download_current_json",
            )
        with tab_review:
            st.subheader(texts["review_header"])
            if review_df.empty:
                st.info(texts["review_empty"])
            else:
                highlight_style = review_df.style.apply(
                    lambda row: ["background-color: #800020" if row["needs_review"] else "" for _ in row],
                    axis=1,
                )
                st.dataframe(highlight_style, width="stretch")

    st.dataframe(results_df, width="stretch")

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
        "name": input_name or "certificate.png",
        "timestamp": timestamp,
        "results": dict(values),
        "values": dict(values),
        "confidences": dict(confidences),
        "review_info": dict(review_info),
        "preview_png": preview_bytes,
        "excel_bytes": excel_bytes,
        "original_bytes": file_bytes,
        "mime": input_mime or "application/octet-stream",
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
        with st.expander(entry_title, expanded=(idx == 0 and file_bytes is not None)):
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
                entry_values = entry.get("values") or entry.get("results", {})
                entry_confidences = entry.get("confidences", {})
                entry_review_info = entry.get("review_info", {})
                tab_hist_values, tab_hist_review = st.tabs([texts["values_tab"], texts["review_tab"]])
                with tab_hist_values:
                    st.json(entry_values)
                with tab_hist_review:
                    review_rows_history = [
                        {
                            "field": name,
                            "value": entry_values.get(name, ""),
                            "confidence": entry_confidences.get(name, 0.0),
                            "needs_review": entry_review_info.get(name, {}).get("needs_review", False),
                            "reason": entry_review_info.get(name, {}).get("reason", ""),
                            "kind": entry_review_info.get(name, {}).get("kind"),
                            "description": entry_review_info.get(name, {}).get("description"),
                        }
                        for name in entry_values.keys()
                    ]
                    if review_rows_history:
                        st.dataframe(pd.DataFrame(review_rows_history), width="stretch")
                    else:
                        st.info(texts["review_empty"])

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
