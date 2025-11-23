import io
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import easyocr


def load_image(uploaded) -> np.ndarray:
    """Load a Streamlit-uploaded file or PIL image into a BGR ndarray."""
    if hasattr(uploaded, "read"):
        data = uploaded.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        pil = uploaded
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_boxes(image: np.ndarray, detections: List[Tuple]) -> np.ndarray:
    """Overlay EasyOCR bounding boxes on top of the image."""
    overlay = image.copy()
    for idx, det in enumerate(detections):
        bbox, text, conf = det
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 180, 0), thickness=2)
        cv2.putText(
            overlay,
            f"{idx+1}:{conf:.2f}",
            (int(pts[0][0]), int(pts[0][1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def main():
    st.set_page_config(page_title="EasyOCR Quick Tester", layout="wide")
    st.title("EasyOCR Quick Tester")
    st.write("Upload or paste an image, then run EasyOCR to inspect detected text.")

    lang = st.sidebar.multiselect(
        "Languages",
        options=["en", "ro", "fr", "de", "it", "es"],
        default=["en"],
        help="EasyOCR language codes",
    )
    gpu = st.sidebar.checkbox("Use GPU (if available)", value=True)
    reader_key = f"reader_{'-'.join(sorted(lang))}_{gpu}"

    if reader_key not in st.session_state:
        st.session_state[reader_key] = easyocr.Reader(lang, gpu=gpu)
    reader = st.session_state[reader_key]

    uploaded = st.file_uploader("Drop an image (PNG/JPG) or paste one", type=["png", "jpg", "jpeg"])
    if not uploaded:
        st.info("Waiting for an image…")
        return

    image = load_image(uploaded)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("Running EasyOCR…"):
        detections = reader.readtext(rgb, detail=1, paragraph=False, allowlist="0123456789")

    texts = [{"text": text, "confidence": float(conf)} for _bbox, text, conf in detections]
    st.subheader("Detected Text")
    if texts:
        st.json(texts)
    else:
        st.warning("No text detected.")

    overlay = draw_boxes(rgb, detections)
    st.subheader("Preview")
    st.image(overlay, caption="OCR bounding boxes", width='stretch')


if __name__ == "__main__":
    main()
