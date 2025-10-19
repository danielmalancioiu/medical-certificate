import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import re

# Initialize EasyOCR once (supports Romanian handwriting/print)
reader = easyocr.Reader(['ro', 'en'], gpu=False)


def load_image(uploaded_file):
    """Load an uploaded Streamlit file or PIL image into OpenCV format (BGR)."""
    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0)
        image = Image.open(io.BytesIO(uploaded_file.read()))
    else:
        image = uploaded_file
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def preprocess(img):
    """Improve contrast and sharpness for OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=15)
    gray = cv2.medianBlur(gray, 3)
    return gray


def find_anchor(img, anchor_text="CERTIFICAT DE CONCEDIU MEDICAL"):
    """Locate anchor text and return its center coordinates."""
    results = reader.readtext(img, detail=1, paragraph=False)
    for (bbox, text, conf) in results:
        if anchor_text.lower() in text.lower() and conf > 0.5:
            pts = np.array(bbox).astype(int)
            x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
            x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
            cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
            return (cx, cy)
    return None


def extract_text_zone(img, y1, y2, x1, x2):
    """Extract text from a given ROI."""
    crop = img[y1:y2, x1:x2]
    result = reader.readtext(crop, detail=0, paragraph=True)
    return " ".join(result).strip()


def extract_fields(uploaded_file, preview=False):
    """Extract key fields from the medical certificate."""
    img = preprocess(load_image(uploaded_file))
    h, w = img.shape[:2]

    # Base coordinates relative to image size
    base_rois = {
        "serie_numar": (int(0.09*h), int(0.60*w), int(0.13*h), int(0.95*w)),
        "nume":        (int(0.26*h), int(0.20*w), int(0.30*h), int(0.95*w)),
        "cnp":         (int(0.31*h), int(0.20*w), int(0.35*h), int(0.95*w)),
        "de_la":       (int(0.53*h), int(0.38*w), int(0.57*h), int(0.48*w)),
        "pana_la":     (int(0.53*h), int(0.50*w), int(0.57*h), int(0.63*w)),
        "cod_diag":    (int(0.53*h), int(0.66*w), int(0.57*h), int(0.83*w))
    }

    # Find anchor
    anchor = find_anchor(img)
    y_shift = 0
    if anchor:
        expected_anchor_y = int(0.10 * h)
        y_shift = anchor[1] - expected_anchor_y
        print(f"✅ Anchor detected, shifting {y_shift}px")

    # Adjust ROIs based on anchor position
    rois = {}
    for k, (y1, x1, y2, x2) in base_rois.items():
        rois[k] = (max(0, y1 - y_shift), x1, min(h, y2 - y_shift), x2)

    # Extract text for each ROI
    data = {}
    for key, (y1, x1, y2, x2) in rois.items():
        data[key] = extract_text_zone(img, y1, y2, x1, x2)
        if preview:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, key, (x1, max(10, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if preview:
        return img, rois

    # Regex cleanup
    serie = re.search(r"CCM[A-Z]*", data["serie_numar"])
    numar = re.search(r"\b\d{5,8}\b", data["serie_numar"])
    cnp = re.search(r"\b\d{13}\b", data["cnp"])

    return {
        "serie": serie.group(0) if serie else "N/A",
        "numar": numar.group(0) if numar else "N/A",
        "nume_pacient": data.get("nume", ""),
        "cnp": cnp.group(0) if cnp else "N/A",
        "data_de_la": data.get("de_la", ""),
        "data_pana_la": data.get("pana_la", ""),
        "cod_diagnostic": data.get("cod_diag", "")
    }
