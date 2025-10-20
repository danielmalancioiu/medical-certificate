from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(["en"], gpu=True)

# Normalised portrait size used after perspective correction
TARGET_WIDTH = 1400
TARGET_HEIGHT = 1980

@dataclass(frozen=True)
class RoiSpec:
    name: str
    top: float
    left: float
    bottom: float
    right: float
    kind: str = "text"  # text | digits | date | checkbox
    description: Optional[str] = None


# ROI map approximated on an aligned reference certificate.
ROI_MAP_PATH = Path(__file__).with_name("roi_map.json")


def _load_roi_specs(path: Path = ROI_MAP_PATH) -> List[RoiSpec]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            raw_map = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ROI map file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in ROI map file: {path}") from exc

    specs: List[RoiSpec] = []
    for name, payload in raw_map.items():
        if not isinstance(payload, dict):
            kind_name = type(payload).__name__
            raise ValueError(f"ROI '{name}' must be an object, got {kind_name}.")

        coords = payload.get("roi")
        if not (isinstance(coords, list) and len(coords) == 4):
            raise ValueError(f"ROI '{name}' must define 'roi' as a list of four numbers.")

        top, left, bottom, right = coords
        specs.append(
            RoiSpec(
                name=name,
                top=float(top),
                left=float(left),
                bottom=float(bottom),
                right=float(right),
                kind=payload.get("kind", "text"),
                description=payload.get("description"),
            )
        )

    return specs


try:
    ROI_SPECS: List[RoiSpec] = _load_roi_specs()
except Exception as exc:
    raise RuntimeError(f"Failed to load ROI specifications from '{ROI_MAP_PATH}': {exc}") from exc

def load_image(uploaded_file) -> np.ndarray:
    """Load bytes/PIL image into OpenCV BGR array."""
    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0)
        data = uploaded_file.read()
        image = Image.open(io.BytesIO(data))
    else:
        image = uploaded_file
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points as (tl, tr, br, bl)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def align_certificate(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Attempt to warp the document to a canonical portrait frame."""
    try:
        ratio = 1000.0 / max(image.shape[:2])
        resized = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32") / ratio
                ordered = order_points(pts)
                target = np.array(
                    [
                        [0, 0],
                        [TARGET_WIDTH - 1, 0],
                        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
                        [0, TARGET_HEIGHT - 1],
                    ],
                    dtype="float32",
                )
                matrix = cv2.getPerspectiveTransform(ordered, target)
                warped = cv2.warpPerspective(image, matrix, (TARGET_WIDTH, TARGET_HEIGHT))
                return warped, True
    except Exception:
        pass

    fallback = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return fallback, False


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Enhance contrast and smooth noise."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    return gray


def roi_to_pixels(roi: RoiSpec, height: int, width: int) -> Tuple[int, int, int, int]:
    y1 = max(0, int(roi.top * height))
    y2 = min(height, int(roi.bottom * height))
    x1 = max(0, int(roi.left * width))
    x2 = min(width, int(roi.right * width))
    return y1, y2, x1, x2


def remove_grid_lines(image: np.ndarray) -> np.ndarray:
    """Suppress long ruling lines without erasing handwritten strokes."""
    if image.size == 0:
        return image

    if image.ndim == 3:
        work = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        work = image.copy()

    height, width = work.shape[:2]
    vertical_kernel_len = min(height, max(3, int(round(height * 0.9))))
    horizontal_kernel_len = min(width, max(3, int(round(width * 0.9))))

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))

    vertical_lines = cv2.morphologyEx(work, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal_lines = cv2.morphologyEx(work, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
    cleaned = cv2.subtract(work, lines)
    cleaned = cv2.medianBlur(cleaned, 3)
    _, cleaned = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cleaned


def upscale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def emphasize_blue_ink(image: np.ndarray) -> np.ndarray:
    """Highlight bluish strokes (common for ballpoint pen digits)."""
    if image.size == 0:
        return image

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 40, 50], dtype=np.uint8)
    upper_blue = np.array([150, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    b_channel, g_channel, r_channel = cv2.split(bgr)
    max_gr = cv2.max(g_channel, r_channel)
    dominance = cv2.subtract(b_channel, max_gr)
    dominance = cv2.normalize(dominance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    combined = cv2.max(mask, dominance)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
    return enhanced


def run_easyocr(image: np.ndarray, allowlist: Optional[str]=None) -> str:
    if image.size == 0:
        return ""
    rgb = cv2.cvtColor(image if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    lines = reader.readtext(rgb, detail=0, paragraph=True, allowlist=allowlist)
    return " ".join(lines).strip()


def read_digits_with_confidence(
    image: np.ndarray,
    allowlist: str = "0123456789",
    min_confidence: float = 0.5,
) -> Tuple[str, float]:
    """Return digits and EasyOCR confidence for the strongest detection."""
    if image.size == 0:
        return "", 0.0

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = reader.readtext(rgb, detail=1, paragraph=False, allowlist=allowlist)

    high_conf_digits: List[str] = []
    fallback_digits: List[str] = []
    best_confidence = 0.0
    for _bbox, text, confidence in results:
        if not text:
            continue
        sanitized = re.sub(r"[^0-9]", "", text)
        if not sanitized:
            continue
        best_confidence = max(best_confidence, float(confidence))
        fallback_digits.append(sanitized)
        if confidence >= min_confidence:
            high_conf_digits.append(sanitized)

    if high_conf_digits:
        return "".join(high_conf_digits), best_confidence
    if fallback_digits and best_confidence >= max(0.01, min_confidence * 0.8):
        return "".join(fallback_digits), best_confidence
    return "", best_confidence


def extract_digits(image: np.ndarray) -> str:
    if image.size == 0:
        return ""
    if len(image.shape) == 3:
        color = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    blue_enhanced = emphasize_blue_ink(color)
    blue_blur = cv2.GaussianBlur(blue_enhanced, (3, 3), 0)
    blue_binary = cv2.adaptiveThreshold(
        blue_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        9,
    )
    binary_clean = remove_grid_lines(blue_binary)

    ink_ratio = float(cv2.countNonZero(blue_binary)) / float(blue_binary.size)
    peak_intensity = cv2.minMaxLoc(blue_enhanced)[1] if blue_enhanced.size else 0
    if ink_ratio < 0.0008 and peak_intensity < 80:
        return ""

    digits_allowlist = "0123456789"
    candidates = [
        {"image": gray, "scale": 2.2},
        {"image": gray_clahe, "scale": 2.4},
        {"image": color, "scale": 2.0},
        {"image": blue_enhanced, "scale": 2.8},
        {"image": blue_binary, "scale": 3.0},
        {"image": cv2.bitwise_not(blue_binary), "scale": 3.0},
        {"image": binary_clean, "scale": 3.0},
        {"image": cv2.bitwise_not(binary_clean), "scale": 3.0},
    ]

    best_digits = ""
    best_confidence = 0.0

    for candidate in candidates:
        candidate_image = candidate["image"]
        factor = candidate.get("scale", 2.0)
        scaled = upscale(candidate_image, factor=factor)
        digits, confidence = read_digits_with_confidence(
            scaled,
            allowlist=digits_allowlist,
            min_confidence=0.55,
        )
        if digits and confidence >= 0.58:
            return digits
        if digits and confidence > best_confidence:
            best_digits = digits
            best_confidence = confidence

    if best_confidence >= 0.48 and best_digits:
        return best_digits

    for candidate in candidates:
        candidate_image = candidate["image"]
        factor = candidate.get("scale", 2.0)
        scaled = upscale(candidate_image, factor=factor)
        digits, confidence = read_digits_with_confidence(
            scaled,
            allowlist=digits_allowlist,
            min_confidence=0.4,
        )
        if digits and confidence >= 0.5:
            return digits
        if digits and confidence > best_confidence:
            best_digits = digits
            best_confidence = confidence

    if best_confidence >= 0.42 and best_digits:
        return best_digits

    if ink_ratio >= 0.0015:
        fallback_text = run_easyocr(upscale(cv2.bitwise_not(binary_clean), factor=3.0), allowlist=digits_allowlist)
        fallback_digits = re.sub(r"[^0-9]", "", fallback_text)
        if fallback_digits:
            return fallback_digits

    return best_digits if best_confidence > 0 else ""


def parse_date(raw: str) -> str:
    digits = re.sub(r"[^0-9]", "", raw)
    if len(digits) >= 8:
        dd, mm, yyyy = digits[:2], digits[2:4], digits[4:8]
        return f"{dd}.{mm}.{yyyy}"
    if len(digits) == 6:
        dd, mm, yy = digits[:2], digits[2:4], digits[4:6]
        prefix = "20" if int(yy) < 50 else "19"
        return f"{dd}.{mm}.{prefix}{yy}"
    return raw.strip()


def detect_checkbox(image: np.ndarray) -> str:
    if image.size == 0:
        return "false"
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fill_ratio = binary.sum() / (255 * binary.size)
    return "true" if fill_ratio > 0.10 else "false"


def extract_roi_value(roi: RoiSpec, color_img: np.ndarray, gray_img: np.ndarray) -> str:
    height, width = gray_img.shape[:2]
    y1, y2, x1, x2 = roi_to_pixels(roi, height, width)
    crop_color = color_img[y1:y2, x1:x2]
    crop_gray = gray_img[y1:y2, x1:x2]

    if roi.kind == "digits":
        value = extract_digits(crop_color)
    elif roi.kind == "date":
        value = parse_date(extract_digits(crop_color))
    elif roi.kind == "checkbox":
        value = detect_checkbox(crop_color)
    else:
        value = run_easyocr(crop_gray)
    return value.strip()

def extract_fields(uploaded_file, preview: bool = False):
    """Return structured values and optional ROI preview overlay."""
    original = load_image(uploaded_file)
    aligned, warped = align_certificate(original)
    if not warped:
        aligned = cv2.resize(original, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

    gray = preprocess_for_ocr(aligned)

    preview_img = aligned.copy()

    extracted = {}
    for roi in ROI_SPECS:
        extracted[roi.name] = extract_roi_value(roi, aligned, gray)

    if preview:
        height, width = gray.shape[:2]
        for roi in ROI_SPECS:
            y1, y2, x1, x2 = roi_to_pixels(roi, height, width)
            cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                preview_img,
                roi.name,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
        return preview_img, extracted

    return extracted

