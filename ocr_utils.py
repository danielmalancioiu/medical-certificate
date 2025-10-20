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


def shrink_crop(image: np.ndarray, margin_ratio: float) -> np.ndarray:
    """Trim uniform margins to reduce the impact of printed borders/guidelines."""
    if image.size == 0 or margin_ratio <= 0:
        return image

    height, width = image.shape[:2]
    margin_y = min(height // 2, max(0, int(round(height * margin_ratio))))
    margin_x = min(width // 2, max(0, int(round(width * margin_ratio))))

    top = margin_y
    bottom = height - margin_y
    left = margin_x
    right = width - margin_x

    if bottom <= top or right <= left:
        return image
    return image[top:bottom, left:right]


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


def suppress_guidelines(image: np.ndarray) -> np.ndarray:
    """Remove contiguous and dotted guide lines using inpainting."""
    if image.size == 0:
        return image

    if image.ndim == 2:
        gray = image
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return_gray = True
    else:
        color = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return_gray = False

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        7,
    )

    height, width = gray.shape[:2]
    horizontal_len = max(8, width // 6)
    vertical_len = max(8, height // 6)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len))

    dotted_h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dotted_v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, dotted_h_kernel, iterations=1)
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, dotted_v_kernel, iterations=1)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    line_mask = cv2.bitwise_or(horizontal, vertical)
    if not np.any(line_mask):
        return gray if return_gray else color

    mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(color, mask, 3, cv2.INPAINT_TELEA)
    if return_gray:
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    return inpainted


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


def emphasize_dark_ink(image: np.ndarray) -> np.ndarray:
    """Boost contrast for darker handwriting regardless of ink colour."""
    if image.size == 0:
        return image

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.medianBlur(enhanced, 3)
    return enhanced


def tidy_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def run_easyocr(image: np.ndarray, allowlist: Optional[str]=None) -> str:
    if image.size == 0:
        return ""
    rgb = cv2.cvtColor(image if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    lines = reader.readtext(rgb, detail=0, paragraph=True, allowlist=allowlist)
    return " ".join(lines).strip()


def read_text_with_confidence(
    image: np.ndarray,
    allowlist: Optional[str] = None,
    min_confidence: float = 0.35,
) -> Tuple[str, float]:
    """OCR with confidences, returning text that meets a threshold."""
    if image.size == 0:
        return "", 0.0

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = reader.readtext(rgb, detail=1, paragraph=False, allowlist=allowlist)

    accepted: List[str] = []
    fallback: List[str] = []
    best_confidence = 0.0
    for _bbox, text, confidence in results:
        if not text:
            continue
        cleaned = re.sub(r"\s+", " ", text.strip())
        if not cleaned:
            continue
        best_confidence = max(best_confidence, float(confidence))
        fallback.append(cleaned)
        if confidence >= min_confidence:
            accepted.append(cleaned)

    if accepted:
        return " ".join(accepted), best_confidence
    if fallback and best_confidence >= max(0.01, min_confidence * 0.7):
        return " ".join(fallback), best_confidence
    return "", best_confidence


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

    guide_suppressed = suppress_guidelines(color)
    if guide_suppressed.ndim == 3:
        guide_gray = cv2.cvtColor(guide_suppressed, cv2.COLOR_BGR2GRAY)
    else:
        guide_gray = guide_suppressed
        guide_suppressed = cv2.cvtColor(guide_gray, cv2.COLOR_GRAY2BGR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    guide_clahe = clahe.apply(guide_gray)

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

    guide_blue = emphasize_blue_ink(guide_suppressed)
    guide_blur = cv2.GaussianBlur(guide_gray, (3, 3), 0)
    guide_binary = cv2.adaptiveThreshold(
        guide_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7,
    )
    guide_binary_clean = remove_grid_lines(guide_binary)

    ink_ratio = float(cv2.countNonZero(blue_binary)) / float(blue_binary.size)
    peak_intensity = cv2.minMaxLoc(blue_enhanced)[1] if blue_enhanced.size else 0
    if ink_ratio < 0.0008 and peak_intensity < 80:
        guide_ratio = float(cv2.countNonZero(guide_binary)) / float(guide_binary.size)
        guide_peak = cv2.minMaxLoc(guide_blue)[1] if guide_blue.size else 0
        if guide_ratio < 0.0008 and guide_peak < 80:
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
        {"image": guide_gray, "scale": 2.4},
        {"image": guide_clahe, "scale": 2.6},
        {"image": guide_suppressed, "scale": 2.2},
        {"image": guide_blue, "scale": 3.0},
        {"image": guide_binary, "scale": 3.0},
        {"image": cv2.bitwise_not(guide_binary), "scale": 3.0},
        {"image": guide_binary_clean, "scale": 3.2},
        {"image": cv2.bitwise_not(guide_binary_clean), "scale": 3.2},
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


def extract_text(image: np.ndarray) -> str:
    if image.size == 0:
        return ""

    if image.ndim == 3:
        color = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    guide_suppressed = suppress_guidelines(color)
    if guide_suppressed.ndim == 3:
        guide_gray = cv2.cvtColor(guide_suppressed, cv2.COLOR_BGR2GRAY)
    else:
        guide_gray = guide_suppressed
        guide_suppressed = cv2.cvtColor(guide_gray, cv2.COLOR_GRAY2BGR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    guide_clahe = clahe.apply(guide_gray)
    dark_enhanced = emphasize_dark_ink(color)
    blue_enhanced = emphasize_blue_ink(color)
    guide_dark = emphasize_dark_ink(guide_suppressed)
    guide_blue = emphasize_blue_ink(guide_suppressed)

    blurred = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        9,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned_grid = remove_grid_lines(adaptive)

    guide_blur = cv2.GaussianBlur(guide_clahe, (3, 3), 0)
    guide_adaptive = cv2.adaptiveThreshold(
        guide_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    guide_closed = cv2.morphologyEx(guide_adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    guide_clean = remove_grid_lines(guide_adaptive)

    text_allowlist = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz-_/.,"
    )

    candidates = [
        {"image": gray, "scale": 2.0, "min_conf": 0.38, "allowlist": None},
        {"image": gray_clahe, "scale": 2.2, "min_conf": 0.4, "allowlist": None},
        {"image": dark_enhanced, "scale": 2.3, "min_conf": 0.4, "allowlist": None},
        {"image": cv2.bitwise_not(adaptive), "scale": 2.6, "min_conf": 0.45, "allowlist": text_allowlist},
        {"image": adaptive, "scale": 2.6, "min_conf": 0.45, "allowlist": text_allowlist},
        {"image": cv2.bitwise_not(closed), "scale": 2.8, "min_conf": 0.45, "allowlist": text_allowlist},
        {"image": closed, "scale": 2.8, "min_conf": 0.45, "allowlist": text_allowlist},
        {"image": cv2.bitwise_not(cleaned_grid), "scale": 2.8, "min_conf": 0.42, "allowlist": text_allowlist},
        {"image": cleaned_grid, "scale": 2.8, "min_conf": 0.42, "allowlist": text_allowlist},
        {"image": guide_gray, "scale": 2.2, "min_conf": 0.38, "allowlist": None},
        {"image": guide_clahe, "scale": 2.4, "min_conf": 0.4, "allowlist": None},
        {"image": guide_dark, "scale": 2.5, "min_conf": 0.4, "allowlist": None},
        {"image": guide_blue, "scale": 2.7, "min_conf": 0.45, "allowlist": text_allowlist},
        {"image": guide_adaptive, "scale": 2.7, "min_conf": 0.45, "allowlist": text_allowlist},
        {"image": cv2.bitwise_not(guide_adaptive), "scale": 2.7, "min_conf": 0.45, "allowlist": text_allowlist},
        {"image": guide_closed, "scale": 2.9, "min_conf": 0.46, "allowlist": text_allowlist},
        {"image": cv2.bitwise_not(guide_closed), "scale": 2.9, "min_conf": 0.46, "allowlist": text_allowlist},
        {"image": guide_clean, "scale": 2.9, "min_conf": 0.44, "allowlist": text_allowlist},
        {"image": cv2.bitwise_not(guide_clean), "scale": 2.9, "min_conf": 0.44, "allowlist": text_allowlist},
    ]

    if np.count_nonzero(blue_enhanced) > 0:
        candidates.append(
            {"image": blue_enhanced, "scale": 2.7, "min_conf": 0.46, "allowlist": text_allowlist}
        )
        candidates.append(
            {"image": cv2.bitwise_not(blue_enhanced), "scale": 2.7, "min_conf": 0.46, "allowlist": text_allowlist}
        )

    best_text = ""
    best_confidence = 0.0

    for candidate in candidates:
        candidate_image = candidate["image"]
        factor = candidate.get("scale", 2.0)
        min_conf = candidate.get("min_conf", 0.35)
        allowlist = candidate.get("allowlist")
        scaled = upscale(candidate_image, factor=factor)
        text, confidence = read_text_with_confidence(
            scaled,
            allowlist=allowlist,
            min_confidence=min_conf,
        )
        cleaned_text = tidy_text(text)
        if cleaned_text and confidence >= (min_conf + 0.05):
            return cleaned_text
        if cleaned_text:
            better_conf = confidence > best_confidence + 0.02
            similar_conf = abs(confidence - best_confidence) <= 0.02 and len(cleaned_text) > len(best_text)
            if better_conf or similar_conf:
                best_text = cleaned_text
                best_confidence = confidence

    if best_text:
        return best_text

    fallback = tidy_text(run_easyocr(upscale(gray, factor=2.4)))
    return fallback


def format_date_from_digits(digits: str) -> Optional[str]:
    if not digits:
        return None

    def valid(day: int, month: int, year: int) -> bool:
        return 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100

    if len(digits) >= 6:
        day = int(digits[:2])
        month = int(digits[2:4])
        year = int(digits[4:6])
        year += 2000 if year < 50 else 1900
        if valid(day, month, year):
            return f"{day:02d}.{month:02d}.{year:04d}"
    return None


def extract_date(image: np.ndarray) -> str:
    digits = extract_digits(image)
    formatted = format_date_from_digits(digits)
    if formatted:
        return formatted
    return digits[:6]


def parse_date(raw: str) -> str:
    digits = re.sub(r"[^0-9]", "", raw)
    formatted = format_date_from_digits(digits)
    if formatted:
        return formatted
    return digits[:6].strip()


def detect_checkbox(image: np.ndarray) -> str:
    if image.size == 0:
        return "false"
    if image.ndim == 3:
        color = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    inner = shrink_crop(color, margin_ratio=0.12)
    if inner.size == 0:
        inner = color

    cleaned = suppress_guidelines(inner)
    if cleaned.ndim == 3:
        cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    else:
        cleaned_gray = cleaned

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(cleaned_gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    height, width = strokes.shape[:2]
    total_area = float(height * width)
    if total_area == 0:
        return "false"

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    significant_area = 0.0
    elongated_component = False
    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        significant_area += area
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        if w >= 0.35 * width or h >= 0.35 * height:
            elongated_component = True

    ink_ratio = significant_area / total_area
    if ink_ratio >= 0.012 or (ink_ratio >= 0.006 and elongated_component):
        return "true"
    return "false"


def extract_roi_value(roi: RoiSpec, color_img: np.ndarray, gray_img: np.ndarray) -> str:
    height, width = gray_img.shape[:2]
    y1, y2, x1, x2 = roi_to_pixels(roi, height, width)
    crop_color = color_img[y1:y2, x1:x2]
    margin_map = {
        "digits": 0.06,
        "date": 0.06,
        "checkbox": 0.08,
        "text": 0.05,
    }
    margin = margin_map.get(roi.kind, 0.05)
    crop_color = shrink_crop(crop_color, margin_ratio=margin)

    if roi.kind == "digits":
        value = extract_digits(crop_color)
    elif roi.kind == "date":
        value = extract_date(crop_color)
    elif roi.kind == "checkbox":
        value = detect_checkbox(crop_color)
    else:
        value = extract_text(crop_color)
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

