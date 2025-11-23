from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    margin: Optional[float] = None
    auto_trim: Optional[float] = None
    expected_length: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowlist: Optional[str] = None
    preferred_ink: Optional[str] = None


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
        margin_value = payload.get("margin")
        auto_trim_value = payload.get("auto_trim")
        expected_length = payload.get("expected_length")
        min_length = payload.get("min_length")
        max_length = payload.get("max_length")

        specs.append(
            RoiSpec(
                name=name,
                top=float(top),
                left=float(left),
                bottom=float(bottom),
                right=float(right),
                kind=payload.get("kind", "text"),
                description=payload.get("description"),
                margin=float(margin_value) if margin_value is not None else None,
                auto_trim=float(auto_trim_value) if auto_trim_value is not None else None,
                expected_length=int(expected_length) if expected_length is not None else None,
                min_length=int(min_length) if min_length is not None else None,
                max_length=int(max_length) if max_length is not None else None,
                allowlist=payload.get("allowlist"),
                preferred_ink=payload.get("preferred_ink"),
            )
        )

    return specs


try:
    ROI_SPECS: List[RoiSpec] = _load_roi_specs()
except Exception as exc:
    raise RuntimeError(f"Failed to load ROI specifications from '{ROI_MAP_PATH}': {exc}") from exc

def _record_debug_image(
    store: Optional[Dict[str, List[dict]]],
    roi_name: Optional[str],
    label: str,
    image: Optional[np.ndarray],
    note: Optional[str] = None,
) -> None:
    """Store a copy of an intermediate image for later inspection."""
    if store is None or roi_name is None or image is None:
        return
    try:
        display = image.copy()
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
        else:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    except Exception:
        display = image
    entry = {"label": label, "image": display}
    if note:
        entry["note"] = note
    store.setdefault(roi_name, []).append(entry)

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

        image_area = float(resized.shape[0] * resized.shape[1])
        min_area_ratio = 0.45  # require the detected contour to cover most of the page

        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            contour_area = cv2.contourArea(approx)
            if contour_area <= 0 or (contour_area / image_area) < min_area_ratio:
                continue

            pts = approx.reshape(4, 2).astype("float32") / ratio
            ordered = order_points(pts)

            # Reject candidates with implausible aspect ratios to avoid warping noise
            width_top = np.linalg.norm(ordered[1] - ordered[0])
            width_bottom = np.linalg.norm(ordered[2] - ordered[3])
            height_left = np.linalg.norm(ordered[3] - ordered[0])
            height_right = np.linalg.norm(ordered[2] - ordered[1])

            avg_width = max((width_top + width_bottom) * 0.5, 1.0)
            avg_height = max((height_left + height_right) * 0.5, 1.0)
            aspect_ratio = avg_height / avg_width
            target_ratio = TARGET_HEIGHT / TARGET_WIDTH
            if aspect_ratio < target_ratio * 0.55 or aspect_ratio > target_ratio * 1.45:
                continue

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


def auto_trim_borders(
    image: np.ndarray,
    max_fraction: Optional[float] = None,
    intensity_threshold: float = 200.0,
    variance_threshold: float = 28.0,
) -> np.ndarray:
    """Remove dark printed borders heuristically while keeping handwriting intact."""
    if image.size == 0:
        return image

    if max_fraction is None or max_fraction <= 0:
        return image

    max_fraction = max(0.0, min(max_fraction, 0.45))
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape[:2]
    if height < 8 or width < 8:
        return image

    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)
    col_mean = gray.mean(axis=0)
    col_std = gray.std(axis=0)

    max_row_trim = max(1, int(round(height * max_fraction)))
    max_col_trim = max(1, int(round(width * max_fraction)))

    def _trim_from_start(mean_arr: np.ndarray, std_arr: np.ndarray, limit: int) -> int:
        trimmed = 0
        for mean_val, std_val in zip(mean_arr, std_arr):
            if trimmed >= limit:
                break
            if mean_val < intensity_threshold and std_val < variance_threshold:
                trimmed += 1
            else:
                break
        return trimmed

    def _trim_from_end(mean_arr: np.ndarray, std_arr: np.ndarray, limit: int) -> int:
        trimmed = 0
        for mean_val, std_val in zip(reversed(mean_arr), reversed(std_arr)):
            if trimmed >= limit:
                break
            if mean_val < intensity_threshold and std_val < variance_threshold:
                trimmed += 1
            else:
                break
        return trimmed

    top_trim = _trim_from_start(row_mean, row_std, max_row_trim)
    bottom_trim = _trim_from_end(row_mean, row_std, max_row_trim)
    left_trim = _trim_from_start(col_mean, col_std, max_col_trim)
    right_trim = _trim_from_end(col_mean, col_std, max_col_trim)

    y1 = top_trim
    y2 = height - bottom_trim
    x1 = left_trim
    x2 = width - right_trim

    if y2 <= y1 or x2 <= x1:
        return image

    return image[y1:y2, x1:x2]


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


def choose_best_digit_candidate(
    candidates: List[Tuple[str, float]],
    expected_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Tuple[str, float]:
    """Pick the digit string that best matches length hints and OCR confidence."""
    best_digits = ""
    best_confidence = 0.0
    best_score = float("-inf")
    seen: set[Tuple[str, int]] = set()

    for raw_digits, confidence in candidates:
        digits = re.sub(r"[^0-9]", "", raw_digits)
        if max_length is not None and len(digits) > max_length:
            digits = digits[:max_length]
        if not digits:
            continue

        key = (digits, int(round(confidence * 1000)))
        if key in seen:
            continue
        seen.add(key)

        length = len(digits)
        score = float(confidence)

        if expected_length is not None:
            score -= abs(length - expected_length) * 0.28
            if length == expected_length:
                score += 0.35
        else:
            score += min(length * 0.05, 0.3)

        if min_length is not None and length < min_length:
            score -= (min_length - length) * 0.25
        if max_length is not None and length > max_length:
            score -= (length - max_length) * 0.2

        if score > best_score:
            best_score = score
            best_digits = digits
            best_confidence = confidence

    return best_digits, best_confidence


def extract_digits(
    image: np.ndarray,
    expected_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    return_confidence: bool = False,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("", 0.0) if return_confidence else ""

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

    _record_debug_image(debug_store, roi_name, "digits: gray", gray)
    _record_debug_image(debug_store, roi_name, "digits: gray clahe", gray_clahe)
    _record_debug_image(debug_store, roi_name, "digits: guide suppressed", guide_suppressed)
    _record_debug_image(debug_store, roi_name, "digits: guide gray", guide_gray)
    _record_debug_image(debug_store, roi_name, "digits: guide clahe", guide_clahe)

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
            return ("", 0.0) if return_confidence else ""

    digits_allowlist = "0123456789"
    candidates = [
        {"label": "gray", "image": gray, "scale": 2.2},
        {"label": "gray clahe", "image": gray_clahe, "scale": 2.4},
        {"label": "color", "image": color, "scale": 2.0},
        {"label": "blue enhanced", "image": blue_enhanced, "scale": 2.8},
        {"label": "blue binary", "image": blue_binary, "scale": 3.0},
        {"label": "blue binary inverted", "image": cv2.bitwise_not(blue_binary), "scale": 3.0},
        {"label": "blue binary clean", "image": binary_clean, "scale": 3.0},
        {"label": "blue binary clean inverted", "image": cv2.bitwise_not(binary_clean), "scale": 3.0},
        {"label": "guide gray", "image": guide_gray, "scale": 2.4},
        {"label": "guide clahe", "image": guide_clahe, "scale": 2.6},
        {"label": "guide suppressed color", "image": guide_suppressed, "scale": 2.2},
        {"label": "guide blue", "image": guide_blue, "scale": 3.0},
        {"label": "guide binary", "image": guide_binary, "scale": 3.0},
        {"label": "guide binary inverted", "image": cv2.bitwise_not(guide_binary), "scale": 3.0},
        {"label": "guide binary clean", "image": guide_binary_clean, "scale": 3.2},
        {"label": "guide binary clean inverted", "image": cv2.bitwise_not(guide_binary_clean), "scale": 3.2},
    ]

    results: List[Tuple[str, float]] = []

    for idx, candidate in enumerate(candidates):
        candidate_image = candidate["image"]
        factor = candidate.get("scale", 2.0)
        scaled = upscale(candidate_image, factor=factor)
        _record_debug_image(
            debug_store,
            roi_name,
            f"digits primary {idx + 1}: {candidate.get('label', 'candidate')}",
            scaled,
            note=f"scale={factor}, min_conf=0.55",
        )
        digits, confidence = read_digits_with_confidence(
            scaled,
            allowlist=digits_allowlist,
            min_confidence=0.55,
        )
        if digits and confidence >= 0.58:
            normalized = re.sub(r"[^0-9]", "", digits)
            if max_length is not None and len(normalized) > max_length:
                normalized = normalized[:max_length]
            if expected_length and len(normalized) == expected_length:
                return (normalized, confidence) if return_confidence else normalized
        if digits:
            results.append((digits, confidence))

    secondary_results: List[Tuple[str, float]] = []

    for idx, candidate in enumerate(candidates):
        candidate_image = candidate["image"]
        factor = candidate.get("scale", 2.0)
        scaled = upscale(candidate_image, factor=factor)
        _record_debug_image(
            debug_store,
            roi_name,
            f"digits secondary {idx + 1}: {candidate.get('label', 'candidate')}",
            scaled,
            note=f"scale={factor}, min_conf=0.40",
        )
        digits, confidence = read_digits_with_confidence(
            scaled,
            allowlist=digits_allowlist,
            min_confidence=0.4,
        )
        if digits and confidence >= 0.5:
            normalized = re.sub(r"[^0-9]", "", digits)
            if max_length is not None and len(normalized) > max_length:
                normalized = normalized[:max_length]
            if expected_length and len(normalized) == expected_length:
                return (normalized, confidence) if return_confidence else normalized
        if digits:
            secondary_results.append((digits, confidence))

    results.extend(secondary_results)

    if ink_ratio >= 0.0015:
        fallback_image = upscale(cv2.bitwise_not(binary_clean), factor=3.0)
        fallback_text = run_easyocr(fallback_image, allowlist=digits_allowlist)
        fallback_digits = re.sub(r"[^0-9]", "", fallback_text)
        if fallback_digits:
            results.append((fallback_digits, 0.38))
            _record_debug_image(
                debug_store,
                roi_name,
                "digits fallback easyocr on inverted clean",
                fallback_image,
                note="easyocr fallback",
            )

    if not results:
        return ("", 0.0) if return_confidence else ""

    best_digits, best_conf = choose_best_digit_candidate(
        results,
        expected_length,
        min_length,
        max_length,
    )
    if return_confidence:
        return best_digits, best_conf
    return best_digits


def _cnp_checksum_ok(digits: str) -> bool:
    if len(digits) != 13 or not digits.isdigit():
        return False
    weights = [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]
    total = sum(int(digits[i]) * weights[i] for i in range(12))
    control = total % 11
    if control == 10:
        control = 1
    return int(digits[12]) == control


def _best_13_digit_substring(raw: str) -> str:
    longest = ""
    best_valid = ""
    for i in range(0, max(0, len(raw) - 12)):
        candidate = raw[i : i + 13]
        if not candidate.isdigit():
            continue
        if len(candidate) > len(longest):
            longest = candidate
        if _cnp_checksum_ok(candidate):
            return candidate
    return best_valid or longest


def _trim_to_content(gray: np.ndarray) -> np.ndarray:
    """Trim uniform margins by locating columns/rows with ink.

    Works on a single-channel image. Returns the cropped gray image. If
    detection fails, returns the original image.
    """
    if gray.size == 0:
        return gray
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # Invert binary so ink is 1s
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = np.sum(binary > 0, axis=0)
    rows = np.sum(binary > 0, axis=1)
    if not np.any(cols) or not np.any(rows):
        return gray
    col_indices = np.where(cols > 0)[0]
    row_indices = np.where(rows > 0)[0]
    x1, x2 = int(col_indices[0]), int(col_indices[-1])
    y1, y2 = int(row_indices[0]), int(row_indices[-1])
    # small padding
    pad_x = max(1, (x2 - x1) // 40)
    pad_y = max(1, (y2 - y1) // 10)
    x1 = max(0, x1 - pad_x)
    x2 = min(gray.shape[1], x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(gray.shape[0], y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return gray
    return gray[y1:y2, x1:x2]


def _cnp_try_single_digit_fix(digits: str, confidences: List[float]) -> str:
    """Try correcting one position (lowest confidence first) to satisfy checksum."""
    if len(digits) != 13 or not digits.isdigit():
        return ""
    order = sorted(range(13), key=lambda i: confidences[i] if i < len(confidences) else 0.0)
    for idx in order:
        original = digits[idx]
        for d in "0123456789":
            if d == original:
                continue
            test = digits[:idx] + d + digits[idx + 1 :]
            if _cnp_checksum_ok(test):
                return test
    return ""


def extract_cnp(
    image: np.ndarray,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
) -> str:
    """Specialised extractor for Romanian CNP (13 digits with checksum).

    Tries multiple preprocessing variants, prefers a 13-digit substring that
    passes checksum; otherwise returns the longest 13-digit run found.
    """
    if image.size == 0:
        return ""

    _record_debug_image(debug_store, roi_name, "cnp: input", image)
    # 1) Try a simple 13-cell segmentation pass. Many forms print 13 small
    # boxes; slicing evenly is often sufficient and robust to stamps.
    try:
        if image.ndim == 3:
            gray_base = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_base = image.copy()

        _record_debug_image(debug_store, roi_name, "cnp: gray base", gray_base)
        # Trim to content to align 13 equal segments better
        gray_base = _trim_to_content(gray_base)
        _record_debug_image(debug_store, roi_name, "cnp: trimmed base", gray_base)
        height, width = gray_base.shape[:2]
        if width >= 130 and height >= 16:
            per = max(1, width // 13)
            assembled: List[str] = []
            confs: List[float] = []
            for idx in range(13):
                x1 = idx * per
                x2 = width if idx == 12 else (idx + 1) * per
                w_slice = gray_base[:, x1:x2]
                # Tighten margins a bit
                inner = shrink_crop(w_slice, margin_ratio=0.08)
                if inner.size == 0:
                    inner = w_slice
                scaled = upscale(inner, factor=2.8)
                _record_debug_image(
                    debug_store,
                    roi_name,
                    f"cnp: cell {idx + 1} scaled",
                    scaled,
                    note="segmented cell",
                )
                digit, conf = read_digits_with_confidence(scaled, allowlist="0123456789", min_confidence=0.28)
                digit = re.sub(r"[^0-9]", "", digit)[:1]
                assembled.append(digit if digit else "")
                confs.append(float(conf) if digit else 0.0)
            candidate = "".join(assembled)
            if len(candidate) == 13 and _cnp_checksum_ok(candidate):
                return candidate
            if len(candidate) == 13 and candidate.isdigit():
                fixed = _cnp_try_single_digit_fix(candidate, confs)
                if fixed:
                    return fixed
                # Keep as fallback if later passes fail
                box_fallback = candidate
            else:
                box_fallback = ""
        else:
            box_fallback = ""
    except Exception:
        box_fallback = ""

    digits, conf = extract_digits(
        image,
        expected_length=13,
        min_length=11,
        max_length=13,
        return_confidence=True,
        debug_store=debug_store,
        roi_name=roi_name,
    )
    if len(digits) == 13 and _cnp_checksum_ok(digits):
        return digits
    if len(digits) == 13 and conf >= 0.45:
        return digits

    # Reuse the rich candidate set from digit extraction, but be more permissive
    # with thresholds and add a couple of morphological joins to connect strokes.
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

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    guide_clahe = clahe.apply(guide_gray)
    _record_debug_image(debug_store, roi_name, "cnp: guide suppressed", guide_suppressed)
    _record_debug_image(debug_store, roi_name, "cnp: guide gray", guide_gray)
    _record_debug_image(debug_store, roi_name, "cnp: guide clahe", guide_clahe)

    # Binary versions that frequently work well for boxed sequences
    blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    joined = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    _record_debug_image(debug_store, roi_name, "cnp: adaptive", adaptive)
    _record_debug_image(debug_store, roi_name, "cnp: joined", joined)

    candidates = [
        {"label": "gray", "image": gray, "scale": 2.4},
        {"label": "gray clahe", "image": gray_clahe, "scale": 2.6},
        {"label": "guide gray", "image": guide_gray, "scale": 2.6},
        {"label": "guide clahe", "image": guide_clahe, "scale": 2.8},
        {"label": "adaptive", "image": adaptive, "scale": 3.0},
        {"label": "adaptive inverted", "image": cv2.bitwise_not(adaptive), "scale": 3.0},
        {"label": "joined", "image": joined, "scale": 3.0},
        {"label": "joined inverted", "image": cv2.bitwise_not(joined), "scale": 3.0},
    ]

    best = ""
    for idx, cand in enumerate(candidates):
        factor = cand.get("scale", 2.5)
        scaled = upscale(cand["image"], factor=factor)
        _record_debug_image(
            debug_store,
            roi_name,
            f"cnp candidate {idx + 1}: {cand.get('label', 'candidate')}",
            scaled,
            note=f"scale={factor}",
        )
        text, conf = read_digits_with_confidence(scaled, allowlist="0123456789", min_confidence=0.35)
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 13:
            chosen = _best_13_digit_substring(digits)
            if _cnp_checksum_ok(chosen):
                return chosen
            if len(chosen) == 13 and len(best) != 13:
                best = chosen
            elif len(chosen) > len(best):
                best = chosen
        elif len(digits) > len(best):
            best = digits

    # Last attempt: use generic extract_digits and validate
    fallback = extract_digits(
        image,
        expected_length=13,
        min_length=11,
        max_length=13,
        debug_store=debug_store,
        roi_name=roi_name,
    )
    chosen = _best_13_digit_substring(fallback)
    if _cnp_checksum_ok(chosen):
        return chosen
    if chosen:
        return chosen
    return box_fallback

def extract_text(
    image: np.ndarray,
    allowlist: Optional[str] = None,
    return_confidence: bool = False,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("", 0.0) if return_confidence else ""

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

    _record_debug_image(debug_store, roi_name, "text: gray", gray)
    _record_debug_image(debug_store, roi_name, "text: gray clahe", gray_clahe)
    _record_debug_image(debug_store, roi_name, "text: dark enhanced", dark_enhanced)
    _record_debug_image(debug_store, roi_name, "text: blue enhanced", blue_enhanced)
    _record_debug_image(debug_store, roi_name, "text: guide suppressed", guide_suppressed)
    _record_debug_image(debug_store, roi_name, "text: guide gray", guide_gray)
    _record_debug_image(debug_store, roi_name, "text: guide clahe", guide_clahe)
    _record_debug_image(debug_store, roi_name, "text: guide dark", guide_dark)
    _record_debug_image(debug_store, roi_name, "text: guide blue", guide_blue)
    _record_debug_image(debug_store, roi_name, "text: adaptive", adaptive)
    _record_debug_image(debug_store, roi_name, "text: closed", closed)
    _record_debug_image(debug_store, roi_name, "text: cleaned grid", cleaned_grid)
    _record_debug_image(debug_store, roi_name, "text: guide adaptive", guide_adaptive)
    _record_debug_image(debug_store, roi_name, "text: guide closed", guide_closed)
    _record_debug_image(debug_store, roi_name, "text: guide clean", guide_clean)

    base_allowlist = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz-_/.,"
    )
    primary_allowlist = allowlist if allowlist is not None else base_allowlist

    candidates = [
        {"label": "gray", "image": gray, "scale": 2.0, "min_conf": 0.38, "allowlist": None},
        {"label": "gray clahe", "image": gray_clahe, "scale": 2.2, "min_conf": 0.4, "allowlist": None},
        {"label": "dark enhanced", "image": dark_enhanced, "scale": 2.3, "min_conf": 0.4, "allowlist": None},
        {"label": "adaptive inverted", "image": cv2.bitwise_not(adaptive), "scale": 2.6, "min_conf": 0.45, "allowlist": primary_allowlist},
        {"label": "adaptive", "image": adaptive, "scale": 2.6, "min_conf": 0.45, "allowlist": primary_allowlist},
        {"label": "closed inverted", "image": cv2.bitwise_not(closed), "scale": 2.8, "min_conf": 0.45, "allowlist": primary_allowlist},
        {"label": "closed", "image": closed, "scale": 2.8, "min_conf": 0.45, "allowlist": primary_allowlist},
        {"label": "cleaned grid inverted", "image": cv2.bitwise_not(cleaned_grid), "scale": 2.8, "min_conf": 0.42, "allowlist": primary_allowlist},
        {"label": "cleaned grid", "image": cleaned_grid, "scale": 2.8, "min_conf": 0.42, "allowlist": primary_allowlist},
        {"label": "guide gray", "image": guide_gray, "scale": 2.2, "min_conf": 0.38, "allowlist": None},
        {"label": "guide clahe", "image": guide_clahe, "scale": 2.4, "min_conf": 0.4, "allowlist": None},
        {"label": "guide dark", "image": guide_dark, "scale": 2.5, "min_conf": 0.4, "allowlist": None},
        {"label": "guide blue", "image": guide_blue, "scale": 2.7, "min_conf": 0.45, "allowlist": primary_allowlist},
        {"label": "guide adaptive", "image": guide_adaptive, "scale": 2.7, "min_conf": 0.45, "allowlist": primary_allowlist},
        {"label": "guide adaptive inverted", "image": cv2.bitwise_not(guide_adaptive), "scale": 2.7, "min_conf": 0.45, "allowlist": primary_allowlist},
        {"label": "guide closed", "image": guide_closed, "scale": 2.9, "min_conf": 0.46, "allowlist": primary_allowlist},
        {"label": "guide closed inverted", "image": cv2.bitwise_not(guide_closed), "scale": 2.9, "min_conf": 0.46, "allowlist": primary_allowlist},
        {"label": "guide clean", "image": guide_clean, "scale": 2.9, "min_conf": 0.44, "allowlist": primary_allowlist},
        {"label": "guide clean inverted", "image": cv2.bitwise_not(guide_clean), "scale": 2.9, "min_conf": 0.44, "allowlist": primary_allowlist},
    ]

    if np.count_nonzero(blue_enhanced) > 0:
        candidates.append(
            {"label": "blue enhanced", "image": blue_enhanced, "scale": 2.7, "min_conf": 0.46, "allowlist": primary_allowlist}
        )
        candidates.append(
            {"label": "blue enhanced inverted", "image": cv2.bitwise_not(blue_enhanced), "scale": 2.7, "min_conf": 0.46, "allowlist": primary_allowlist}
        )

    best_text = ""
    best_confidence = 0.0

    for idx, candidate in enumerate(candidates):
        candidate_image = candidate["image"]
        factor = candidate.get("scale", 2.0)
        min_conf = candidate.get("min_conf", 0.35)
        candidate_allowlist = candidate.get("allowlist")
        scaled = upscale(candidate_image, factor=factor)
        _record_debug_image(
            debug_store,
            roi_name,
            f"text candidate {idx + 1}: {candidate.get('label', 'candidate')}",
            scaled,
            note=f"scale={factor}, min_conf={min_conf}",
        )
        text, confidence = read_text_with_confidence(
            scaled,
            allowlist=candidate_allowlist,
            min_confidence=min_conf,
        )
        cleaned_text = tidy_text(text)
        if cleaned_text and confidence >= (min_conf + 0.05):
            return (cleaned_text, confidence) if return_confidence else cleaned_text
        if cleaned_text:
            better_conf = confidence > best_confidence + 0.02
            similar_conf = abs(confidence - best_confidence) <= 0.02 and len(cleaned_text) > len(best_text)
            if better_conf or similar_conf:
                best_text = cleaned_text
                best_confidence = confidence

    if best_text:
        return (best_text, best_confidence) if return_confidence else best_text

    fallback_image = upscale(gray, factor=2.4)
    fallback = tidy_text(run_easyocr(fallback_image, allowlist=allowlist))
    _record_debug_image(
        debug_store,
        roi_name,
        "text fallback easyocr",
        fallback_image,
        note="easyocr fallback",
    )
    if return_confidence:
        fallback_conf = 0.25 if fallback else 0.0
        return fallback, fallback_conf
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


def extract_date(
    image: np.ndarray,
    expected_length: int = 6,
    return_confidence: bool = False,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
) -> Union[str, Tuple[str, float]]:
    digits, digits_conf = extract_digits(
        image,
        expected_length=expected_length,
        min_length=expected_length,
        max_length=expected_length,
        return_confidence=True,
        debug_store=debug_store,
        roi_name=roi_name,
    )
    formatted = format_date_from_digits(digits)
    final_conf = digits_conf
    if formatted:
        if return_confidence:
            return formatted, final_conf
        return formatted
    trimmed = re.sub(r"[^0-9]", "", digits)
    if expected_length and len(trimmed) >= expected_length:
        trimmed = trimmed[:expected_length]
    if return_confidence:
        return trimmed, final_conf
    return trimmed


def parse_date(raw: str) -> str:
    digits = re.sub(r"[^0-9]", "", raw)
    formatted = format_date_from_digits(digits)
    if formatted:
        return formatted
    return digits[:6].strip()


def detect_checkbox(
    image: np.ndarray,
    return_confidence: bool = False,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("false", 0.0) if return_confidence else "false"
    if image.ndim == 3:
        color = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    _record_debug_image(debug_store, roi_name, "checkbox: input", color)

    inner = shrink_crop(color, margin_ratio=0.12)
    if inner.size == 0:
        inner = color
    inner = auto_trim_borders(inner, max_fraction=0.2, intensity_threshold=210.0, variance_threshold=22.0)
    _record_debug_image(debug_store, roi_name, "checkbox: trimmed", inner)

    cleaned = suppress_guidelines(inner)
    if cleaned.ndim == 3:
        cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    else:
        cleaned_gray = cleaned
    _record_debug_image(debug_store, roi_name, "checkbox: cleaned", cleaned)
    _record_debug_image(debug_store, roi_name, "checkbox: cleaned gray", cleaned_gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(cleaned_gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _record_debug_image(debug_store, roi_name, "checkbox: enhanced", enhanced)
    _record_debug_image(debug_store, roi_name, "checkbox: binary", binary)

    kernel = np.ones((3, 3), np.uint8)
    strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    _record_debug_image(debug_store, roi_name, "checkbox: strokes", strokes)

    height, width = strokes.shape[:2]
    total_area = float(height * width)
    if total_area == 0:
        return ("false", 0.0) if return_confidence else "false"

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    significant_area = 0.0
    elongated_component = False
    max_component_area = 0.0
    for idx in range(1, num_labels):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        significant_area += area
        if area > max_component_area:
            max_component_area = area
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        if w >= 0.35 * width or h >= 0.35 * height:
            elongated_component = True

    ink_ratio = significant_area / total_area
    if significant_area <= 0:
        return ("false", 0.0) if return_confidence else "false"

    core_margin = max(1, int(round(min(height, width) * 0.18)))
    if core_margin * 2 >= min(height, width):
        core_margin = max(1, min(height, width) // 3)
    core = strokes[core_margin: height - core_margin, core_margin: width - core_margin]
    if core.size == 0:
        core = strokes
    core_non_zero = float(cv2.countNonZero(core))
    core_ratio = core_non_zero / total_area
    edge_ratio = max(0.0, ink_ratio - core_ratio)

    central_strength = core_non_zero >= 14 and core_ratio >= 0.0028
    edge_dominant = edge_ratio > core_ratio * 1.2

    diagonal_presence = False
    if max_component_area >= min(24.0, total_area * 0.02):
        dilated = cv2.dilate(strokes, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=18,
            minLineLength=min(width, height) * 0.4,
            maxLineGap=4,
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                angle = min(angle, 180 - angle)
                if 30 <= angle <= 60:
                    diagonal_presence = True
                    break

    if elongated_component and central_strength and not edge_dominant and ink_ratio >= 0.006:
        confidence = min(1.0, max(0.45, ink_ratio * 90))
        if diagonal_presence:
            confidence = min(1.0, confidence + 0.1)
        return ("true", confidence) if return_confidence else "true"

    if central_strength and not edge_dominant and (ink_ratio >= 0.007 or diagonal_presence):
        confidence = min(1.0, max(0.4, ink_ratio * 80 + (0.15 if diagonal_presence else 0.0)))
        return ("true", confidence) if return_confidence else "true"

    confidence = max(0.0, min(0.4, core_ratio * 40))
    return ("false", confidence) if return_confidence else "false"


def extract_roi_value(
    roi: RoiSpec,
    color_img: np.ndarray,
    gray_img: np.ndarray,
    debug_store: Optional[Dict[str, List[dict]]] = None,
) -> str:
    height, width = gray_img.shape[:2]
    y1, y2, x1, x2 = roi_to_pixels(roi, height, width)
    crop_color = color_img[y1:y2, x1:x2]
    _record_debug_image(debug_store, roi.name, "roi: raw crop", crop_color)
    margin_defaults = {
        "digits": 0.06,
        "date": 0.06,
        "checkbox": 0.08,
        "text": 0.05,
    }
    auto_trim_defaults = {
        "digits": None,
        "date": None,
        "checkbox": 0.18,
        "text": None,
    }
    margin = roi.margin if roi.margin is not None else margin_defaults.get(roi.kind, 0.05)
    if roi.name in {"cnp", "cnp_copil"} and roi.margin is None:
        margin = 0.02
    if margin and margin > 0:
        crop_color = shrink_crop(crop_color, margin_ratio=margin)
        _record_debug_image(debug_store, roi.name, "roi: margin trimmed", crop_color, note=f"margin={margin}")

    auto_trim_fraction = roi.auto_trim if roi.auto_trim is not None else auto_trim_defaults.get(roi.kind)
    if auto_trim_fraction and auto_trim_fraction > 0:
        crop_color = auto_trim_borders(crop_color, max_fraction=auto_trim_fraction)
        _record_debug_image(
            debug_store,
            roi.name,
            "roi: auto-trimmed",
            crop_color,
            note=f"auto_trim={auto_trim_fraction}",
        )

    if roi.kind == "digits":
        if roi.name in {"cnp", "cnp_copil"}:
            value = extract_cnp(crop_color, debug_store=debug_store, roi_name=roi.name)
        else:
            expected = roi.expected_length
            min_len = roi.min_length if roi.min_length is not None else expected
            max_len = roi.max_length if roi.max_length is not None else expected
            value = extract_digits(
                crop_color,
                expected_length=expected,
                min_length=min_len,
                max_length=max_len,
                debug_store=debug_store,
                roi_name=roi.name,
            )
    elif roi.kind == "date":
        expected = roi.expected_length if roi.expected_length is not None else 6
        value = extract_date(
            crop_color,
            expected_length=expected,
            debug_store=debug_store,
            roi_name=roi.name,
        )
    elif roi.kind == "checkbox":
        value = detect_checkbox(crop_color, debug_store=debug_store, roi_name=roi.name)
    else:
        value = extract_text(crop_color, allowlist=roi.allowlist, debug_store=debug_store, roi_name=roi.name)
    return value.strip()

def extract_fields(
    uploaded_file,
    preview: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    debug: bool = False,
):
    """Return structured values and optional ROI preview overlay."""
    total_roi = len(ROI_SPECS)
    base_steps = 3.0
    total_steps = float(max(total_roi + base_steps, 1.0))
    completed = 0.0
    debug_store: Optional[Dict[str, List[dict]]] = {} if debug else None

    def advance(message: str, weight: float = 1.0) -> None:
        nonlocal completed
        completed += weight
        fraction = min(max(completed / total_steps, 0.0), 1.0)
        if progress_callback:
            progress_callback(fraction, message)

    if progress_callback:
        progress_callback(0.0, "Reading uploaded image")

    original = load_image(uploaded_file)
    advance("Loaded image data")
    aligned, warped = align_certificate(original)
    align_message = "Aligned certificate" if warped else "Resized certificate"
    advance(align_message)
    if not warped:
        aligned = cv2.resize(original, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

    gray = preprocess_for_ocr(aligned)
    advance("Prepared image for OCR analysis")

    preview_img = aligned.copy()

    extracted = {}
    for roi in ROI_SPECS:
        extracted[roi.name] = extract_roi_value(roi, aligned, gray, debug_store=debug_store)
        advance(f"Processed ROI: {roi.name}", weight=1.0)

    if progress_callback:
        progress_callback(1.0, "Extraction complete")

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
        return preview_img, extracted, (debug_store or {})

    return extracted, (debug_store or {})
