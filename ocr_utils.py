from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from paddleocr import TextRecognition
from skimage.filters.rank import threshold

paddle_model = TextRecognition(model_name="en_PP-OCRv5_mobile_rec")
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

@dataclass(frozen=True)
class ExtractedField:
    name: str
    value: str
    confidence: float
    needs_review: bool
    review_reason: Optional[str] = None

def choose_color_path(roi: RoiSpec) -> str:
    """Return the preferred color emphasis path for a ROI."""
    if roi.preferred_ink == "black":
        return "black"
    return "blue"

def _record_debug_image(
    store: Optional[Dict[str, List[dict]]],
    roi_name: Optional[str],
    label: str,
    image: Optional[np.ndarray],
    note: Optional[str] = None,
    confidence: Optional[float] = None,
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
    if confidence is not None:
        entry["confidence"] = float(confidence)
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
    combined = cv2.max(mask, dominance)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
    return enhanced

def emphasize_blue_ink_thin(image: np.ndarray) -> np.ndarray:
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

    combined = cv2.max(mask, dominance)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)

    return enhanced

def tidy_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().replace(" ", "")

DIGIT_LOOKALIKES = {
    "O": "0",
    "o": "0",
    "D": "0",
    "I": "1",
    "l": "1",
    "L": "1",
    "|": "1",
    "Z": "2",
    "S": "5",
    "s": "5",
    "B": "8",
    "G": "6",
    "g": "9",
    "q": "9",
    "y": "7",
    "f": "7",
    "/": "7",
    "h": "4",
    "A": "4",
    "d": "2",
}

MIN_CONFIDENCE_BY_KIND = {
    "checkbox": 0.9,
    "digits": 0.65,
    "date": 0.75,
    "text": 0.65,
}

def normalize_digits_from_text(text: str) -> str:
    """Collapse whitespace, map letter lookalikes to digits, strip non-digit characters."""
    compact = re.sub(r"\s+", "", text or "")
    mapped_chars: List[str] = []
    for ch in compact:
        mapped_chars.append(DIGIT_LOOKALIKES.get(ch, DIGIT_LOOKALIKES.get(ch.upper(), ch)))
    mapped = "".join(mapped_chars)
    return re.sub(r"\D", "", mapped)

def format_date_digits(clean_digits: str) -> Optional[str]:
    """Format ddmmyy or ddmmyyyy strings into dd.mm.yyyy."""
    if len(clean_digits) == 8:
        return f"{clean_digits[0:2]}.{clean_digits[2:4]}.{clean_digits[4:8]}"
    if len(clean_digits) == 6:
        return f"{clean_digits[0:2]}.{clean_digits[2:4]}.20{clean_digits[4:6]}"
    return None

def paddle_textrec_read(
    image: np.ndarray
) -> Tuple[str, float]:
    """Run PaddleOCR TextRecognition on an image and return text and confidence."""
    if image.size == 0:
        return "", 0.0
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        res = paddle_model.predict(input=rgb)
    except TypeError:
        res = paddle_model.predict(rgb)

    if isinstance(res, tuple) and len(res) > 0:
        res = res[0]

    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        payload = res[0]
        text = payload.get("rec_text", "") or ""
        score = float(payload.get("rec_score", 0.0) or 0.0)
        return text, score

    return "", 0.0

def _base_gray_clahe(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def denoise_img(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, 5, 55, 60)

def threshold_img(img: np.ndarray, threshold_val: int) -> np.ndarray:
    _, img_thresh = cv2.threshold(img, threshold_val, 255, 1)
    return img_thresh

def _generate_variants(image: np.ndarray, color_path: str) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []
    if color_path == "black":
        base_gray = _base_gray_clahe(image)
        variants.append(("gray", base_gray))
        return variants

    emphasis = emphasize_blue_ink(image)
    emphasis_thin = emphasize_blue_ink_thin(image)
    variants.append(("emphasis_thin", emphasis_thin))
    variants.append(("emphasis_thin_inv", cv2.bitwise_not(emphasis_thin)))
    variants.append(("emphasis", emphasis))
    variants.append(("emphasis_inv", cv2.bitwise_not(emphasis)))
    denoise = cv2.medianBlur(emphasis, 3)
    variants.append(("emphasis_denoised", denoise))
    variants.append(("emphasis_denoised_inv", cv2.bitwise_not(denoise)))
    denoise_thin = cv2.medianBlur(emphasis_thin, 3)
    variants.append(("emphasis_thin_denoised", denoise_thin))
    variants.append(("emphasis_thin_denoised_inv", cv2.bitwise_not(denoise_thin)))
    return variants

@dataclass
class Candidate:
    text: str
    conf: float
    img: np.ndarray


def _pick_best_candidate(
    candidates: List[Candidate],
    roi: Optional[RoiSpec],
) -> Tuple[str, float, Optional[np.ndarray]]:
    """
    Priority:
    1. Candidates whose len(text) == roi.expected_length (if present), pick highest conf.
    2. Else candidates whose len(text) is within [roi.min_length, roi.max_length] (where defined), pick highest conf.
    3. Else pick highest conf overall.
    """
    if not candidates:
        return "", 0.0, None

    # If no ROI at all, just pick by confidence
    if roi is None:
        best = max(candidates, key=lambda c: c.conf)
        return best.text, best.conf, best.img

    expected_length = getattr(roi, "expected_length", None)
    min_length = getattr(roi, "min_length", None)
    max_length = getattr(roi, "max_length", None)

    # 1) Exact expected_length matches
    exact_matches: List[Candidate] = []
    if expected_length is not None:
        exact_matches = [c for c in candidates if len(c.text) == expected_length]

    if exact_matches:
        best = max(exact_matches, key=lambda c: c.conf)
        return best.text, best.conf, best.img

    # 2) In-range matches, if we have any length constraints
    ranged_matches: List[Candidate] = []
    if min_length is not None or max_length is not None:
        for c in candidates:
            l = len(c.text)
            if min_length is not None and l < min_length:
                continue
            if max_length is not None and l > max_length:
                continue
            ranged_matches.append(c)

    if ranged_matches:
        best = max(ranged_matches, key=lambda c: c.conf)
        return best.text, best.conf, best.img

    # 3) Fallback: best by confidence overall
    best = max(candidates, key=lambda c: c.conf)
    return best.text, best.conf, best.img

def extract_digits(
    image: np.ndarray,
    return_confidence: bool = False,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi: Optional[RoiSpec] = None,
    color_path: str = "blue",
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("", 0.0) if return_confidence else ""

    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    variants = _generate_variants(color, color_path)

    candidates: List[Candidate] = []

    for idx, (label, variant) in enumerate(variants):
        scaled = upscale(variant, factor=2.5)
        raw_text, conf = paddle_textrec_read(scaled)
        text = normalize_digits_from_text(raw_text)

        candidates.append(Candidate(text=text, conf=conf, img=scaled))

        _record_debug_image(
            debug_store,
            roi.name if roi else None,
            f"digits variant {idx + 1}: {label}",
            scaled,
            confidence=conf,
            note=f"raw: {raw_text} | cleaned: {text}",
        )

    best_text, best_conf, best_img = _pick_best_candidate(candidates, roi)

    _record_debug_image(
        debug_store,
        roi.name if roi else None,
        "final version",
        best_img if best_img is not None else color,
        confidence=best_conf,
        note=best_text,
    )

    return (best_text, best_conf) if return_confidence else best_text

def extract_text(
    image: np.ndarray,
    return_confidence: bool = False,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
    color_path: str = "blue",
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("", 0.0) if return_confidence else ""
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    variants = _generate_variants(color, color_path)
    best_text = ""
    best_conf = 0.0
    best_score = float("-inf")
    best_image: Optional[np.ndarray] = None

    for idx, (label, variant) in enumerate(variants):
        scaled = upscale(variant, factor=2.8)
        text, conf = paddle_textrec_read(scaled)
        cleaned_text = tidy_text(text)
        score = conf + min(len(cleaned_text) * 0.01, 0.4) if cleaned_text else float("-inf")
        if score > best_score:
            best_score = score
            best_text = cleaned_text
            best_conf = conf
            best_image = scaled
        _record_debug_image(
            debug_store,
            roi_name,
            f"text variant {idx + 1}: {label}",
            scaled,
            confidence=conf,
            note=f"{text}",
        )

    _record_debug_image(
        debug_store,
        roi_name,
        "final version",
        best_image if best_image is not None else color,
        note=best_text,
        confidence=best_conf,
    )
    return (best_text, best_conf) if return_confidence else best_text

def detect_checkbox(
    image: np.ndarray,
    return_confidence: bool = False,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
    color_path: str = "blue",
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("false", 0.0) if return_confidence else "false"

    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    emphasis = _base_gray_clahe(color) if color_path == "black" else emphasize_blue_ink(color)
    _, mask = cv2.threshold(emphasis, 200, 255, cv2.THRESH_BINARY)

    ratio = float(cv2.countNonZero(mask)) / float(mask.size) if mask.size else 0.0
    checked = ratio > 0.003
    value = "true" if checked else "false"
    confidence = float(min(1.0, ratio / 0.01)) if checked else float(max(0.0, 1.0 - ratio / 0.01))

    note = f"value={value}, ratio={ratio:.5f}"
    _record_debug_image(debug_store, roi_name, "checkbox emphasis", emphasis, confidence=confidence, note=note)
    if return_confidence:
        return value, confidence
    return value

def extract_roi_value(
    roi: RoiSpec,
    color_img: np.ndarray,
    gray_img: np.ndarray,
    debug_store: Optional[Dict[str, List[dict]]] = None,
) -> ExtractedField:
    height, width = gray_img.shape[:2]
    y1, y2, x1, x2 = roi_to_pixels(roi, height, width)
    crop_color = color_img[y1:y2, x1:x2]
    color_path = choose_color_path(roi)
    _record_debug_image(debug_store, roi.name, f"roi crop", crop_color)
    raw_value: str
    confidence: float

    if roi.kind in ("digits", "date"):
        raw_value, confidence = extract_digits(
            crop_color,
            return_confidence=True,
            debug_store=debug_store,
            roi=roi,
            color_path=color_path,
        )
        cleaned_digits = normalize_digits_from_text(raw_value)
        value = cleaned_digits
        if roi.kind == "date":
            formatted = format_date_digits(cleaned_digits)
            value = formatted if formatted is not None else cleaned_digits
    elif roi.kind == "checkbox":
        raw_value, confidence = detect_checkbox(
            crop_color,
            return_confidence=True,
            debug_store=debug_store,
            roi_name=roi.name,
            color_path=color_path,
        )
        value = raw_value.strip().replace(" ", "")
    else:
        raw_value, confidence = extract_text(
            crop_color,
            return_confidence=True,
            debug_store=debug_store,
            roi_name=roi.name,
            color_path=color_path,
        )
        value = tidy_text(raw_value)

    min_conf = MIN_CONFIDENCE_BY_KIND.get(roi.kind, MIN_CONFIDENCE_BY_KIND["text"])
    reasons: List[str] = []
    length = len(value)

    if roi.kind != "checkbox" and not value:
        reasons.append("empty value")
    if confidence < min_conf:
        reasons.append(f"low confidence {confidence:.2f} < {min_conf:.2f}")
    if roi.expected_length is not None and length != roi.expected_length:
        reasons.append(f"length {length} != expected {roi.expected_length}")
    if roi.min_length is not None and length < roi.min_length:
        reasons.append(f"length {length} < min {roi.min_length}")
    if roi.max_length is not None and length > roi.max_length:
        reasons.append(f"length {length} > max {roi.max_length}")

    needs_review = bool(reasons)
    review_reason = "; ".join(reasons) if reasons else None
    return ExtractedField(
        name=roi.name,
        value=value,
        confidence=float(confidence),
        needs_review=needs_review,
        review_reason=review_reason,
    )


def extract_fields(
    uploaded_file,
    preview: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    debug: bool = False,
):
    """Return normalized values, confidences, review flags, and optional ROI preview overlay."""
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

    # _record_debug_image(debug_store, "preview", f"original", original)
    # _record_debug_image(debug_store, "preview", f"aligned", aligned)
    # _record_debug_image(debug_store, "preview", f"preprocessed_gray", gray)
    # _record_debug_image(debug_store, "preview", f"preview img", preview_img)
    # _record_debug_image(debug_store, "preview", f"deskew original", deskew_img(original))
    # _record_debug_image(debug_store, "preview", f"deskew aligned", deskew_img(aligned))
    # _record_debug_image(debug_store, "preview", f"denoise original", denoise_img(original))

    values: Dict[str, str] = {}
    confidences: Dict[str, float] = {}
    review_info: Dict[str, Dict[str, Any]] = {}
    for roi in ROI_SPECS:
        field = extract_roi_value(roi, aligned, gray, debug_store=debug_store)
        values[roi.name] = field.value
        confidences[roi.name] = field.confidence
        review_info[roi.name] = {
            "needs_review": field.needs_review,
            "reason": field.review_reason or "",
            "description": roi.description,
            "kind": roi.kind,
        }
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
        return preview_img, values, confidences, review_info, (debug_store or {})

    return values, confidences, review_info, (debug_store or {})
