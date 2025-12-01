from __future__ import annotations

import datetime
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import easyocr
import numpy as np
from PIL import Image
from deskew import determine_skew

reader = easyocr.Reader(["en"], gpu=True)

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

def tidy_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()

def run_easyocr(image: np.ndarray, allowlist: Optional[str] = None) -> str:
    if image.size == 0:
        return ""
    rgb = cv2.cvtColor(image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    lines = reader.readtext(rgb, detail=0, paragraph=True, allowlist=allowlist)
    return " ".join(lines).strip()

def easyocr_raw_read(image: np.ndarray, allowlist: Optional[str] = None) -> Tuple[str, float]:
    """Run EasyOCR without thresholding; return concatenated text and best confidence."""
    if image.size == 0:
        return "", 0.0
    rgb = cv2.cvtColor(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, detail=1, paragraph=False, allowlist=allowlist)
    texts: List[str] = []
    best_conf = 0.0
    for _bbox, text, conf in results:
        if not text:
            continue
        texts.append(text)
        best_conf = max(best_conf, float(conf))
    return " ".join(texts).strip(), best_conf

def detect_blue_checkbox(roi_bgr: np.ndarray) -> str:
    blue = emphasize_blue_ink(roi_bgr)
    binv = cv2.adaptiveThreshold(blue, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 7)
    ratio = cv2.countNonZero(binv) / float(binv.size)
    return "true" if ratio >= 0.006 else "false"


def _base_gray_clahe(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew_img(image: np.ndarray) -> np.ndarray:
    try:
        angle = determine_skew(image)
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(
            np.cos(angle_radian) * old_width
        )
        height = abs(np.sin(angle_radian) * old_width) + abs(
            np.cos(angle_radian) * old_height
        )

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(
            image,
            rot_mat,
            (int(round(height)), int(round(width))),
            borderValue=(0, 0, 0),
        )
    except:
        return image

def normalize_img(img: np.ndarray) -> np.ndarray:
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def denoise_img(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, 5, 55, 60)

def grayscale_img(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    variants.append(("emphasis_inv", cv2.bitwise_not(emphasis)))
    denoise = cv2.medianBlur(emphasis, 3)
    variants.append(("emphasis_denoised_inv", cv2.bitwise_not(denoise)))
    return variants


def extract_digits(
    image: np.ndarray,
    expected_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
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
    best_score = float("-inf")
    best_conf = 0.0
    for idx, (label, variant) in enumerate(variants):
        scaled = upscale(variant, factor=3.0)
        _record_debug_image(debug_store, roi_name, f"digits variant {idx + 1}: {label}", scaled)
        raw_text, conf = easyocr_raw_read(scaled, allowlist="0123456789")
        cleaned = re.sub(r"[\s\W]+", "", raw_text)
        digits_only = re.sub(r"[^0-9]", "", cleaned)
        if max_length is not None and len(digits_only) > max_length:
            digits_only = digits_only[:max_length]
        candidates = [digits_only]
        target_len = expected_length or max_length
        if target_len and len(digits_only) > target_len:
            for start in range(0, len(digits_only) - target_len + 1):
                window = digits_only[start : start + target_len]
                candidates.append(window)
        for candidate_text in candidates:
            if expected_length is not None and len(candidate_text) != expected_length:
                continue
            if min_length is not None and len(candidate_text) < min_length:
                continue
            score = conf + min(len(candidate_text) * 0.02, 0.5)
            if score > best_score:
                best_score = score
                best_text = candidate_text
                best_conf = conf
    if not best_text:
        return ("", 0.0) if return_confidence else ""
    if max_length is not None and len(best_text) > max_length:
        best_text = best_text[:max_length]
    return (best_text, best_conf) if return_confidence else best_text


def extract_cnp(
    image: np.ndarray,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
    color_path: str = "blue",
) -> str:
    if image.size == 0:
        return ""
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    variants = _generate_variants(color, color_path)
    best_cnp = ""
    best_score = float("-inf")
    for idx, (label, variant) in enumerate(variants):
        scaled = upscale(variant, factor=3.0)
        _record_debug_image(debug_store, roi_name, f"cnp variant {idx + 1}: {label}", scaled)
        raw_text, conf = easyocr_raw_read(scaled, allowlist="0123456789")
    if not best_cnp:
        return ""
    if not (len(best_cnp) == 13 and best_cnp[0] in "1256"):
        return ""
    return best_cnp


def extract_text(
    image: np.ndarray,
    allowlist: Optional[str] = None,
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
    best_score = float("-inf")
    best_conf = 0.0
    for idx, (label, variant) in enumerate(variants):
        scaled = upscale(variant, factor=2.8)
        _record_debug_image(debug_store, roi_name, f"text variant {idx + 1}: {label}", scaled)
        text, conf = easyocr_raw_read(scaled, allowlist=allowlist)
        cleaned_text = tidy_text(text)
        if not cleaned_text:
            continue
        score = conf + min(len(cleaned_text) * 0.01, 0.4)
        if score > best_score:
            best_score = score
            best_text = cleaned_text
            best_conf = conf
    if return_confidence:
        return best_text, best_conf
    return best_text


def extract_date(
    image: np.ndarray,
    expected_length: int = 6,
    return_confidence: bool = False,
    debug_store: Optional[Dict[str, List[dict]]] = None,
    roi_name: Optional[str] = None,
    prefer_century: Optional[int] = None,
    color_path: str = "blue",
) -> Union[str, Tuple[str, float]]:
    if image.size == 0:
        return ("", 0.0) if return_confidence else ""
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    variants = _generate_variants(color, color_path)
    best_value = ""
    best_score = float("-inf")
    best_conf = 0.0
    for idx, (label, variant) in enumerate(variants):
        scaled = upscale(variant, factor=3.0)
        _record_debug_image(debug_store, roi_name, f"date variant {idx + 1}: {label}", scaled)
        raw_text, conf = easyocr_raw_read(scaled, allowlist="0123456789.")
        cleaned_digits = re.sub(r"[^0-9]", "", raw_text)

        candidates = [cleaned_digits]
        if expected_length and len(cleaned_digits) > expected_length:
            for start in range(0, len(cleaned_digits) - expected_length + 1):
                window = cleaned_digits[start : start + expected_length]
                candidates.append(window)
        for candidate in candidates:
            if len(candidate) != expected_length:
                continue
            elif expected_length == 8:
                day = int(candidate[:2])
                month = int(candidate[2:4])
                year = int(candidate[4:8])
                try:
                    _ = datetime.date(year, month, day)
                    valid = True
                except ValueError:
                    valid = False
            else:
                valid = True

            if not valid:
                continue

            score = conf + min(len(candidate) * 0.02, 0.5)
            if score > best_score:
                best_score = score
                best_value = candidate
                best_conf = conf

    if best_score == float("-inf"):
        return ("", 0.0) if return_confidence else ""
    if return_confidence:
        return best_value, best_conf
    return best_value


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
    _record_debug_image(debug_store, roi_name, "checkbox: input", color)
    value = detect_blue_checkbox(color)
    if return_confidence:
        return value, 1.0 if value == "true" else 0.0
    return value


def _parse_date_strict(value: str) -> Optional[datetime.date]:
    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) == 8:
        day, month, year = int(digits[:2]), int(digits[2:4]), int(digits[4:8])
    elif len(digits) >= 6:
        day, month, year2 = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
        year = 2000 + year2 if year2 < 50 else 1900 + year2
    else:
        return None
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


def _format_date(dt: datetime.date) -> str:
    return dt.strftime("%d.%m.%Y")

def extract_roi_value(
    roi: RoiSpec,
    color_img: np.ndarray,
    gray_img: np.ndarray,
    debug_store: Optional[Dict[str, List[dict]]] = None,
) -> str:
    height, width = gray_img.shape[:2]
    y1, y2, x1, x2 = roi_to_pixels(roi, height, width)
    crop_color = color_img[y1:y2, x1:x2]
    color_path = choose_color_path(roi)

    if roi.kind == "digits":
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
            color_path=color_path,
        )
        value = re.sub(r"[^0-9]", "", value)
    elif roi.kind == "date":
        expected = roi.expected_length if roi.expected_length is not None else 6
        prefer_century = None
        if roi.name in {"de_la", "pana_la", "data_acordarii", "data_primirii"}:
            prefer_century = None
        value = extract_date(
            crop_color,
            expected_length=expected,
            debug_store=debug_store,
            roi_name=roi.name,
            prefer_century=prefer_century,
            color_path=color_path,
        )
        if roi.name == "data_primirii":
            parsed = _parse_date_strict(value)
            value = _format_date(parsed) if parsed else ""
    elif roi.kind == "checkbox":
        value = detect_checkbox(crop_color, debug_store=debug_store, roi_name=roi.name, color_path=color_path)
    else:
        text_allowlist = roi.allowlist
        if roi.name in {"cod_parafa_medic", "cod_parafa_medic_sef"}:
            text_allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        value = extract_text(
            crop_color,
            allowlist=text_allowlist,
            debug_store=debug_store,
            roi_name=roi.name,
            color_path=color_path,
        )
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

    # _record_debug_image(debug_store, "preview", f"original", original)
    # _record_debug_image(debug_store, "preview", f"aligned", aligned)
    # _record_debug_image(debug_store, "preview", f"preprocessed_gray", gray)
    # _record_debug_image(debug_store, "preview", f"preview img", preview_img)
    # _record_debug_image(debug_store, "preview", f"deskew original", deskew_img(original))
    # _record_debug_image(debug_store, "preview", f"deskew aligned", deskew_img(aligned))
    # _record_debug_image(debug_store, "preview", f"denoise original", denoise_img(original))

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
