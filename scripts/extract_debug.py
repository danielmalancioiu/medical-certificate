import cv2
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
aligned_path = BASE / "aligned.png"
img = cv2.imread(str(aligned_path))
if img is None:
    raise SystemExit("aligned.png not found")

height, width = img.shape[:2]

with open(BASE / "roi_map.json", "r", encoding="utf-8-sig") as fh:
    rois = json.load(fh)

out_dir = BASE / "debug_rois"
out_dir.mkdir(exist_ok=True)

for name, spec in rois.items():
    top, left, bottom, right = spec["roi"]
    y1 = int(top * height)
    y2 = int(bottom * height)
    x1 = int(left * width)
    x2 = int(right * width)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(out_dir / f"{name}.png"), crop)
