"""Utility script to convert hand-marked rectangles into ROI map entries.

Edit the `FIELDS` list with rectangles expressed in absolute pixels for the
aligned certificate (`TARGET_WIDTH` x `TARGET_HEIGHT`). Running the script
prints a JSON object that mirrors the structure expected in `roi_map.json`.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Dict, Iterable, List

TARGET_WIDTH = 560
TARGET_HEIGHT = 800


def _slugify(label: str) -> str:
    """Convert a human-readable label into a safe dictionary key."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "field"


def _normalise_rect(rect: Dict[str, float]) -> List[float]:
    """Transform absolute rectangle coordinates into normalised ROI values."""
    top = rect["yMin"] / TARGET_HEIGHT
    left = rect["xMin"] / TARGET_WIDTH
    bottom = (rect["yMin"] + rect["height"]) / TARGET_HEIGHT
    right = (rect["xMin"] + rect["width"]) / TARGET_WIDTH
    return [round(value, 6) for value in (top, left, bottom, right)]


FIELDS: Iterable[Dict[str, object]] = [
    {
        "label": "valabil pentru luna",
        "rect": {
            "xMin": 116.05995906631603,
            "yMin": 123.78057244261589,
            "width": 38.647384884060315,
            "height": 21.00401352394583,
        },
        "kind": "text",
        "description": "Valabil pentru luna (litere)",
    },
    {
        "label": "valabil pentru luna2",
        "rect": {
            "xMin": 155.5475044913343,
            "yMin": 121.2600908197424,
            "width": 72.67388679285256,
            "height": 23.104414876340407,
        },
        "kind": "text",
        "description": "Valabil pentru luna (camp vecin)",
    },
    {
        "label": "anu",
        "rect": {
            "xMin": 260.1474918405845,
            "yMin": 124.20065271309483,
            "width": 34.446582179271154,
            "height": 18.90361217155124,
        },
        "kind": "digits",
        "description": "Anul (numeric)",
    },
    {
        "label": "cod indemnizatie",
        "rect": {
            "xMin": 417.6775932701783,
            "yMin": 122.9404119016581,
            "width": 35.706822990707906,
            "height": 22.264254335382578,
        },
        "kind": "digits",
        "description": "Cod indemnizatie (1-17)",
    },
    {
        "label": "cnp",
        "rect": {
            "xMin": 154.54657421621874,
            "yMin": 176.99162230339778,
            "width": 178.8447814686262,
            "height": 18.283388440101195,
        },
        "kind": "digits",
        "description": "CNP titular",
    },
    {
        "label": "initial",
        "rect": {
            "xMin": 435.14039074241106,
            "yMin": 15.891519217659809,
            "width": 21.441147207173636,
            "height": 17.569828961433952,
        },
        "kind": "checkbox",
        "description": "Bifa certificat initial",
    },
    {
        "label": "in continuare",
        "rect": {
            "xMin": 435.7359781648325,
            "yMin": 34.05693560151522,
            "width": 22.036734629595127,
            "height": 16.97424153901246,
        },
        "kind": "checkbox",
        "description": "Bifa certificat in continuare",
    },
    {
        "label": "nrinreg",
        "rect": {
            "xMin": 80.64518225960146,
            "yMin": 269.0321276961701,
            "width": 72.64540762546643,
            "height": 18.646949550921864,
        },
        "kind": "text",
        "description": "Numar inregistrare",
    },
    {
        "label": "dataacordarii",
        "rect": {
            "xMin": 157.95232727279839,
            "yMin": 289.621467825313,
            "width": 81.96888240092736,
            "height": 20.977818244787098,
        },
        "kind": "date",
        "description": "Data acordarii",
    },
    {
        "label": "nrzile",
        "rect": {
            "xMin": 242.64055648323514,
            "yMin": 288.4560334783804,
            "width": 25.639555632517563,
            "height": 22.143252591719715,
        },
        "kind": "digits",
        "description": "Numar zile",
    },
    {
        "label": "dela",
        "rect": {
            "xMin": 271.7764151565506,
            "yMin": 289.23298970966886,
            "width": 82.35736051657156,
            "height": 19.812383897854485,
        },
        "kind": "date",
        "description": "De la (data inceput)",
    },
    {
        "label": "panala",
        "rect": {
            "xMin": 355.55070073616037,
            "yMin": 288.5051817290952,
            "width": 80.10967830176014,
            "height": 22.129745387226553,
        },
        "kind": "date",
        "description": "Pana la (data sfarsit)",
    },
    {
        "label": "coddiagnostic",
        "rect": {
            "xMin": 436.9881637611541,
            "yMin": 289.3903715445843,
            "width": 68.15961579265779,
            "height": 19.031581033014835,
        },
        "kind": "text",
        "description": "Cod diagnostic",
    },
    {
        "label": "adult",
        "rect": {
            "xMin": 506.56914098531405,
            "yMin": 288.8806860480151,
            "width": 14.90363852823505,
            "height": 18.93164894127155,
        },
        "kind": "checkbox",
        "description": "Bifa adult",
    },
    {
        "label": "cui",
        "rect": {
            "xMin": 30.458310164399816,
            "yMin": 394.81735991087515,
            "width": 177.63525921490967,
            "height": 16.51484269344965,
        },
        "kind": "text",
        "description": "CUI unitate",
    },
    {
        "label": "codparafa",
        "rect": {
            "xMin": 245.77399951060727,
            "yMin": 395.7211781989078,
            "width": 81.94841305650507,
            "height": 15.530546022805396,
        },
        "kind": "digits",
        "description": "Cod parafa medic",
    },
    {
        "label": "codparafamedicsef",
        "rect": {
            "xMin": 421.8970001522088,
            "yMin": 395.060303900065,
            "width": 83.27016165419064,
            "height": 16.521857471069573,
        },
        "kind": "digits",
        "description": "Cod parafa medic sef",
    },
    {
        "label": "urgentamedico",
        "rect": {
            "xMin": 298.0861094284918,
            "yMin": 17.171757992255774,
            "width": 41.972478126200166,
            "height": 14.917928972565122,
        },
        "kind": "digits",
        "description": "Cod urgenta medico-chirurgicala",
    },
    {
        "label": "cnpcopil",
        "rect": {
            "xMin": 234.87397526256862,
            "yMin": 196.69463767877525,
            "width": 181.31331986343955,
            "height": 15.76637564029909,
        },
        "kind": "digits",
        "description": "CNP copil/pacient",
    },
    {
        "label": "dataprimirii",
        "rect": {
            "xMin": 436.7963141539121,
            "yMin": 754.9710873825151,
            "width": 75.65093432621916,
            "height": 19.689969208194025,
        },
        "kind": "date",
        "description": "Data primirii",
    },
    {
        "label": "serieccmat",
        "rect": {
            "xMin": 444.56867042030444,
            "yMin": 102.61131808998141,
            "width": 81.3506622549069,
            "height": 21.244440461472504,
        },
        "kind": "text",
        "description": "Serie CCMAT",
    },
    {
        "label": "asiguratlacas",
        "rect": {
            "xMin": 156.99148856378642,
            "yMin": 146.13651318177872,
            "width": 175.13709453604162,
            "height": 15.544712532784757,
        },
        "kind": "text",
        "description": "Asigurat la CAS",
    },
    {
        "label": "casemitenta",
        "rect": {
            "xMin": 101.54868053018745,
            "yMin": 415.5781970833812,
            "width": 102.07694563195324,
            "height": 22.280754630324818,
        },
        "kind": "text",
        "description": "Casa emitenta",
    },
]


def build_roi_map(fields: Iterable[Dict[str, object]]) -> "OrderedDict[str, Dict[str, object]]":
    """Compose an ordered ROI map ready to dump as JSON."""
    roi_map: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    for field in fields:
        label = str(field["label"])
        key = str(field.get("key") or _slugify(label))
        rect = field["rect"]
        roi_map[key] = {
            "roi": _normalise_rect(rect),
            "kind": field.get("kind", "text"),
        }
        if "description" in field:
            roi_map[key]["description"] = field["description"]
    return roi_map


def main() -> None:
    roi_map = build_roi_map(FIELDS)
    print(json.dumps(roi_map, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
