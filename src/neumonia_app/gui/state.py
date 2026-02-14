from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image


def safe_float(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def safe_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


@dataclass
class Patient:
    name: str = ""
    doc_type: str = "CC"
    doc_num: str = ""
    sex: str = ""
    age: str = ""
    height: str = ""
    weight: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name.strip(),
            "doc_type": self.doc_type.strip(),
            "doc_num": self.doc_num.strip(),
            "sex": self.sex.strip(),
            "age": self.age.strip(),
            "height": self.height.strip(),
            "weight": self.weight.strip(),
        }


@dataclass
class AppState:
    output_dir: Optional[str] = None
    filepath: Optional[str] = None

    array_bgr: Optional[np.ndarray] = None
    original_pil: Optional[Image.Image] = None

    label: Optional[str] = None
    proba: Optional[float] = None
    heatmap_rgb: Optional[np.ndarray] = None

    def clear_prediction(self) -> None:
        self.label = None
        self.proba = None
        self.heatmap_rgb = None
