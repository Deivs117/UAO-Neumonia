from __future__ import annotations
"""
state.py

Modelos de estado (POO) para la aplicación:
- Patient: datos del paciente
- AppState: estado global de UI (imagen, resultados, rutas)
Incluye validadores safe_float y safe_int.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image


def safe_float(s: str) -> Optional[float]:
    """Convierte string a float seguro (acepta coma como separador decimal)."""
    s = (s or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def safe_int(s: str) -> Optional[int]:
    """Convierte string a int seguro."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


@dataclass
class Patient:
    """Entidad de paciente para el formulario y reportes."""
    name: str = ""
    doc_type: str = "CC"
    doc_num: str = ""
    sex: str = ""
    age: str = ""
    height: str = ""
    weight: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Retorna los datos del paciente como dict con strings normalizados."""
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
    """Estado de la aplicación: rutas, imagen cargada y resultados de inferencia."""
    output_dir: Optional[str] = None
    filepath: Optional[str] = None

    array_bgr: Optional[np.ndarray] = None
    original_pil: Optional[Image.Image] = None

    label: Optional[str] = None
    proba: Optional[float] = None
    heatmap_rgb: Optional[np.ndarray] = None

    def clear_prediction(self) -> None:
        """Limpia el resultado de predicción manteniendo la imagen cargada."""
        self.label = None
        self.proba = None
        self.heatmap_rgb = None
