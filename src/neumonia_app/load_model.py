# load_model.py
"""
Carga del modelo Keras (.h5) con cache.

- get_model(): retorna un tf.keras.Model, reutilizando cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import load_model

_MODEL: Optional[tf.keras.Model] = None


def _repo_root() -> Path:
    """
    Retorna la raíz del repo asumiendo estructura:
    <repo>/src/neumonia_app/load_model.py
    """
    return Path(__file__).resolve().parents[2]


def _clean_env_path(value: str) -> str:
    """Limpia comillas y espacios comunes en Windows."""
    return value.strip().strip('"').strip("'")


def get_model(*, model_path: str | None = None) -> tf.keras.Model:
    """
    Carga el modelo (.h5) una sola vez y lo reutiliza.

    Prioridad:
    1) model_path (argumento)
    2) NEUMONIA_MODEL_PATH (env var)
    3) <repo>/models/conv_MLP_84.h5
    5) <repo>/conv_MLP_84.h5 (fallback)
    """
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    root = _repo_root()

    candidates: list[Path] = []

    if model_path:
        candidates.append(Path(model_path))

    env_path = os.environ.get("NEUMONIA_MODEL_PATH")
    if env_path:
        candidates.append(Path(_clean_env_path(env_path)))

    candidates.extend(
        [
            root / "models" / "conv_MLP_84.h5",
            root / "conv_MLP_84.h5",
        ]
    )

    for p in candidates:
        try:
            p = p.expanduser()
        except Exception:
            pass

        if p.is_file():
            _MODEL = load_model(str(p), compile=False)
            return _MODEL

    checked = "\n".join([str(p) for p in candidates])
    raise FileNotFoundError(
        "No se encontró el modelo .h5. Ubícalo en el proyecto o define "
        "NEUMONIA_MODEL_PATH.\n\nRutas revisadas:\n"
        f"{checked}"
    )
