# load_model.py
"""
Carga del modelo Keras (.h5) con cache.

- get_model(): retorna un tf.keras.Model, reutilizando cache.
"""

from __future__ import annotations

import os
from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import load_model

_MODEL: Optional[tf.keras.Model] = None


def get_model(*, model_path: str | None = None) -> tf.keras.Model:
    """
    Carga el modelo (.h5) una sola vez y lo reutiliza.

    Prioridad:
    1) model_path (argumento)
    2) NEUMONIA_MODEL_PATH (env var)
    3) conv_MLP_84.h5
    4) WilhemNet86.h5
    """
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    candidates = []
    if model_path:
        candidates.append(model_path)

    env_path = os.environ.get("NEUMONIA_MODEL_PATH")
    if env_path:
        candidates.append(env_path)

    candidates += ["conv_MLP_84.h5", "WilhemNet86.h5"]

    for path in candidates:
        if path and os.path.exists(path):
            _MODEL = load_model(path, compile=False)
            return _MODEL

    raise FileNotFoundError(
        "No se encontró el modelo .h5. "
        "Ubícalo en el proyecto o define NEUMONIA_MODEL_PATH."
    )
