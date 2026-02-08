# integrator.py
"""
Módulo integrador: expone una API mínima para la GUI.

Retorna únicamente:
- label (bacteriana | normal | viral)
- probabilidad (float en %)
- heatmap (np.ndarray RGB) generado con Grad-CAM
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.neumonia_app.grad_cam import generate_grad_cam
from src.neumonia_app.load_model import get_model
from src.neumonia_app.preprocess_img import preprocess_image

CLASS_NAMES = ["bacteriana", "normal", "viral"]


def _first_input_name(model) -> str:
    """Obtiene el nombre del primer input del modelo (sin ':0')."""
    if hasattr(model, "input_names") and model.input_names:
        return str(model.input_names[0])
    name = getattr(model.inputs[0], "name", "input_1")
    return str(name).split(":")[0]


def _pack_input(model, batch: np.ndarray) -> dict:
    """Empaqueta el batch en dict para respetar la estructura esperada por Keras."""
    return {_first_input_name(model): batch}


def predict_from_array(
    array_bgr: np.ndarray,
    *,
    model_path: str | None = None,
    layer_name: str = "conv10_thisone",
) -> Tuple[str, float, np.ndarray]:
    """
    Ejecuta predicción + Grad-CAM sobre una imagen ya cargada (BGR).

    Args:
        array_bgr: imagen (H,W,3) en BGR.
        model_path: ruta al .h5 (opcional).
        layer_name: capa conv objetivo para Grad-CAM.

    Returns:
        (label, proba_pct, heatmap_rgb)
    """
    model = get_model(model_path=model_path)
    batch = preprocess_image(array_bgr)

    preds = model.predict(_pack_input(model, batch), verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    preds = np.asarray(preds)
    pred_idx = int(np.argmax(preds, axis=-1)[0])
    proba = float(preds[0, pred_idx]) * 100.0

    if 0 <= pred_idx < len(CLASS_NAMES):
        label = CLASS_NAMES[pred_idx]
    else:
        label = str(pred_idx)

    heatmap_rgb = generate_grad_cam(
        array_bgr,
        model=model,
        layer_name=layer_name,
    )
    return label, proba, heatmap_rgb
