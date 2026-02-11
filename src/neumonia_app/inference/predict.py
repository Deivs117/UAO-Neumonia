from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Fuente de verdad de las clases 
CLASS_NAMES = ["bacteriana", "normal", "viral"]

@dataclass(frozen=True)
class PredictionResult:
    label: str
    proba_pct: float   # porcentaje 0..100
    raw: Any

def _first_input_name(model: Any) -> str:
    """Obtiene el nombre del primer input del modelo (sin ':0')."""
    if hasattr(model, "input_names") and model.input_names:
        return str(model.input_names[0]).split(":")[0]
    name = getattr(model.inputs[0], "name", "input_1")
    return str(name).split(":")[0]

def _pack_input(model: Any, batch: np.ndarray) -> dict:
    """Empaqueta el batch en dict para respetar la estructura esperada por Keras."""
    return {_first_input_name(model): batch}

def predict(model: Any, batch: np.ndarray) -> PredictionResult:
    # Debug opcional:
    print("DEBUG: predict() ejecutandose", flush=True)

    preds = model.predict(_pack_input(model, batch), verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    preds = np.asarray(preds)
    pred_idx = int(np.argmax(preds, axis=-1)[0])
    proba_pct = float(preds[0, pred_idx]) * 100.0

    label = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else str(pred_idx)

    return PredictionResult(label=label, proba_pct=proba_pct, raw=preds)

                   
