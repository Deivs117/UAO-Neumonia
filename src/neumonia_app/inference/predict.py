from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from src.neumonia_app.model_io.keras_inputs import pack_input

import numpy as np

# Fuente de verdad de las clases 
CLASS_NAMES = ["bacteriana", "normal", "viral"]

@dataclass(frozen=True)
class PredictionResult:
    label: str
    proba_pct: float   # porcentaje 0..100
    raw: Any


def predict(model: Any, batch: np.ndarray) -> PredictionResult:
    # Debug opcional:
    print("DEBUG: predict() ejecutandose", flush=True)

    preds = model.predict(pack_input(model, batch), verbose=0)

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    preds = np.asarray(preds)
    pred_idx = int(np.argmax(preds, axis=-1)[0])
    proba_pct = float(preds[0, pred_idx]) * 100.0

    label = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else str(pred_idx)

    return PredictionResult(label=label, proba_pct=proba_pct, raw=preds)