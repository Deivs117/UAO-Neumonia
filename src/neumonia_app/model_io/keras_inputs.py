from __future__ import annotations
from typing import Any
import numpy as np

def first_input_name(model: Any) -> str:
    """Obtiene el nombre del primer input del modelo (sin ':0')."""
    if hasattr(model, "input_names") and model.input_names:
        return str(model.input_names[0]).split(":")[0]
    name = getattr(model.inputs[0], "name", "input_1")
    return str(name).split(":")[0]

def pack_input(model: Any, batch: np.ndarray) -> dict:
    """Empaqueta el batch como dict para modelos Keras que esperan inputs nombrados."""
    return {first_input_name(model): batch}
