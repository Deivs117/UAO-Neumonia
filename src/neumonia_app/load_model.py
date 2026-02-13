from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tensorflow as tf


@dataclass
class ModelLoader:
    """
    Carga el modelo desde archivo.
    - Si model_path está definido, usa esa ruta.
    - Si no, intenta con una lista de candidatos.
    """
    model_path: Optional[str] = None

    # Lista de candidatos (ajústala a tu repo)
    candidates: tuple[str, ...] = (
        "conv_MLP_84.h5",
        "WilhemNet86.h5",
        "model.h5",
        "best_model.h5",
        "model.keras",
        "best_model.keras",
    )

    def _resolve_path(self) -> Path:
        # 1) Si se especifica directamente
        if self.model_path:
            p = Path(self.model_path)
            if not p.exists():
                raise FileNotFoundError(f"Model path no existe: {p}")
            return p

        # 2) Buscar en el directorio del archivo (mismo folder que load_model.py)
        base = Path(__file__).resolve().parent
        # Ruta estándar dentro del contenedor Docker
        docker_models_dir = Path("/app/models")

        # 3) Probar candidatos en base y en posibles subcarpetas típicas
        search_dirs = (
            docker_models_dir, 
            base,
            base / "models",
            base.parent / "models",
            base.parent,

            Path("/app"),
            Path("/app/models"),
            Path("/app/model"),
            Path("/app/weights"),
            Path("/app/src"),
            Path("/app/src/models"),
        )

        for d in search_dirs:
            for name in self.candidates:
                p = d / name
                if p.exists():
                    return p

        raise FileNotFoundError(
            "No se encontró un archivo de modelo. "
            f"Probé: {self.candidates} en {', '.join(str(x) for x in search_dirs)}"
        )

    def load(self):
        """
        Retorna un tf.keras.Model listo para usar.
        """
        model_file = self._resolve_path()
        model = tf.keras.models.load_model(model_file, compile=False)
        return model
