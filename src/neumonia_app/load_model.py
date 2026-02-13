from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf


@dataclass
class ModelLoader:
    """
    Carga el modelo desde archivo.

    Prioridad:
    1) Si model_path está definido -> usa esa ruta.
    2) Si no -> busca en ubicaciones típicas (Docker y repo local).
    """
    model_path: Optional[str] = None

    # Nombres típicos del archivo del modelo
    candidates: Tuple[str, ...] = (
        "conv_MLP_84.h5",
        "WilhemNet86.h5",
        "model.h5",
        "best_model.h5",
        "model.keras",
        "best_model.keras",
    )

    def _repo_root(self) -> Path:
        """
        Intenta inferir el root del repo:
        .../UAO-Neumonia/src/neumonia_app/load_model.py
                     ^^^^^^^^^^^ = repo root
        """
        here = Path(__file__).resolve()
        # load_model.py está en src/neumonia_app/, así que parents[2] suele ser repo root
        # (0=load_model.py, 1=neumonia_app, 2=src, 3=repo) -> ojo: en tu estructura real es parents[2] o parents[3]
        # Usamos heurística: buscamos carpeta "src" en parents y tomamos su parent como root.
        for p in here.parents:
            if p.name == "src":
                return p.parent
        # fallback: 3 niveles arriba
        return here.parents[3]

    def _resolve_path(self) -> Path:
        # 1) Si se especifica directamente
        if self.model_path:
            p = Path(self.model_path).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"Model path no existe: {p}")
            return p

        base = Path(__file__).resolve().parent
        repo = self._repo_root()

        # Docker estándar
        docker_models_dir = Path("/app/models")

        # 2) Directorios típicos a buscar (Docker + local)
        search_dirs = (
            # Docker (tu caso funcionando)
            docker_models_dir,
            Path("/app"),
            Path("/app/models"),
            Path("/app/model"),
            Path("/app/weights"),
            Path("/app/src"),
            Path("/app/src/models"),

            # Local repo (Windows/Linux/Mac)
            repo / "models",                 # ✅ recomendado: <repo>/models/conv_MLP_84.h5
            repo / "weights",
            repo / "src" / "models",
            repo / "src" / "neumonia_app" / "models",
            repo / "src" / "neumonia_app" / "weights",

            # Cerca del módulo (por si guardan el modelo dentro del paquete)
            base / "models",
            base / "weights",
            base.parent / "models",
            base.parent / "weights",
            base,
            base.parent,
        )

        for d in search_dirs:
            for name in self.candidates:
                p = d / name
                if p.exists():
                    return p

        raise FileNotFoundError(
            "No se encontró un archivo de modelo. "
            f"Probé: {self.candidates} en {', '.join(str(x) for x in search_dirs)}.\n\n"
            "Sugerencia: en local coloca el modelo en <repo>/models/conv_MLP_84.h5 "
            "o pasa model_path explícitamente."
        )

    def load(self) -> tf.keras.Model:
        """
        Retorna un tf.keras.Model listo para inferencia.
        Nota: compile=False evita errores de compatibilidad (ej: reduction='auto').
        """
        model_file = self._resolve_path()
        model = tf.keras.models.load_model(model_file, compile=False)
        return model
