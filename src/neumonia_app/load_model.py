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

    def _repo_root(self) -> Path:
        """
        Intenta inferir el root del proyecto sin depender del nombre del repo.
        Heurística: subir directorios hasta encontrar un marcador típico.
        """
        here = Path(__file__).resolve()
        markers = (".git", "pyproject.toml", "requirements.txt")
        for p in here.parents:
            if any((p / m).exists() for m in markers):
                return p
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
            if not d.exists() or not d.is_dir():
                continue
            h5_files = list(d.glob("*.h5"))
            if not h5_files:
                continue
            if len(h5_files) == 1:
                return h5_files[0]
            h5_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return h5_files[0]

        raise FileNotFoundError(
            "No se encontró un archivo de modelo (.h5). "
            f"Busqué *.h5 en {', '.join(str(x) for x in search_dirs)}.\n\n"
            "Sugerencia: pasa model_path explícitamente si quieres fijar una ruta."
        )

    def load(self) -> tf.keras.Model:
        """
        Retorna un tf.keras.Model listo para inferencia.
        Nota: compile=False evita errores de compatibilidad (ej: reduction='auto').
        """
        model_file = self._resolve_path()
        model = tf.keras.models.load_model(model_file, compile=False)
        return model