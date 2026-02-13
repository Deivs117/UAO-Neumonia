"""
Paquete neumonia_app (versión POO).

Expone las clases principales del pipeline:
- ReadGlobal: lectura de imágenes
- Preprocessor: preprocesamiento
- ModelLoader: carga del modelo
- GradCamService: predicción + Grad-CAM
- Integrator: orquestador completo del flujo
"""

from .read_img import ReadGlobal, ReadDICOM, ReadJPGJPEGPNG
from .preprocess_img import Preprocessor
from .load_model import ModelLoader
from .grad_cam import GradCamService
from .integrator import Integrator

__all__ = [
    "ReadGlobal",
    "ReadDICOM",
    "ReadJPGJPEGPNG",
    "Preprocessor",
    "ModelLoader",
    "GradCamService",
    "Integrator",
]
