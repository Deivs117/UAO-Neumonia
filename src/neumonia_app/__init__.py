"""
Paquete neumonia_app.

Contiene los módulos base para:
- lectura de imágenes (DICOM/JPG)
- preprocesamiento
- carga del modelo
- Grad-CAM
- integración para la GUI
"""

__all__ = [
    "integrator",
    "read_img",
    "preprocess_img",
    "load_model",
    "grad_cam",
]
