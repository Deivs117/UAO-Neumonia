# preprocess_img.py
"""
Preprocesamiento para el modelo.

- Resize a 512x512
- Grises
- CLAHE
- Normalización [0,1]
- Batch: (1, 512, 512, 1)
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(array_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen BGR a batch tensor listo para el modelo.

    Args:
        array_bgr: (H,W,3) en BGR.

    Returns:
        np.ndarray shape (1, 512, 512, 1) float32.
    """
    img = cv2.resize(array_bgr, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # (512,512,1)
    img = np.expand_dims(img, axis=0)   # (1,512,512,1)
    return img
