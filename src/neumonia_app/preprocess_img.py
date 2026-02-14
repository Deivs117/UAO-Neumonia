from __future__ import annotations
"""
preprocess_img.py

Preprocesamiento de imágenes para inferencia:
- Resize
- Conversión a escala de grises
- CLAHE
- Normalización y expansión a batch
"""

from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class Preprocessor:
    """Preprocesador de imagen (BGR) a batch (1, H, W, 1) normalizado."""
    size: int = 512
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (4, 4)

    def preprocess(self, array_bgr: np.ndarray) -> np.ndarray:
        """Convierte un array BGR a batch listo para el modelo."""
        img = cv2.resize(array_bgr, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img = clahe.apply(img)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # (512,512,1)
        img = np.expand_dims(img, axis=0)   # (1,512,512,1)
        return img