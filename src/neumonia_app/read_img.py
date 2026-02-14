from __future__ import annotations
"""
read_img.py

Lectura de imágenes desde disco (DICOM y formatos estándar) y retorno en:
- array BGR (np.ndarray) para pipeline/Grad-CAM
- imagen PIL para visualización UI
"""

import os
from typing import Tuple

import cv2
import numpy as np
import pydicom
from PIL import Image


class ReadDICOM:
    """Lee archivos .dcm y retorna (array_bgr, img_pil)."""

    def read(self, path: str) -> Tuple[np.ndarray, Image.Image]:
        """Lee un DICOM desde `path` y retorna (array_bgr, img_pil)."""
        ds = pydicom.dcmread(path)
        img_array = ds.pixel_array

        img_pil = Image.fromarray(img_array)

        # Normalización a 0..255
        img2 = img_array.astype(np.float32)
        maxv = float(np.max(img2)) if np.max(img2) != 0 else 1.0
        img2 = (np.maximum(img2, 0) / maxv) * 255.0
        img2 = img2.astype(np.uint8)

        # Convertir a 3 canales BGR
        array_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        return array_bgr, img_pil


class ReadJPGJPEGPNG:
    """Lee .jpg/.jpeg/.png y retorna (array_bgr, img_pil)."""

    def read(self, path: str) -> Tuple[np.ndarray, Image.Image]:
        """Lee una imagen estándar desde `path` y retorna (array_bgr, img_pil)."""
        array_bgr = cv2.imread(path)
        if array_bgr is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")

        img_rgb = cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        return array_bgr, img_pil


class ReadGlobal:
    """Selector de reader según extensión."""

    def __init__(self):
        """Inicializa readers internos para DICOM y formatos estándar."""
        self._dicom = ReadDICOM()
        self._std = ReadJPGJPEGPNG()

    def read(self, path: str) -> Tuple[np.ndarray, Image.Image]:
        """Despacha a un reader según la extensión del archivo."""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".dcm":
            return self._dicom.read(path)

        if ext in (".jpg", ".jpeg", ".png"):
            return self._std.read(path)

        raise ValueError(f"Extensión no soportada: {ext}")