# read_img.py
"""
Lectura de imágenes para la GUI.

- read_dicom_file: lee DICOM, retorna array BGR (para IA) y PIL (para visualizar).
- read_jpg_file: lee JPG/PNG, retorna array BGR (para IA) y PIL (para visualizar).
- load_image: wrapper por extensión.
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np
import pydicom
from PIL import Image


def read_dicom_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Lee un archivo DICOM.

    Returns:
        array_bgr: np.ndarray (H,W,3) BGR listo para preprocess.
        img_pil: PIL.Image para visualización (grises).
    """
    ds = pydicom.dcmread(path)
    img_array = ds.pixel_array

    # PIL para visualizar
    img_pil = Image.fromarray(img_array)

    # Normalización a 0..255
    img2 = img_array.astype(float)
    maxv = float(np.max(img2)) if np.max(img2) != 0 else 1.0
    img2 = (np.maximum(img2, 0) / maxv) * 255.0
    img2 = np.uint8(img2)

    # Convertir a 3 canales BGR (para preprocess consistentemente)
    array_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    return array_bgr, img_pil


def read_jpg_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Lee una imagen JPG/PNG.

    Returns:
        array_bgr: np.ndarray (H,W,3) BGR listo para preprocess.
        img_pil: PIL.Image para visualización.
    """
    array_bgr = cv2.imread(path)
    if array_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")

    # PIL para visualizar (RGB)
    img_rgb = cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return array_bgr, img_pil


def load_image(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Carga una imagen desde disco según extensión.

    Soporta: .dcm, .jpg, .jpeg, .png

    Returns:
        array_bgr, img_pil
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        return read_dicom_file(path)
    return read_jpg_file(path)