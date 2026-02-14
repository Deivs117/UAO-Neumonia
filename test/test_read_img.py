from __future__ import annotations

import numpy as np
import cv2
import pytest
from PIL import Image
import tempfile
import os

from neumonia_app.read_img import ReadJPGJPEGPNG


def test_read_jpgjpegpng_reads_image_correctly():
    """
    AAA Test explicado:

    A1 - Arrange:
        Creamos una imagen artificial conocida,
        la guardamos temporalmente en disco.
    A2 - Act:
        Llamamos al método read().
    A3 - Assert:
        Verificamos tipo, forma y consistencia.
    """

    # =========================
    # Arrange (Preparar)
    # =========================

    # Creamos imagen BGR conocida (color azul)
    img_bgr_original = np.zeros((100, 100, 3), dtype=np.uint8)
    img_bgr_original[:, :] = [255, 0, 0]  # Azul en BGR

    # Archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name

    cv2.imwrite(temp_path, img_bgr_original)

    reader = ReadJPGJPEGPNG()

    # =========================
    # Act (Ejecutar)
    # =========================
    array_bgr, img_pil = reader.read(temp_path)

    # =========================
    # Assert (Verificar)
    # =========================

    # 1. Tipo correcto
    assert isinstance(array_bgr, np.ndarray)
    assert isinstance(img_pil, Image.Image)

    # 2. Forma correcta
    assert array_bgr.shape == (100, 100, 3)

    # 3. Que la imagen cargada sea igual a la original
    np.testing.assert_array_equal(array_bgr, img_bgr_original)

    # Limpieza
    os.remove(temp_path)

