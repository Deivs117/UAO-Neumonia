from __future__ import annotations

"""Tests unitarios para lectura de imágenes estándar (PNG/JPG/JPEG)."""

import os
import tempfile

import cv2
import numpy as np
from PIL import Image

from src.neumonia_app.read_img import ReadJPGJPEGPNG


def test_read_jpgjpegpng_reads_image_correctly() -> None:
    """
    AAA test (Arrange, Act, Assert):

    - Arrange: crear una imagen BGR conocida y guardarla en disco temporal.
    - Act: llamar a `ReadJPGJPEGPNG.read()`.
    - Assert: verificar tipo, forma y equivalencia de pixeles.
    """
    # Arrange: imagen BGR conocida (azul en BGR).
    img_bgr_original = np.zeros((100, 100, 3), dtype=np.uint8)
    img_bgr_original[:, :] = [255, 0, 0]

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_path = tmp.name
    tmp.close()

    try:
        cv2.imwrite(temp_path, img_bgr_original)
        reader = ReadJPGJPEGPNG()

        # Act
        array_bgr, img_pil = reader.read(temp_path)

        # Assert: tipos
        assert isinstance(array_bgr, np.ndarray)
        assert isinstance(img_pil, Image.Image)

        # Assert: forma
        assert array_bgr.shape == (100, 100, 3)

        # Assert: contenido
        np.testing.assert_array_equal(array_bgr, img_bgr_original)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
