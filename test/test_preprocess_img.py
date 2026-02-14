from __future__ import annotations

"""Tests unitarios para el preprocesamiento de imágenes."""

import cv2
import numpy as np

from src.neumonia_app.preprocess_img import Preprocessor


def test_preprocess_image_known_matrix_expected_output_matches_pipeline() -> None:
    """Debe replicar el pipeline BGR->GRAY->CLAHE->norm->batch con entrada conocida."""
    # Arrange: imagen BGR constante 512x512 con valor fijo.
    value = 64
    img_bgr = np.full((512, 512, 3), value, dtype=np.uint8)
    preprocessor = Preprocessor()

    # Expected: aplicar el pipeline manualmente paso a paso.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply(gray)
    expected = (gray_eq.astype(np.float32) / 255.0)[None, ..., None]

    # Act
    out = preprocessor.preprocess(img_bgr)

    # Assert
    assert out.shape == (1, 512, 512, 1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-7)
