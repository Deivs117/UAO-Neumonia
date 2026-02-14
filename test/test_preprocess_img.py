from __future__ import annotations

import cv2
import numpy as np

from neumonia_app.preprocess_img import preprocess_image


def test_preprocess_image_known_matrix_expected_output_matches_pipeline():
    # Arrange (Preparar)
    # 1) Definimos una entrada totalmente conocida:
    #    una “imagen” BGR constante 512x512 con valor fijo v=64.
    v = 64
    img_bgr = np.full((512, 512, 3), v, dtype=np.uint8)

    # 2) Construimos la salida esperada (“expected”) aplicando el MISMO pipeline
    #    paso a paso: BGR->GRAY, CLAHE, normalización, expand dims.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply(gray)
    expected = (gray_eq.astype(np.float32) / 255.0)[None, ..., None]  # (1,512,512,1)

    # Act (Actuar)
    # Ejecutamos la función bajo prueba con la entrada conocida.
    out = preprocess_image(img_bgr)

    # Assert (Verificar)
    # Confirmamos que “out” cumple lo esperado:
    # - forma y tipo
    # - y que numéricamente coincide con expected.
    assert out.shape == (1, 512, 512, 1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-7)
