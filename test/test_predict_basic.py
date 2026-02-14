from __future__ import annotations

"""Tests unitarios para predicción básica (sin TensorFlow real)."""

import numpy as np

from src.neumonia_app.grad_cam import GradCamService, PredictionResult


class FakeTensor:
    """Tensor mínimo para simular `model.inputs[0].name` en Keras."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeModel:
    """Modelo mínimo con método `predict()` para simular Keras."""

    def __init__(self, preds: np.ndarray) -> None:
        self.inputs = [FakeTensor("input_1:0")]
        self._preds = preds

    def predict(self, packed_input, verbose: int = 0):
        """Retorna predicciones predefinidas."""
        return self._preds


def test_predict_basic() -> None:
    """Debe mapear la clase con mayor probabilidad y retornar PredictionResult."""
    # Arrange
    service = GradCamService(class_names=("bacteriana", "normal", "viral"))
    batch = np.zeros((1, 512, 512, 1), dtype=np.float32)
    preds = np.array([[0.10, 0.70, 0.20]], dtype=np.float32)
    model = FakeModel(preds)

    # Act
    result = service.predict(model, batch)

    # Assert
    assert isinstance(result, PredictionResult)
    assert result.class_index == 1
    assert result.label == "normal"
    assert abs(result.proba_pct - 70.0) < 1e-4
