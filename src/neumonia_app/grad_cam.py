from __future__ import annotations

"""
grad_cam.py

Servicio de inferencia y explicación visual (Grad-CAM) para modelos Keras.
Responsabilidad: predecir clase/probabilidad y generar un heatmap superpuesto.
"""

from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np
import tensorflow as tf

@dataclass(frozen=True)
class PredictionResult:
    """Resultado de predicción del modelo (DTO inmutable)."""
    label: str
    proba_pct: float
    class_index: int
    raw: Any


@dataclass
class GradCamService:
    """
    Servicio Grad-CAM (POO):
    - _pack_input(): adaptación de inputs para Keras
    - predict(): obtiene clase y probabilidad
    - grad_cam(): genera heatmap superpuesto
    - run(): pipeline predict + gradcam
    """

    layer_name: str = "conv10_thisone"
    class_names: Tuple[str, ...] = ("bacteriana", "normal", "viral")
    out_size: int = 512

    def _resolve_layer_name(self, model: tf.keras.Model) -> str:
        """
        Resuelve el nombre de la capa conv para Grad-CAM.
        - Si self.layer_name existe en el modelo -> úsala.
        - Si no existe -> usa la última capa convolucional compatible encontrada.
        """
        try:
            model.get_layer(self.layer_name)
            return self.layer_name
        except Exception:
            pass

        conv_types = (
            tf.keras.layers.Conv2D,
            tf.keras.layers.SeparableConv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.Conv3D,
        )

        for layer in reversed(model.layers):
            if isinstance(layer, conv_types):
                return layer.name

        # fallback: última capa con salida 4D (batch, h, w, c)
        for layer in reversed(model.layers):
            shape = getattr(layer, "output_shape", None)
            if isinstance(shape, tuple) and len(shape) == 4:
                return layer.name

        raise ValueError("No se encontró una capa convolucional compatible para Grad-CAM.")

    def _pack_input(self, model: tf.keras.Model, batch: np.ndarray):
        """
        Adapta el formato esperado del modelo para prevención de warnings desde keras.
        """
        if hasattr(model, "input_names") and model.input_names:
            name = model.input_names[0].split(":")[0]
        else:
            name = model.inputs[0].name.split(":")[0]
        return {name: batch}

    def predict(self, model: tf.keras.Model, batch: np.ndarray) -> PredictionResult:
        """
        Ejecuta la predicción del modelo.
        """
        preds = model.predict(self._pack_input(model, batch), verbose=0)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = np.asarray(preds)
        class_index = int(np.argmax(preds, axis=-1)[0])
        proba_pct = float(preds[0, class_index]) * 100.0

        label = (
            self.class_names[class_index]
            if 0 <= class_index < len(self.class_names)
            else str(class_index)
        )

        return PredictionResult(
            label=label,
            proba_pct=proba_pct,
            class_index=class_index,
            raw=preds,
        )

    def grad_cam(
        self,
        *,
        model: tf.keras.Model,
        batch: np.ndarray,
        array_bgr: np.ndarray,
        class_index: int,
    ) -> np.ndarray:
        """
        Genera heatmap Grad-CAM superpuesto. Retorna RGB (para GUI/PIL).
        """
        # En modelos con múltiples salidas, usa la primera
        out_tensor = model.outputs[0] if isinstance(model.outputs, (list, tuple)) else model.output

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(self._resolve_layer_name(model)).output, out_tensor],
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(self._pack_input(model, batch), training=False)
            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out[0]
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (self.out_size, self.out_size))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        base = cv2.resize(array_bgr, (self.out_size, self.out_size))
        superimposed = cv2.add((heatmap * 0.8).astype(np.uint8), base).astype(np.uint8)

        # BGR -> RGB
        return superimposed[:, :, ::-1]

    def run(
        self,
        *,
        model: tf.keras.Model,
        batch: np.ndarray,
        array_bgr: np.ndarray,
    ) -> Tuple[str, float, np.ndarray]:
        """
        Pipeline completo dentro de GradCam:
        Predict -> GradCam -> retorna (label, proba_pct, heatmap_rgb)
        """
        pred = self.predict(model, batch)
        heatmap_rgb = self.grad_cam(
            model=model,
            batch=batch,
            array_bgr=array_bgr,
            class_index=pred.class_index,
        )
        return pred.label, pred.proba_pct, heatmap_rgb