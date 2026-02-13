from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.neumonia_app.model_io.keras_inputs import pack_input


@dataclass(frozen=True)
class PredictionResult:
    label: str
    proba_pct: float
    class_index: int
    raw: Any


@dataclass
class GradCamService:
    """
    Servicio Grad-CAM (POO):
    - predict(): obtiene clase y probabilidad
    - grad_cam(): genera heatmap superpuesto
    - run(): pipeline predict + gradcam
    """

    layer_name: str = "conv10_thisone"
    class_names: Tuple[str, ...] = ("bacteriana", "normal", "viral")
    out_size: int = 512

    def predict(self, model: tf.keras.Model, batch: np.ndarray) -> PredictionResult:
        """
        Ejecuta la predicción del modelo.
        """
        preds = model.predict(pack_input(model, batch), verbose=0)

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
            outputs=[model.get_layer(self.layer_name).output, out_tensor],
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(pack_input(model, batch), training=False)
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

