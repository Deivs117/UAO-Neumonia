# grad_cam.py
"""
Grad-CAM (TF2 / GradientTape).

- generate_grad_cam: recibe imagen BGR y modelo cargado, retorna heatmap RGB.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from src.neumonia_app.model_io.keras_inputs import pack_input

from src.neumonia_app.preprocess_img import preprocess_image

#-----------------------
def generate_grad_cam_from_batch(
    batch: np.ndarray,
    *,
    array_bgr: np.ndarray,
    model: tf.keras.Model,
    layer_name: str = "conv10_thisone",
) -> np.ndarray:
    """
    Genera un heatmap Grad-CAM superpuesto usando un batch ya preprocesado.

    Args:
        batch: tensor batch (1,512,512,1) ya listo para el modelo.
        array_bgr: imagen original (H,W,3) en BGR para superponer.
        model: modelo Keras ya cargado.
        layer_name: capa convolucional de interés.

    Returns:
        np.ndarray RGB con el heatmap superpuesto (512x512x3).
    """
    # En modelos con múltiples salidas, usa la primera
    out_tensor = model.outputs[0] if isinstance(model.outputs, (list, tuple)) else model.output

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, out_tensor],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(pack_input(model, batch), training=False)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    base = cv2.resize(array_bgr, (512, 512))
    superimposed = cv2.add((heatmap * 0.8).astype(np.uint8), base).astype(np.uint8)

    # Convertir BGR -> RGB para PIL/GUI
    return superimposed[:, :, ::-1]
