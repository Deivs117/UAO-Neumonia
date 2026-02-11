from __future__ import annotations

from typing import Tuple
import numpy as np

from src.neumonia_app.inference.predict import predict as infer_predict
#from src.neumonia_app.grad_cam import generate_grad_cam
from src.neumonia_app.grad_cam import generate_grad_cam_from_batch

from src.neumonia_app.load_model import get_model
from src.neumonia_app.preprocess_img import preprocess_image

def predict_from_array(
    array_bgr: np.ndarray,
    *,
    model_path: str | None = None,
    layer_name: str = "conv10_thisone",
) -> Tuple[str, float, np.ndarray]:

    model = get_model(model_path=model_path)
    batch = preprocess_image(array_bgr)

    pred_res = infer_predict(model, batch)
    label = pred_res.label
    proba = float(pred_res.proba_pct)

    heatmap_rgb = generate_grad_cam_from_batch(
        batch,
        array_bgr=array_bgr,
        model=model,
        layer_name=layer_name,
    )
    return label, proba, heatmap_rgb
