from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from src.neumonia_app.read_img import ReadGlobal
from src.neumonia_app.preprocess_img import Preprocessor
from src.neumonia_app.load_model import ModelLoader
from src.neumonia_app.grad_cam import GradCamService


@dataclass
class Integrator:
    reader: Optional[ReadGlobal] = None
    model_loader: Optional[ModelLoader] = None
    gradcam: Optional[GradCamService] = None
    preprocessor: Optional[Preprocessor] = None

    def __post_init__(self):
        if self.reader is None:
            self.reader = ReadGlobal()

        if self.model_loader is None:
            self.model_loader = ModelLoader()

        if self.gradcam is None:
            self.gradcam = GradCamService()

        if self.preprocessor is None:
            self.preprocessor = Preprocessor()

    def ReceiveAndPreprocessImage(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        array_bgr, _img_pil = self.reader.read(image_path)
        batch = self.preprocessor.preprocess(array_bgr)
        return batch, array_bgr

    def LoadModel(self) -> tf.keras.Model:
        return self.model_loader.load()

    def Run(self, image_path: str) -> Tuple[str, float, np.ndarray]:
        batch, array_bgr = self.ReceiveAndPreprocessImage(image_path)
        model = self.LoadModel()

        label, proba_pct, heatmap_rgb = self.gradcam.run(
            model=model,
            batch=batch,
            array_bgr=array_bgr,
        )
        return label, proba_pct, heatmap_rgb
