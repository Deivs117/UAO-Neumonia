from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

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
    _model: Optional[tf.keras.Model] = None

    def __post_init__(self):
        if self.reader is None:
            self.reader = ReadGlobal()

        if self.model_loader is None:
            self.model_loader = ModelLoader()

        if self.gradcam is None:
            self.gradcam = GradCamService()

        if self.preprocessor is None:
            self.preprocessor = Preprocessor()

    def LoadImage(self, image_path: str) -> Tuple[np.ndarray, Image.Image]:
        """
        Carga la imagen desde disco usando el reader configurado.
        Retorna (array_bgr, img_pil).
        """
        return self.reader.read(image_path)

    def ReceiveAndPreprocessImage(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        array_bgr, _ = self.LoadImage(image_path)
        batch = self.preprocessor.preprocess(array_bgr)
        return batch, array_bgr

    def LoadModel(self) -> tf.keras.Model:
        return self.model_loader.load()

    def Run(self, array_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Ejecuta inferencia + Grad-CAM partiendo de un array ya cargado (BGR),
        evitando re-leer el archivo desde disco.
        """
        batch = self.preprocessor.preprocess(array_bgr)
        model = self.LoadModel()

        label, proba_pct, heatmap_rgb = self.gradcam.run(
            model=model,
            batch=batch,
            array_bgr=array_bgr,
        )
        return label, proba_pct, heatmap_rgb