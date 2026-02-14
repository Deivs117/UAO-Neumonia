from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.neumonia_app.integrator import Integrator
from src.neumonia_app.read_img import ReadGlobal

from .image_utils import pretty_label, default_report_text


class InferenceService:
    """
    Puente UI -> CORE POO
    - load_image(path): ReadGlobal.read
    - predict(path): Integrator.Run
    """
    def __init__(self) -> None:
        self.reader = ReadGlobal()
        self.integrator = Integrator()

    def load_image(self, filepath: str) -> Tuple[np.ndarray, Image.Image]:
        return self.reader.read(filepath)

    def predict(self, filepath: str) -> Tuple[str, float, np.ndarray]:
        return self.integrator.Run(filepath)


class ReportService:
    @staticmethod
    def generate_pdf_report(
        pdf_path: str,
        patient: Dict[str, Any],
        label: str,
        proba: float,
        original_pil: Image.Image,
        heatmap_rgb: np.ndarray,
        source_filename: str = "",
    ) -> None:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        page_w, page_h = A4
        margin = 2.0 * cm
        y = page_h - margin

        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "REPORTE DE APOYO AL DIAGNÓSTICO DE NEUMONÍA (IA)")
        y -= 0.55 * cm

        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.35, 0.35, 0.35)
        c.drawString(margin, y, "Software: UAO-Neumonia · Radiografía de tórax + Grad-CAM")
        c.setFillColorRGB(0, 0, 0)
        y -= 0.35 * cm

        c.line(margin, y, page_w - margin, y)
        y -= 0.70 * cm

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"Fecha/Hora: {now}")
        y -= 0.45 * cm

        name = patient.get("name", "")
        doc_type = patient.get("doc_type", "")
        doc_num = patient.get("doc_num", "")
        sex = patient.get("sex", "")
        age = patient.get("age", "")
        height = patient.get("height", "")
        weight = patient.get("weight", "")

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y, "Datos del paciente")
        y -= 0.40 * cm

        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"Nombre: {name}")
        y -= 0.42 * cm
        c.drawString(margin, y, f"Documento: {doc_type} {doc_num}".strip())
        y -= 0.42 * cm
        c.drawString(margin, y, f"Sexo: {sex}   Edad: {age}")
        y -= 0.42 * cm
        c.drawString(margin, y, f"Altura: {height}   Peso: {weight}")
        y -= 0.55 * cm

        if source_filename:
            c.setFont("Helvetica", 9.5)
            c.setFillColorRGB(0.35, 0.35, 0.35)
            c.drawString(margin, y, f"Archivo de imagen: {source_filename}")
            c.setFillColorRGB(0, 0, 0)
            y -= 0.55 * cm

        usable_w = page_w - 2 * margin
        gap = 0.8 * cm
        img_w = (usable_w - gap) / 2.0
        img_h = 8.5 * cm

        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Imagen original")
        c.drawString(margin + img_w + gap, y, "Grad-CAM (mapa de calor)")
        y -= 0.4 * cm

        img_y = y - img_h

        c.rect(margin, img_y, img_w, img_h, stroke=1, fill=0)
        c.rect(margin + img_w + gap, img_y, img_w, img_h, stroke=1, fill=0)

        orig = original_pil.convert("RGB")
        hm_pil = Image.fromarray(heatmap_rgb).convert("RGB")

        c.drawImage(ImageReader(orig), margin, img_y, width=img_w, height=img_h, preserveAspectRatio=True, anchor="c")
        c.drawImage(
            ImageReader(hm_pil),
            margin + img_w + gap,
            img_y,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            anchor="c",
        )

        y = img_y - 1.0 * cm

        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Observación automática")
        y -= 0.6 * cm

        c.setFont("Helvetica", 10)
        txt = c.beginText(margin, y)
        for line in default_report_text(label).splitlines():
            txt.textLine(line)
        c.drawText(txt)

        c.setFont("Helvetica-Oblique", 8)
        c.setFillColorRGB(0.35, 0.35, 0.35)
        c.drawString(
            margin,
            1.2 * cm,
            "Este reporte es generado automáticamente para fines de apoyo/educación. No reemplaza evaluación clínica.",
        )

        c.showPage()
        c.save()
