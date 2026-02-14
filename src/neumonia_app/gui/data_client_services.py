from __future__ import annotations

from datetime import datetime
import csv
import os
from typing import Dict, Any, Callable, Optional

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .image_utils import pretty_label

class ReportService:
    CSV_FILENAME = "historial.csv"

    def ensure_output_dir(
        self,
        *,
        current_dir: Optional[str],
        ask_directory: Callable[[], str],
        on_cancel: Callable[..., None],
    ) -> Optional[str]:
        if current_dir and os.path.isdir(current_dir):
            return current_dir

        folder = ask_directory()
        if not folder:
            on_cancel(title="Carpeta de salida", message="No se seleccionó carpeta de salida.")
            return None

        os.makedirs(folder, exist_ok=True)
        return folder

    def _safe_doc_token(self, doc_num: str) -> str:
        token = "".join(ch for ch in (doc_num or "") if ch.isalnum() or ch in ("-", "_"))
        return token or "sin_documento"

    def _ensure_dir(self, output_dir: str) -> None:
        if not output_dir:
            raise ValueError("output_dir vacío.")
        os.makedirs(output_dir, exist_ok=True)

    def save_csv_history(
        self,
        *,
        output_dir: str,
        patient: Dict[str, Any],
        label: str,
        proba: float,
        source_filename: str = "",
    ) -> str:
        """
        Agrega una fila al CSV histórico (crea el archivo si no existe).
        Retorna la ruta del CSV.
        """
        self._ensure_dir(output_dir)

        csv_path = os.path.join(output_dir, self.CSV_FILENAME)
        new_file = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",")
            if new_file:
                writer.writerow(
                    [
                        "timestamp",
                        "nombre",
                        "tipo_doc",
                        "num_doc",
                        "sexo",
                        "edad",
                        "altura_cm",
                        "peso_kg",
                        "clase",
                        "probabilidad_pct",
                        "archivo_imagen",
                    ]
                )

            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    patient.get("name", ""),
                    patient.get("doc_type", ""),
                    patient.get("doc_num", ""),
                    patient.get("sex", ""),
                    patient.get("age", ""),
                    patient.get("height", ""),
                    patient.get("weight", ""),
                    pretty_label(label),
                    f"{float(proba):.2f}",
                    source_filename or "",
                ]
            )

        return csv_path

    def build_pdf_path(self, *, output_dir: str, doc_num: str) -> str:
        """
        Construye un nombre consistente para el PDF en output_dir.
        """
        self._ensure_dir(output_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_doc = self._safe_doc_token(doc_num)
        return os.path.join(output_dir, f"reporte_{safe_doc}_{ts}.pdf")

    def save_pdf_report(
        self,
        *,
        output_dir: str,
        patient: Dict[str, Any],
        label: str,
        proba: float,
        original_pil: Image.Image,
        heatmap_rgb: np.ndarray,
        source_filename: str = "",
    ) -> str:
        """
        Genera el PDF y retorna la ruta del archivo.
        """
        pdf_path = self.build_pdf_path(output_dir=output_dir, doc_num=str(patient.get("doc_num", "")))
        self.generate_pdf_report(
            pdf_path=pdf_path,
            patient=patient,
            label=label,
            proba=float(proba),
            original_pil=original_pil,
            heatmap_rgb=heatmap_rgb,
            source_filename=source_filename,
        )
        return pdf_path

    def _default_report_text(self, label: str) -> str:
        l = (label or "").strip().lower()

        if l == "normal":
            return (
                "Resultado sugerido por el modelo: NORMAL (sin hallazgos compatibles con neumonía).\n\n"
                "Interpretación: La radiografía no presenta patrones típicos de neumonía según el modelo.\n\n"
                "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico.\n"
                "La interpretación final debe realizarla personal médico."
            )
        if l == "viral":
            return (
                "Resultado sugerido por el modelo: NEUMONÍA VIRAL.\n\n"
                "Interpretación: El modelo detecta patrones radiográficos compatibles con neumonía de origen viral.\n\n"
                "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico.\n"
                "La interpretación final debe realizarla personal médico."
            )
        if l == "bacteriana":
            return (
                "Resultado sugerido por el modelo: NEUMONÍA BACTERIANA.\n\n"
                "Interpretación: El modelo detecta patrones radiográficos compatibles con neumonía de origen bacteriano.\n\n"
                "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico.\n"
                "La interpretación final debe realizarla personal médico."
            )

        return (
            f"Resultado sugerido por el modelo: {pretty_label(label)}\n\n"
            "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico.\n"
            "La interpretación final debe realizarla personal médico."
        )

    def generate_pdf_report(
        self,
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
        for line in self._default_report_text(label).splitlines():
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
