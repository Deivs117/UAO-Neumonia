# main.py
"""
Entry-point de la aplicación.

GUI mínima usando los módulos:
- read_img.py
- integrator.py

Incluye:
- Guardado CSV (requiere cédula, y carpeta elegida)
- Reporte PDF (carpeta elegida)
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import csv
import tkinter as tk
from tkinter import filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

import numpy as np
from PIL import Image, ImageOps, ImageTk
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .integrator import predict_from_array
from .read_img import load_image

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def fit_square(pil_img: Image.Image, size: int = 250, fill: int = 0) -> Image.Image:
    """Ajusta una imagen al cuadro size x size manteniendo aspecto y con padding."""
    if pil_img.mode not in ("L", "RGB"):
        pil_img = pil_img.convert("RGB")

    pil_fit = ImageOps.contain(pil_img, (size, size), RESAMPLE)

    if pil_fit.mode == "L":
        bg = Image.new(pil_fit.mode, (size, size), color=fill)
    else:
        bg = Image.new(pil_fit.mode, (size, size), color=(fill, fill, fill))

    x = (size - pil_fit.size[0]) // 2
    y = (size - pil_fit.size[1]) // 2
    bg.paste(pil_fit, (x, y))
    return bg


def pretty_label(label: str) -> str:
    """Etiqueta amigable para el reporte."""
    l = (label or "").strip().lower()
    if l == "normal":
        return "Normal"
    if l == "viral":
        return "Neumonía viral"
    if l == "bacteriana":
        return "Neumonía bacteriana"
    return label


def default_report_text(label: str) -> str:
    """Texto por defecto según clase predicha (bacteriana | normal | viral)."""
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
        f"Resultado sugerido por el modelo: {label}\n\n"
        "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico.\n"
        "La interpretación final debe realizarla personal médico."
    )


def generate_pdf_report(
    pdf_path: str,
    cedula: str,
    label: str,
    proba: float,
    original_pil: Image.Image,
    heatmap_rgb: np.ndarray,
) -> None:
    """Genera un PDF tipo reporte (no screenshot)."""
    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_w, page_h = A4

    margin = 2.0 * cm
    y = page_h - margin

    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, y, "REPORTE DE APOYO AL DIAGNÓSTICO DE NEUMONÍA (IA)")
    y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(margin, y, f"Fecha/Hora: {now}")
    y -= 0.55 * cm
    c.drawString(margin, y, f"Cédula: {cedula}")
    y -= 0.9 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Resultado del modelo")
    y -= 0.6 * cm

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Clase estimada: {pretty_label(label)}")
    y -= 0.55 * cm
    c.drawString(margin, y, f"Probabilidad: {proba:.2f}%")
    y -= 1.0 * cm

    usable_w = page_w - 2 * margin
    gap = 0.8 * cm
    box_w = (usable_w - gap) / 2.0
    box_h = 8.5 * cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Imagen original")
    c.drawString(margin + box_w + gap, y, "Grad-CAM (mapa de calor)")
    y -= 0.4 * cm

    orig = original_pil.convert("RGB")
    orig_reader = ImageReader(orig)

    hm_pil = Image.fromarray(heatmap_rgb).convert("RGB")
    hm_reader = ImageReader(hm_pil)

    img_y = y - box_h
    c.rect(margin, img_y, box_w, box_h, stroke=1, fill=0)
    c.rect(margin + box_w + gap, img_y, box_w, box_h, stroke=1, fill=0)

    c.drawImage(
        orig_reader,
        margin,
        img_y,
        width=box_w,
        height=box_h,
        preserveAspectRatio=True,
        anchor="c",
    )
    c.drawImage(
        hm_reader,
        margin + box_w + gap,
        img_y,
        width=box_w,
        height=box_h,
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
    c.drawString(
        margin,
        1.2 * cm,
        "Este reporte es generado automáticamente para fines de apoyo/educación. "
        "No reemplaza evaluación clínica.",
    )

    c.showPage()
    c.save()


class App:
    """GUI principal (mínima) que consume integrator.py."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("815x560")
        self.root.resizable(False, False)

        self.output_dir: Optional[str] = None
        self.filepath: Optional[str] = None
        self.array_bgr: Optional[np.ndarray] = None
        self.original_pil: Optional[Image.Image] = None

        self.label: Optional[str] = None
        self.proba: Optional[float] = None
        self.heatmap_rgb: Optional[np.ndarray] = None

        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Elegir carpeta de salida...", command=self.choose_output_dir)
        menubar.add_cascade(label="Archivo", menu=filemenu)
        self.root.config(menu=menubar)

        bold_font = font.Font(weight="bold")

        ttk.Label(self.root, text="Imagen Radiográfica", font=bold_font).place(x=110, y=65)
        ttk.Label(self.root, text="Imagen con Heatmap", font=bold_font).place(x=545, y=65)
        ttk.Label(self.root, text="Resultado:", font=bold_font).place(x=500, y=350)
        ttk.Label(self.root, text="Cédula Paciente:", font=bold_font).place(x=65, y=350)
        ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA s",
            font=bold_font,
        ).place(x=122, y=25)
        ttk.Label(self.root, text="Probabilidad: ", font=bold_font).place(x=500, y=400)

        self.patient_id_var = tk.StringVar(value="")
        self.text1 = ttk.Entry(self.root, textvariable=self.patient_id_var, width=18)
        self.text1.place(x=200, y=350)

        self.pred_var = tk.StringVar(value="")
        self.proba_var = tk.StringVar(value="")
        self.pred_box = ttk.Entry(self.root, textvariable=self.pred_var, width=12, justify="center", state="readonly")
        self.proba_box = ttk.Entry(self.root, textvariable=self.proba_var, width=12, justify="center", state="readonly")
        self.pred_box.place(x=610, y=350, width=90, height=30)
        self.proba_box.place(x=610, y=400, width=90, height=30)

        self.img_panel1 = tk.Label(self.root, bd=1, relief="solid", bg="#111", fg="white", text="Sin imagen")
        self.img_panel2 = tk.Label(self.root, bd=1, relief="solid", bg="#111", fg="white", text="Sin heatmap")
        self.img_panel1.place(x=65, y=90, width=250, height=250)
        self.img_panel2.place(x=500, y=90, width=250, height=250)

        self.img1_tk: Optional[ImageTk.PhotoImage] = None
        self.img2_tk: Optional[ImageTk.PhotoImage] = None

        self.button_load = ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file)
        self.button_predict = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button_save = ttk.Button(self.root, text="Guardar", command=self.save_results_csv)
        self.button_pdf = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button_delete = ttk.Button(self.root, text="Borrar", command=self.delete)

        self.button_load.place(x=70, y=460)
        self.button_predict.place(x=220, y=460)
        self.button_save.place(x=370, y=460)
        self.button_pdf.place(x=520, y=460)
        self.button_delete.place(x=670, y=460)

        self.text1.focus_set()
        self.root.mainloop()

    def choose_output_dir(self) -> None:
        """Permite al usuario elegir la carpeta donde se guardan PDFs y CSV."""
        folder = filedialog.askdirectory(title="Seleccione carpeta para guardar PDFs y CSV")
        if not folder:
            return
        self.output_dir = folder
        showinfo(title="Carpeta de salida", message=f"Carpeta seleccionada:\n{self.output_dir}")

    def ensure_output_dir(self) -> bool:
        """Asegura que exista una carpeta de salida seleccionada."""
        if self.output_dir and os.path.isdir(self.output_dir):
            return True
        folder = filedialog.askdirectory(title="Seleccione carpeta para guardar PDFs y CSV")
        if not folder:
            showinfo(title="Carpeta de salida", message="No se seleccionó carpeta de salida.")
            return False
        self.output_dir = folder
        return True

    def load_img_file(self) -> None:
        """Carga una imagen desde disco (DICOM/JPG/PNG) y la muestra."""
        filepath = filedialog.askopenfilename(
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
                ("All files", "*.*"),
            ),
        )
        if not filepath:
            return

        self.filepath = filepath
        self.array_bgr, self.original_pil = load_image(filepath)

        self.label = None
        self.proba = None
        self.heatmap_rgb = None
        self.pred_var.set("")
        self.proba_var.set("")
        self.img_panel2.configure(image="", text="Sin heatmap")
        self.img2_tk = None

        pil_img = fit_square(self.original_pil, 250, fill=0)
        self.img1_tk = ImageTk.PhotoImage(pil_img)
        self.img_panel1.configure(image=self.img1_tk, text="")

        self.button_predict["state"] = "enabled"

    def run_model(self) -> None:
        """Ejecuta predicción y muestra heatmap."""
        if self.array_bgr is None:
            showinfo(title="Predecir", message="Primero carga una imagen.")
            return

        self.label, self.proba, self.heatmap_rgb = predict_from_array(self.array_bgr)

        pil_heat = Image.fromarray(self.heatmap_rgb).convert("RGB")
        pil_heat = fit_square(pil_heat, 250, fill=0)
        self.img2_tk = ImageTk.PhotoImage(pil_heat)
        self.img_panel2.configure(image=self.img2_tk, text="")

        self.pred_var.set(self.label)
        self.proba_var.set(f"{self.proba:.2f}%")

    def save_results_csv(self) -> None:
        """Guarda historial.csv (requiere cédula y predicción)."""
        cedula = self.text1.get().strip()
        if not cedula:
            showinfo(title="Guardar", message="Ingresa la cédula antes de guardar en CSV.")
            return
        if self.label is None or self.proba is None:
            showinfo(title="Guardar", message="Primero realiza una predicción.")
            return
        if not self.ensure_output_dir():
            return

        csv_path = os.path.join(self.output_dir, "historial.csv")
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="-")
            writer.writerow([cedula, self.label, f"{self.proba:.2f}%"])

        showinfo(title="Guardar", message=f"Datos guardados con éxito en:\n{csv_path}")

    def create_pdf(self) -> None:
        """Genera el reporte PDF (requiere cédula y predicción)."""
        cedula = self.text1.get().strip()
        if not cedula:
            showinfo(title="PDF", message="Ingresa la cédula antes de generar el reporte.")
            return
        if self.label is None or self.proba is None or self.heatmap_rgb is None:
            showinfo(title="PDF", message="Primero carga una imagen y presiona 'Predecir'.")
            return
        if self.original_pil is None:
            showinfo(title="PDF", message="No se encontró la imagen original para el reporte.")
            return
        if not self.ensure_output_dir():
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(self.output_dir, f"reporte_{cedula}_{ts}.pdf")

        generate_pdf_report(
            pdf_path=pdf_path,
            cedula=cedula,
            label=self.label,
            proba=self.proba,
            original_pil=self.original_pil,
            heatmap_rgb=self.heatmap_rgb,
        )

        showinfo(title="PDF", message=f"Reporte generado con éxito:\n{pdf_path}")

    def delete(self) -> None:
        """Limpia la GUI y el estado."""
        answer = askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=WARNING,
        )
        if not answer:
            return

        self.text1.delete(0, "end")
        self.filepath = None
        self.array_bgr = None
        self.original_pil = None

        self.label = None
        self.proba = None
        self.heatmap_rgb = None

        self.pred_var.set("")
        self.proba_var.set("")
        self.img_panel1.configure(image="", text="Sin imagen")
        self.img_panel2.configure(image="", text="Sin heatmap")
        self.img1_tk = None
        self.img2_tk = None

        self.button_predict["state"] = "disabled"
        showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main() -> int:
    """Punto de entrada."""
    App()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
