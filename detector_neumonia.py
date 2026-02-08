#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detector de neumonía (GUI) con predicción y Grad-CAM.

Este archivo contiene una aplicación Tkinter que:
- Carga imágenes DICOM/JPG/PNG.
- Preprocesa y ejecuta un modelo Keras (.h5) para clasificar:
  bacteriana / normal / viral.
- Genera un heatmap Grad-CAM.
- Permite guardar resultados en CSV y generar un reporte PDF (no screenshot).
"""

from __future__ import annotations

# =========================
# Standard library imports
# =========================
import os
import warnings
import csv
from datetime import datetime
from typing import Optional, Tuple

# Logs/Warnings OFF antes de TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tkinter as tk
from tkinter import filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

# =========================
# Third-party imports
# =========================
import cv2
import numpy as np
import pydicom
import tensorflow as tf
from PIL import Image, ImageOps, ImageTk

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from tensorflow.keras.models import load_model

tf.get_logger().setLevel("ERROR")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# =========================
# Constants / Globals
# =========================
try:
    RESAMPLE = Image.Resampling.LANCZOS  # Pillow >= 9.1
except AttributeError:
    RESAMPLE = Image.LANCZOS  # Pillow older

CLASS_NAMES = ["bacteriana", "normal", "viral"]

# Cache para no recargar el modelo cada vez que presionas "Predecir"
_MODEL = None


def model_fun() -> tf.keras.Model:
    """
    Carga el modelo .h5 una sola vez y lo reutiliza.

    Busca primero en la variable de entorno NEUMONIA_MODEL_PATH y luego
    en nombres comunes dentro de la carpeta del proyecto.
    """
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    model_path = os.environ.get("NEUMONIA_MODEL_PATH")

    candidates = []
    if model_path:
        candidates.append(model_path)

    candidates += ["conv_MLP_84.h5", "WilhemNet86.h5"]

    for path in candidates:
        if path and os.path.exists(path):
            _MODEL = load_model(path, compile=False)
            return _MODEL

    raise FileNotFoundError(
        "No se encontró el modelo .h5. "
        "Pon el archivo en la misma carpeta del script o define "
        "NEUMONIA_MODEL_PATH con la ruta al .h5."
    )


def preprocess(array: np.ndarray) -> np.ndarray:
    """
    Preprocesa una imagen para el modelo:
    - Resize a 512x512.
    - Grises.
    - CLAHE.
    - Normalización [0,1].
    - Batch shape (1, 512, 512, 1).
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)

    array = array / 255.0
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


def grad_cam(array: np.ndarray, layer_name: str = "conv10_thisone") -> np.ndarray:
    """
    Genera un heatmap Grad-CAM usando GradientTape (TF2).

    Args:
        array: Imagen en formato numpy (BGR/RGB).
        layer_name: Nombre de la capa convolucional objetivo.

    Returns:
        Imagen RGB (numpy) con el heatmap superpuesto.
    """
    img = preprocess(array)
    model = model_fun()

    out_tensor = model.outputs[0] if isinstance(model.outputs, (list, tuple)) else model.output

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, out_tensor],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(pack_input(grad_model, img), training=False)
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

    img2 = cv2.resize(array, (512, 512))
    superimposed = cv2.add((heatmap * 0.8).astype(np.uint8), img2).astype(np.uint8)

    # return RGB for PIL
    return superimposed[:, :, ::-1]

def first_input_name(model) -> str:
    """Obtiene el nombre del primer input del modelo (sin ':0')."""
    if hasattr(model, "input_names") and model.input_names:
        return model.input_names[0]
    name = getattr(model.inputs[0], "name", "input_1")
    return name.split(":")[0]


def pack_input(model, batch):
    """Empaqueta batch según el nombre del input esperado."""
    return {first_input_name(model): batch}


def predict(array: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Ejecuta la predicción del modelo y genera el heatmap.

    Args:
        array: Imagen en formato numpy.

    Returns:
        label: (bacteriana|normal|viral)
        proba: probabilidad en porcentaje
        heatmap: imagen RGB con Grad-CAM
    """
    batch = preprocess(array)
    model = model_fun()

    preds = model.predict(pack_input(model, batch), verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    pred_idx = int(np.argmax(preds))
    proba = float(np.max(preds)) * 100.0
    label = CLASS_NAMES[pred_idx]

    heatmap = grad_cam(array)
    return label, proba, heatmap


def read_dicom_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Lee un archivo DICOM y devuelve:
    - Imagen numpy en RGB (para OpenCV/heatmap).
    - Imagen PIL (para visualización).
    """
    img = pydicom.dcmread(path)
    img_array = img.pixel_array

    img_pil = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)

    img_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_rgb, img_pil


def read_jpg_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Lee un archivo JPG/PNG y devuelve:
    - Imagen numpy (BGR) para el modelo/heatmap.
    - Imagen PIL para visualización.
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img_pil = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img_pil


def fit_square(pil_img: Image.Image, size: int = 250, fill: int = 0) -> Image.Image:
    """
    Ajusta una imagen al cuadro size x size manteniendo aspecto y con padding.
    """
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
    """
    Formatea el label para impresión amigable en reportes.
    """
    l = (label or "").strip().lower()
    if l == "normal":
        return "Normal"
    if l == "viral":
        return "Neumonía viral"
    if l == "bacteriana":
        return "Neumonía bacteriana"
    return label


def default_report_text(label: str) -> str:
    """
    Texto por defecto según clase predicha (bacteriana | normal | viral).
    """
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
    heatmap_np: np.ndarray,
) -> None:
    """
    Genera un PDF tipo reporte (no screenshot):
    - Cédula
    - Imagen original
    - Imagen con Grad-CAM
    - Predicción y probabilidad
    - Texto por defecto según clase
    """
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

    hm_pil = Image.fromarray(heatmap_np).convert("RGB")
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
    """
    Interfaz gráfica principal (Tkinter) para cargar imagen, predecir y generar reporte.
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("815x560")
        self.root.resizable(False, False)

        # Carpeta de salida para PDFs y CSV (se elige por el usuario)
        self.output_dir: Optional[str] = None

        # Menú para elegir carpeta sin mover la GUI
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(
            label="Elegir carpeta de salida...",
            command=self.choose_output_dir,
        )
        menubar.add_cascade(label="Archivo", menu=filemenu)
        self.root.config(menu=menubar)


        bold_font = font.Font(weight="bold")

        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=bold_font)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=bold_font)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=bold_font)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=bold_font)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=bold_font,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=bold_font)

        # Estado / datos
        self.array: Optional[np.ndarray] = None
        self.label: Optional[str] = None
        self.proba: Optional[float] = None
        self.heatmap: Optional[np.ndarray] = None

        self.filepath: Optional[str] = None
        self.original_pil: Optional[Image.Image] = None

        # Entrada: cédula
        self.patient_id_var = tk.StringVar(value="")
        self.text1 = ttk.Entry(self.root, textvariable=self.patient_id_var, width=10)

        # Resultado/probabilidad
        self.pred_var = tk.StringVar(value="")
        self.proba_var = tk.StringVar(value="")

        self.pred_box = ttk.Entry(
            self.root,
            textvariable=self.pred_var,
            width=12,
            justify="center",
            state="readonly",
        )
        self.proba_box = ttk.Entry(
            self.root,
            textvariable=self.proba_var,
            width=12,
            justify="center",
            state="readonly",
        )

        # Paneles de imagen
        self.img_panel1 = tk.Label(
            self.root,
            bd=1,
            relief="solid",
            bg="#111",
            fg="white",
            text="Sin imagen",
            compound="center",
        )
        self.img_panel2 = tk.Label(
            self.root,
            bd=1,
            relief="solid",
            bg="#111",
            fg="white",
            text="Sin heatmap",
            compound="center",
        )

        self.img1_tk: Optional[ImageTk.PhotoImage] = None
        self.img2_tk: Optional[ImageTk.PhotoImage] = None

        # Botones
        self.button_predict = ttk.Button(
            self.root,
            text="Predecir",
            state="disabled",
            command=self.run_model,
        )
        self.button_load = ttk.Button(
            self.root,
            text="Cargar Imagen",
            command=self.load_img_file,
        )
        self.button_delete = ttk.Button(
            self.root,
            text="Borrar",
            command=self.delete,
        )
        self.button_pdf = ttk.Button(
            self.root,
            text="PDF",
            command=self.create_pdf,
        )
        self.button_save = ttk.Button(
            self.root,
            text="Guardar",
            command=self.save_results_csv,
        )

        # Layout
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)

        self.button_load.place(x=70, y=460)
        self.button_predict.place(x=220, y=460)
        self.button_save.place(x=370, y=460)
        self.button_pdf.place(x=520, y=460)
        self.button_delete.place(x=670, y=460)

        self.text1.place(x=200, y=350)
        self.pred_box.place(x=610, y=350, width=90, height=30)
        self.proba_box.place(x=610, y=400, width=90, height=30)

        self.img_panel1.place(x=65, y=90, width=250, height=250)
        self.img_panel2.place(x=500, y=90, width=250, height=250)

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
        """
        Asegura que exista una carpeta de salida seleccionada.
        Si no existe, se solicita al usuario.
        """
        if self.output_dir and os.path.isdir(self.output_dir):
            return True

        folder = filedialog.askdirectory(title="Seleccione carpeta para guardar PDFs y CSV")
        if not folder:
            showinfo(title="Carpeta de salida", message="No se seleccionó carpeta de salida.")
            return False

        self.output_dir = folder
        return True


    def load_img_file(self) -> None:
        """
        Abre un diálogo para seleccionar una imagen, la carga y la muestra.
        Guarda filepath y una copia PIL original para el reporte PDF.
        """
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

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".dcm":
            self.array, img2show = read_dicom_file(filepath)
        else:
            self.array, img2show = read_jpg_file(filepath)

        self.original_pil = img2show.copy()

        # Limpiar resultados previos
        self.label = None
        self.proba = None
        self.heatmap = None
        self.pred_var.set("")
        self.proba_var.set("")
        self.img_panel2.configure(image="", text="Sin heatmap")
        self.img2_tk = None

        # Render izquierda
        pil_img = fit_square(img2show, 250, fill=0)
        self.img1_tk = ImageTk.PhotoImage(pil_img)
        self.img_panel1.configure(image=self.img1_tk, text="")

        self.button_predict["state"] = "enabled"

    def run_model(self) -> None:
        """
        Ejecuta predicción y actualiza GUI (resultado, probabilidad y heatmap).
        """
        if self.array is None:
            showinfo(title="Predecir", message="Primero carga una imagen.")
            return

        self.label, self.proba, self.heatmap = predict(self.array)

        pil_heat = Image.fromarray(self.heatmap).convert("RGB")
        pil_heat = fit_square(pil_heat, 250, fill=0)
        self.img2_tk = ImageTk.PhotoImage(pil_heat)
        self.img_panel2.configure(image=self.img2_tk, text="")

        self.pred_var.set(self.label)
        self.proba_var.set(f"{self.proba:.2f}%")

    def save_results_csv(self) -> None:
        """
        Guarda en historial.csv: cédula - label - probabilidad.
        Requiere cédula y predicción previa.
        """
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

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([cedula, self.label, f"{self.proba:.2f}%"])

        showinfo(title="Guardar", message=f"Datos guardados con éxito en:\n{csv_path}")


    def create_pdf(self) -> None:
        """
        Genera un reporte PDF con la cédula, imagen original, Grad-CAM y texto automático.
        """
        cedula = self.text1.get().strip()
        if not cedula:
            showinfo(title="PDF", message="Ingresa la cédula antes de generar el reporte.")
            return

        if self.label is None or self.proba is None or self.heatmap is None:
            showinfo(title="PDF", message="Primero carga una imagen y presiona 'Predecir'.")
            return

        if self.original_pil is None:
            # Backup: re-lee desde filepath si existe
            if not self.filepath:
                showinfo(title="PDF", message="No se encontró la imagen original para el reporte.")
                return

            ext = os.path.splitext(self.filepath)[1].lower()
            if ext == ".dcm":
                _, img2show = read_dicom_file(self.filepath)
            else:
                _, img2show = read_jpg_file(self.filepath)

            self.original_pil = img2show.copy()

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
            heatmap_np=self.heatmap,
        )

        showinfo(title="PDF", message=f"Reporte generado con éxito:\n{pdf_path}")

    def delete(self) -> None:
        """
        Limpia los datos actuales y reinicia el estado de la interfaz.
        """
        answer = askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=WARNING,
        )
        if not answer:
            return

        self.text1.delete(0, "end")

        self.array = None
        self.label = None
        self.proba = None
        self.heatmap = None
        self.filepath = None
        self.original_pil = None

        self.pred_var.set("")
        self.proba_var.set("")

        self.img_panel1.configure(image="", text="Sin imagen")
        self.img_panel2.configure(image="", text="Sin heatmap")
        self.img1_tk = None
        self.img2_tk = None

        self.button_predict["state"] = "disabled"
        showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main() -> int:
    """
    Punto de entrada de la aplicación.
    """
    App()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
