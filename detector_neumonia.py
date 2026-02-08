#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry

import pydicom
from tkinter.messagebox import askokcancel, showinfo, WARNING
import getpass
from PIL import ImageTk, Image, ImageOps
try:
    RESAMPLE = Image.Resampling.LANCZOS  
except AttributeError:
    RESAMPLE = Image.LANCZOS            
import csv
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.experimental.output_all_intermediates(True)
import cv2

import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# Cache para no recargar el modelo cada vez que presionas "Predecir"
_MODEL = None

def model_fun():
    """
    Carga el modelo .h5 una sola vez y lo reutiliza.
    Busca primero una ruta por variable de entorno y luego nombres comunes.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # 1) Permite definir la ruta sin tocar el código:
    #    set NEUMONIA_MODEL_PATH=mi_modelo.h5   (Windows)
    model_path = os.environ.get("NEUMONIA_MODEL_PATH")

    # 2) Fallbacks si no se define env var
    candidates = []
    if model_path:
        candidates.append(model_path)

    # nombres que aparecen en tu repo/README
    candidates += ["conv_MLP_84.h5", "WilhemNet86.h5"]

    for p in candidates:
        if p and os.path.exists(p):
            _MODEL = load_model(p, compile=False)
            return _MODEL

    raise FileNotFoundError(
        "No se encontró el modelo .h5. "
        "Pon el archivo en la misma carpeta del script "
        "o define la variable de entorno NEUMONIA_MODEL_PATH con la ruta al .h5."
    )


def grad_cam(array, layer_name="conv10_thisone"):
    img = preprocess(array)
    model = model_fun()

    # Si el modelo tiene múltiples salidas, tomamos la primera (consistente con preds[0])
    out_tensor = model.outputs[0] if isinstance(model.outputs, (list, tuple)) else model.output

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, out_tensor],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img, training=False)
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
    superimposed_img = cv2.add((heatmap * 0.8).astype(np.uint8), img2).astype(np.uint8)

    return superimposed_img[:, :, ::-1]


def predict(array):
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = preprocess(array)
    #   2. call function to load model and predict: it returns predicted class and probability
    model = model_fun()
    # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    heatmap = grad_cam(array)
    return (label, proba, heatmap)


def read_dicom_file(path):
    img = pydicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array

def fit_square(pil_img: Image.Image, size: int = 250, fill: int = 0) -> Image.Image:
    """Ajusta una imagen al cuadro size x size manteniendo aspecto y con padding."""
    if pil_img.mode not in ("L", "RGB"):
        pil_img = pil_img.convert("RGB")

    # Contain mantiene el aspect ratio
    pil_fit = ImageOps.contain(pil_img, (size, size), RESAMPLE)

    # Crea fondo y centra
    bg = Image.new(pil_fit.mode, (size, size), color=fill if pil_fit.mode == "L" else (fill, fill, fill))
    x = (size - pil_fit.size[0]) // 2
    y = (size - pil_fit.size[1]) // 2
    bg.paste(pil_fit, (x, y))
    return bg

def generate_pdf_report(
    pdf_path: str,
    cedula: str,
    label: str,
    proba: float,
    original_pil,
    heatmap_np,
):
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

    # Header
    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, y, "REPORTE DE APOYO AL DIAGNÓSTICO DE NEUMONÍA (IA)")
    y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(margin, y, f"Fecha/Hora: {now}")
    y -= 0.55 * cm
    c.drawString(margin, y, f"Cédula: {cedula}")
    y -= 0.9 * cm

    # Resultado
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Resultado del modelo")
    y -= 0.6 * cm

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Clase estimada: {pretty_label(label)}")
    y -= 0.55 * cm
    c.drawString(margin, y, f"Probabilidad: {proba:.2f}%")
    y -= 1.0 * cm

    # Bloque de imágenes (2 columnas)
    # Caja disponible en ancho:
    usable_w = page_w - 2 * margin
    gap = 0.8 * cm
    box_w = (usable_w - gap) / 2.0
    box_h = 8.5 * cm

    # Títulos
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Imagen original")
    c.drawString(margin + box_w + gap, y, "Grad-CAM (mapa de calor)")
    y -= 0.4 * cm

    # Preparar imágenes
    # Original PIL
    orig = original_pil.convert("RGB")
    orig_reader = ImageReader(orig)

    # Heatmap numpy -> PIL
    from PIL import Image as PILImage  # import local para no ensuciar imports globales
    if heatmap_np.ndim == 2:
        hm_pil = PILImage.fromarray(heatmap_np).convert("RGB")
    else:
        hm_pil = PILImage.fromarray(heatmap_np).convert("RGB")
    hm_reader = ImageReader(hm_pil)

    # Dibujar imágenes “encajadas” en sus cajas
    img_y = y - box_h
    c.rect(margin, img_y, box_w, box_h, stroke=1, fill=0)
    c.rect(margin + box_w + gap, img_y, box_w, box_h, stroke=1, fill=0)

    c.drawImage(orig_reader, margin, img_y, width=box_w, height=box_h, preserveAspectRatio=True, anchor="c")
    c.drawImage(hm_reader, margin + box_w + gap, img_y, width=box_w, height=box_h, preserveAspectRatio=True, anchor="c")

    y = img_y - 1.0 * cm

    # Texto automático
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Observación automática")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    text = c.beginText(margin, y)
    for line in default_report_text(label).splitlines():
        text.textLine(line)
    c.drawText(text)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(
        margin,
        1.2 * cm,
        "Este reporte es generado automáticamente para fines de apoyo/educación. No reemplaza evaluación clínica.",
    )

    c.showPage()
    c.save()

def default_report_text(label: str) -> str:
    """Texto por defecto según clase predicha (bacteriana | normal | viral)."""
    l = (label or "").strip().lower()

    if l == "normal":
        return (
            "Resultado sugerido por el modelo: NORMAL (sin hallazgos compatibles con neumonía).\n\n"
            "Interpretación: La radiografía no presenta patrones típicos de neumonía según el modelo.\n\n"
            "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico. \n\n"
            "La interpretación final debe realizarla personal médico."
        )

    if l == "viral":
        return (
            "Resultado sugerido por el modelo: NEUMONÍA VIRAL.\n\n"
            "Interpretación: El modelo detecta patrones radiográficos compatibles con neumonía de origen viral.\n\n"
            "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico. \n\n"
            "La interpretación final debe realizarla personal médico."
        )

    if l == "bacteriana":
        return (
            "Resultado sugerido por el modelo: NEUMONÍA BACTERIANA.\n\n"
            "Interpretación: El modelo detecta patrones radiográficos compatibles con neumonía de origen bacteriano.\n\n"
            "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico. \n\n"
            "La interpretación final debe realizarla personal médico."
        )

    # Fallback por seguridad (si llega algo inesperado)
    return (
        f"Resultado sugerido por el modelo: {label}\n\n"
        "Nota: Este resultado es una estimación automatizada y NO constituye un diagnóstico. \n\n"
        "La interpretación final debe realizarla personal médico."
    )

def pretty_label(label: str) -> str:
    l = (label or "").strip().lower()
    if l == "normal":
        return "Normal"
    if l == "viral":
        return "Neumonía viral"
    if l == "bacteriana":
        return "Neumonía bacteriana"
    return label


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

       # Variables para mostrar resultado/probabilidad (sin insertar texto en Text widgets)
        self.pred_var = StringVar(value="")
        self.proba_var = StringVar(value="")

        # Paneles de imagen (en píxeles, no en caracteres)
        self.img_panel1 = Label(self.root, bd=1, relief="solid", bg="#111", fg="white", text="Sin imagen", compound="center")
        self.img_panel2 = Label(self.root, bd=1, relief="solid", bg="#111", fg="white", text="Sin heatmap", compound="center")

        # Caja de texto bonita para resultado y probabilidad (readonly)
        self.pred_box = ttk.Entry(self.root, textvariable=self.pred_var, width=12, justify="center", state="readonly")
        self.proba_box = ttk.Entry(self.root, textvariable=self.proba_var, width=12, justify="center", state="readonly")

        # Mantener referencias para que Tkinter no “pierda” las imágenes
        self.img1_tk = None
        self.img2_tk = None

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.pred_box.place(x=610, y=350, width=90, height=30)
        self.proba_box.place(x=610, y=400, width=90, height=30)

        self.img_panel1.place(x=65, y=90, width=250, height=250)
        self.img_panel2.place(x=500, y=90, width=250, height=250)


        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

        #   METHODS
    def load_img_file(self):
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

        # 1) Guardar ruta seleccionada (viene de cualquier carpeta, no importa)
        self.filepath = filepath

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".dcm":
            self.array, img2show = read_dicom_file(filepath)
        else:
            self.array, img2show = read_jpg_file(filepath)

        # 2) Guardar una COPIA de la imagen original para el PDF (sin resize)
        self.original_pil = img2show.copy()

        # Limpiar resultados anteriores
        self.pred_var.set("")
        self.proba_var.set("")
        self.img_panel2.configure(image="", text="Sin heatmap")
        self.img2_tk = None

        # Render imagen izquierda (sin deformar)
        pil_img = fit_square(img2show, 250, fill=0)
        self.img1_tk = ImageTk.PhotoImage(pil_img)
        self.img_panel1.configure(image=self.img1_tk, text="")

        self.button1["state"] = "enabled"


    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)

        # Heatmap (derecha)
        pil_heat = Image.fromarray(self.heatmap).convert("RGB")
        pil_heat = fit_square(pil_heat, 250, fill=0)
        self.img2_tk = ImageTk.PhotoImage(pil_heat)
        self.img_panel2.configure(image=self.img2_tk, text="")

        # Texto (sin acumular ni pisar imagen)
        self.pred_var.set(self.label)
        self.proba_var.set(f"{self.proba:.2f}%")


    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cedula = self.text1.get().strip()
        if not cedula:
            showinfo(title="PDF", message="Ingresa la cédula antes de generar el reporte.")
            return

        # Debe existir predicción y heatmap
        if getattr(self, "heatmap", None) is None or getattr(self, "label", None) is None:
            showinfo(title="PDF", message="Primero carga una imagen y presiona 'Predecir'.")
            return

        # Si no está en memoria, intenta reconstruirla desde la ruta seleccionada
        if getattr(self, "original_pil", None) is None:
            fp = getattr(self, "filepath", None)
            if not fp:
                showinfo(title="PDF", message="No se encontró la imagen original para el reporte.")
                return

            ext = os.path.splitext(fp)[1].lower()
            if ext == ".dcm":
                _, img2show = read_dicom_file(fp)
            else:
                _, img2show = read_jpg_file(fp)

            self.original_pil = img2show.copy()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"reporte_{cedula}_{ts}.pdf"

        generate_pdf_report(
            pdf_path=pdf_path,
            cedula=cedula,
            label=self.label,
            proba=self.proba,
            original_pil=self.original_pil,
            heatmap_np=self.heatmap,
        )

        showinfo(title="PDF", message=f"Reporte generado con éxito:\n{pdf_path}")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if not answer:
            return

        self.text1.delete(0, "end")
        self.pred_var.set("")
        self.proba_var.set("")
        self.array = None

        self.img_panel1.configure(image="", text="Sin imagen")
        self.img_panel2.configure(image="", text="Sin heatmap")
        self.img1_tk = None
        self.img2_tk = None

        self.button1["state"] = "disabled"
        showinfo(title="Borrar", message="Los datos se borraron con éxito")

def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
