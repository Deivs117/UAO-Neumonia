# ui.py
"""
ui.py - Módulo de interfaz (GUI) para UAO-Neumonia.

OBJETIVO (2026-02-12):
- Re-diseño en 2 escenas (wizard):
  1) Registro del paciente + adjuntar radiografía
  2) Predicción + Grad-CAM + guardado (CSV/PDF)
- Mantener GUI responsive (re-render de imágenes al redimensionar)
- CSV tipo “base de datos” con encabezados y datos del paciente
- PDF tipo reporte clínico con datos del paciente + imágenes + resultado
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Optional, Dict, Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import csv
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

import numpy as np
from PIL import Image, ImageOps, ImageTk
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.neumonia_app.integrator import predict_from_array
from src.neumonia_app.read_img import load_image

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


# ===== GUI (paleta/estilo clínico) =====
BG_APP = "#F6F8FB"
FG_TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
CARD_BG = "#FFFFFF"
PANEL_BG = "#0B1220"

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_H1 = ("Segoe UI", 11, "bold")
FONT_BASE = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)


def apply_clinical_theme(root: tk.Tk) -> ttk.Style:
    root.configure(bg=BG_APP)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("TFrame", background=BG_APP)
    style.configure("TLabel", background=BG_APP, foreground=FG_TEXT, font=FONT_BASE)
    style.configure("Muted.TLabel", background=BG_APP, foreground=MUTED, font=FONT_SMALL)
    style.configure("Title.TLabel", background=BG_APP, foreground=FG_TEXT, font=FONT_TITLE)
    style.configure("H1.TLabel", background=BG_APP, foreground=FG_TEXT, font=FONT_H1)

    style.configure("TEntry", padding=(8, 6), font=FONT_BASE)
    style.configure("TCombobox", padding=(8, 6), font=FONT_BASE)
    style.configure("TButton", padding=(14, 8), font=FONT_BASE)
    style.configure("Primary.TButton", padding=(14, 8), font=("Segoe UI", 10, "bold"))
    style.configure("Danger.TButton", padding=(14, 8), font=("Segoe UI", 10, "bold"))

    # Cards
    style.configure("Card.TFrame", background=CARD_BG)
    style.configure("Card.TLabel", background=CARD_BG, foreground=FG_TEXT, font=FONT_BASE)
    style.configure("Card.Muted.TLabel", background=CARD_BG, foreground=MUTED, font=FONT_SMALL)
    style.configure("Card.H1.TLabel", background=CARD_BG, foreground=FG_TEXT, font=FONT_H1)

    return style


def _card(parent: ttk.Frame) -> tuple[tk.Frame, ttk.Frame]:
    outer = tk.Frame(parent, bg=BORDER, highlightthickness=0, bd=0)
    inner = ttk.Frame(outer, style="Card.TFrame", padding=12)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    return outer, inner


def fit_box(pil_img: Image.Image, box_w: int, box_h: int, fill: int = 0) -> Image.Image:
    box_w = max(1, int(box_w))
    box_h = max(1, int(box_h))

    if pil_img.mode not in ("L", "RGB"):
        pil_img = pil_img.convert("RGB")

    pil_fit = ImageOps.contain(pil_img, (box_w, box_h), RESAMPLE)

    if pil_fit.mode == "L":
        bg = Image.new(pil_fit.mode, (box_w, box_h), color=fill)
    else:
        bg = Image.new(pil_fit.mode, (box_w, box_h), color=(fill, fill, fill))

    x = (box_w - pil_fit.size[0]) // 2
    y = (box_h - pil_fit.size[1]) // 2
    bg.paste(pil_fit, (x, y))
    return bg


def pretty_label(label: str) -> str:
    l = (label or "").strip().lower()
    if l == "normal":
        return "Normal"
    if l == "viral":
        return "Neumonía viral"
    if l == "bacteriana":
        return "Neumonía bacteriana"
    return (label or "").strip()


def default_report_text(label: str) -> str:
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


def _safe_float(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def generate_pdf_report_clinical(
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

    # Header
    c.setLineWidth(1)
    c.setStrokeColorRGB(0.2, 0.2, 0.2)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "REPORTE DE APOYO AL DIAGNÓSTICO DE NEUMONÍA (IA)")
    y -= 0.55 * cm

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawString(margin, y, "Software: UAO-Neumonia · Radiografía de tórax + Grad-CAM")
    y -= 0.35 * cm

    c.setFillColorRGB(0, 0, 0)
    c.line(margin, y, page_w - margin, y)
    y -= 0.70 * cm

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Fecha/Hora: {now}")
    y -= 0.45 * cm

    # Datos del paciente (bloque compacto)
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

    # Caja resultado
    box_w = page_w - 2 * margin
    box_h = 2.0 * cm
    box_y = y - box_h
    c.setStrokeColorRGB(0.85, 0.85, 0.85)
    c.rect(margin, box_y, box_w, box_h, stroke=1, fill=0)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin + 0.35 * cm, y - 0.60 * cm, "Resultado del modelo")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 0.35 * cm, y - 1.30 * cm, f"Clase estimada: {pretty_label(label)}")
    c.drawString(margin + 9.0 * cm, y - 1.30 * cm, f"Probabilidad: {proba:.2f}%")
    y = box_y - 0.90 * cm

    # Imágenes
    usable_w = page_w - 2 * margin
    gap = 0.8 * cm
    img_w = (usable_w - gap) / 2.0
    img_h = 8.5 * cm

    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "Imagen original")
    c.drawString(margin + img_w + gap, y, "Grad-CAM (mapa de calor)")
    y -= 0.35 * cm

    img_y = y - img_h

    c.setStrokeColorRGB(0.85, 0.85, 0.85)
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

    y = img_y - 0.90 * cm

    # Observación
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "Observación automática")
    y -= 0.45 * cm

    c.setFont("Helvetica", 9.5)
    text = c.beginText(margin, y)
    for line in default_report_text(label).splitlines():
        text.textLine(line)
    c.drawText(text)

    # Pie
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawString(
        margin,
        1.2 * cm,
        "Este reporte es generado automáticamente para fines de apoyo/educación. No reemplaza evaluación clínica.",
    )

    c.showPage()
    c.save()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        apply_clinical_theme(self)

        self.title("Herramienta para la detección rápida de neumonía")
        self.geometry("1040x740")
        self.minsize(980, 680)

        # Estado de archivos / modelo
        self.output_dir: Optional[str] = None
        self.filepath: Optional[str] = None
        self.array_bgr: Optional[np.ndarray] = None
        self.original_pil: Optional[Image.Image] = None

        self.label: Optional[str] = None
        self.proba: Optional[float] = None
        self.heatmap_rgb: Optional[np.ndarray] = None

        # Patient vars (escena 1)
        self.p_name = tk.StringVar(value="")
        self.p_doc_type = tk.StringVar(value="CC")
        self.p_doc_num = tk.StringVar(value="")
        self.p_sex = tk.StringVar(value="")
        self.p_weight = tk.StringVar(value="")
        self.p_height = tk.StringVar(value="")
        self.p_age = tk.StringVar(value="")

        # Pred vars (escena 2)
        self.pred_var = tk.StringVar(value="")
        self.proba_var = tk.StringVar(value="")

        self.status_var = tk.StringVar(value="Listo. Registre los datos del paciente y adjunte una radiografía.")

        # Tk images
        self.tk_img_reg: Optional[ImageTk.PhotoImage] = None
        self.tk_img_orig: Optional[ImageTk.PhotoImage] = None
        self.tk_img_hm: Optional[ImageTk.PhotoImage] = None

        # Resize throttle jobs
        self._job_reg: Optional[str] = None
        self._job_orig: Optional[str] = None
        self._job_hm: Optional[str] = None

        # Menu
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Elegir carpeta de salida...", command=self.choose_output_dir)
        menubar.add_cascade(label="Archivo", menu=filemenu)
        self.config(menu=menubar)

        self._build()

        # Live validation
        for v in (self.p_name, self.p_doc_type, self.p_doc_num, self.p_sex, self.p_weight, self.p_height, self.p_age, self.p_sex):
            v.trace_add("write", lambda *_: self._update_wizard_buttons())
    
    def _build_scene_registration(self) -> None:
        s = self.scene_reg

        s.grid_rowconfigure(0, weight=1)
        s.grid_columnconfigure(0, weight=1, uniform="col")
        s.grid_columnconfigure(1, weight=1, uniform="col")

        # ===== LEFT CARD – DATOS PACIENTE =====
        card_l_outer, card_l = _card(s)
        card_l_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        card_l.grid_columnconfigure(1, weight=1)

        ttk.Label(card_l, text="Datos del paciente", style="Card.H1.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(card_l, text="Nombre").grid(row=1, column=0, sticky="w")
        self.entry_name = ttk.Entry(card_l, textvariable=self.p_name)
        self.entry_name.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="ID").grid(row=2, column=0, sticky="w")
        self.entry_doc = ttk.Entry(card_l, textvariable=self.p_doc_num)
        self.entry_doc.grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="Edad").grid(row=3, column=0, sticky="w")
        ttk.Entry(card_l, textvariable=self.p_age).grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="Altura (cm)").grid(row=4, column=0, sticky="w")
        ttk.Entry(card_l, textvariable=self.p_height).grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="Peso (kg)").grid(row=5, column=0, sticky="w")
        ttk.Entry(card_l, textvariable=self.p_weight).grid(row=5, column=1, sticky="ew", pady=4)
        
        ttk.Label(card_l, text="Sexo").grid(row=6, column=0, sticky="w")
        ttk.Entry(card_l, textvariable=self.p_sex).grid(row=6, column=1, sticky="ew", pady=4)

        # ===== RIGHT CARD – IMAGEN =====
        card_r_outer, card_r = _card(s)
        card_r_outer.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        card_r.grid_columnconfigure(0, weight=1)
        card_r.grid_rowconfigure(1, weight=1)

        ttk.Label(card_r, text="Radiografía de tórax", style="Card.H1.TLabel").grid(row=0, column=0, sticky="w")

        self.panel_reg = tk.Label(card_r, bd=0, bg=PANEL_BG, fg="white", text="Sin imagen")
        self.panel_reg.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.panel_reg.bind("<Configure>", self._on_reg_resize)

        actions = ttk.Frame(card_r, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        self.btn_load_img = ttk.Button(actions, text="Cargar imagen", command=self.load_img_file)
        self.btn_load_img.grid(row=0, column=0, sticky="w")

        self.lbl_img_name = ttk.Label(actions, text="Ningún archivo cargado.", style="Card.Muted.TLabel")
        self.lbl_img_name.grid(row=1, column=0, sticky="w", pady=(8, 0))

        self.entry_name.focus_set()


    # ===== Layout base =====

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ttk.Label(header, text="UAO-Neumonia", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Flujo: Registro del paciente → Predicción (IA) + Grad-CAM → Guardado (CSV/PDF)",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        ttk.Separator(root).grid(row=1, column=0, sticky="ew", pady=(12, 12))

        # Body (stacked scenes)
        self.body = ttk.Frame(root)
        self.body.grid(row=2, column=0, sticky="nsew")
        root.grid_rowconfigure(2, weight=1)
        self.body.grid_rowconfigure(0, weight=1)
        self.body.grid_columnconfigure(0, weight=1)

        self.scene_reg = ttk.Frame(self.body)
        self.scene_pred = ttk.Frame(self.body)

        for f in (self.scene_reg, self.scene_pred):
            f.grid(row=0, column=0, sticky="nsew")

        # Construcción de escenas
        self._build_scene_registration()
        self._build_scene_prediction()

        # Footer
        ttk.Separator(root).grid(row=3, column=0, sticky="ew", pady=(14, 10))

        footer = ttk.Frame(root)
        footer.grid(row=4, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=1)

        self.footer_left = ttk.Frame(footer)
        self.footer_left.grid(row=0, column=0, sticky="w")

        self.footer_right = ttk.Frame(footer)
        self.footer_right.grid(row=0, column=1, sticky="e")

        self.btn_back = ttk.Button(self.footer_left, text="← Volver", command=self.go_registration)
        self.btn_next = ttk.Button(self.footer_left, text="Siguiente →", style="Primary.TButton", command=self.go_prediction)

        self.btn_back.grid(row=0, column=0, padx=(0, 10))
        self.btn_next.grid(row=0, column=1)

        self.btn_predict = ttk.Button(self.footer_left, text="Predecir", style="Primary.TButton", command=self.run_model)
        self.btn_predict.grid(row=0, column=2, padx=(10, 0))
        self.btn_predict.state(["disabled"])

        self.btn_save_csv = ttk.Button(self.footer_right, text="Guardar CSV", command=self.save_results_csv)
        self.btn_save_pdf = ttk.Button(self.footer_right, text="Generar PDF", command=self.create_pdf)
        self.btn_clear = ttk.Button(self.footer_right, text="Borrar", command=self.delete, style="Danger.TButton")

        self.btn_save_csv.grid(row=0, column=0, padx=(0, 10))
        self.btn_save_pdf.grid(row=0, column=1, padx=(0, 10))
        self.btn_clear.grid(row=0, column=2)

        status = ttk.Label(root, textvariable=self.status_var, style="Muted.TLabel")
        status.grid(row=5, column=0, sticky="ew", pady=(10, 0))

        self.go_registration()


    def _build_scene_prediction(self) -> None:
        s = self.scene_pred

        s.grid_rowconfigure(0, weight=1)
        s.grid_columnconfigure(0, weight=1, uniform="col")
        s.grid_columnconfigure(1, weight=1, uniform="col")

        # Left: original
        card_l_outer, card_l = _card(s)
        card_l_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        card_l.grid_columnconfigure(0, weight=1)
        card_l.grid_rowconfigure(1, weight=1)

        ttk.Label(card_l, text="Imagen radiográfica", style="Card.H1.TLabel").grid(row=0, column=0, sticky="w")

        self.panel_orig = tk.Label(card_l, bd=0, bg=PANEL_BG, fg="white", text="Sin imagen")
        self.panel_orig.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.panel_orig.bind("<Configure>", self._on_orig_resize)

        # Right: heatmap + results
        card_r_outer, card_r = _card(s)
        card_r_outer.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        card_r.grid_columnconfigure(0, weight=1)
        card_r.grid_rowconfigure(1, weight=1)

        ttk.Label(card_r, text="Mapa de calor (Grad-CAM)", style="Card.H1.TLabel").grid(row=0, column=0, sticky="w")

        self.panel_hm = tk.Label(card_r, bd=0, bg=PANEL_BG, fg="white", text="Sin heatmap")
        self.panel_hm.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.panel_hm.bind("<Configure>", self._on_hm_resize)

        form = ttk.Frame(card_r, style="Card.TFrame")
        form.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        form.grid_columnconfigure(0, weight=1)

        ttk.Label(form, text="Resultado", style="Card.H1.TLabel").grid(row=0, column=0, sticky="w")
        self.pred_box = ttk.Entry(form, textvariable=self.pred_var, justify="center", state="readonly")
        self.pred_box.grid(row=1, column=0, sticky="ew", pady=(6, 10))

        ttk.Label(form, text="Probabilidad", style="Card.H1.TLabel").grid(row=2, column=0, sticky="w")
        self.proba_box = ttk.Entry(form, textvariable=self.proba_var, justify="center", state="readonly")
        self.proba_box.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        self.lbl_patient_summary = ttk.Label(form, text="", style="Card.Muted.TLabel")
        self.lbl_patient_summary.grid(row=4, column=0, sticky="w", pady=(10, 0))

    # ===== Navigation =====

    def go_registration(self) -> None:
        self.scene_reg.tkraise()
        self.btn_back.state(["disabled"])
        self.btn_predict.grid_remove()
        self.btn_save_csv.state(["disabled"])
        self.btn_save_pdf.state(["disabled"])

        self.btn_next.grid()
        self._update_wizard_buttons()

        self.status_var.set("Escena 1/2: registre los datos del paciente y adjunte una radiografía.")
        self.after_idle(self._render_all_panels)

    def go_prediction(self) -> None:
        if not self._can_go_next():
            showinfo(title="Siguiente", message="Complete los datos requeridos y cargue una radiografía.")
            return

        self.scene_pred.tkraise()
        self.btn_back.state(["!disabled"])

        # Toggle buttons
        self.btn_next.grid_remove()
        self.btn_predict.grid()
        self.btn_predict.state(["!disabled"] if self.array_bgr is not None else ["disabled"])

        # Save buttons enabled only after prediction
        self.btn_save_csv.state(["!disabled"] if (self.label is not None and self.proba is not None) else ["disabled"])
        self.btn_save_pdf.state(["!disabled"] if (self.label is not None and self.heatmap_rgb is not None) else ["disabled"])

        self._update_patient_summary()
        self.status_var.set("Escena 2/2: ejecute la predicción y luego guarde CSV/PDF.")
        self.after_idle(self._render_all_panels)

    # ===== Helpers =====

    def _patient_dict(self) -> Dict[str, Any]:
        return {
            "name": self.p_name.get().strip(),
            "doc_type": self.p_doc_type.get().strip(),
            "doc_num": self.p_doc_num.get().strip(),
            "sex": self.p_sex.get().strip(),
            "age": self.p_age.get().strip(),
            "height": self.p_height.get().strip(),
            "weight": self.p_weight.get().strip(),
        }

    def _update_patient_summary(self) -> None:
        p = self._patient_dict()
        summary = f"{p['name']} · {p['doc_type']} {p['doc_num']} · {p['sex']} · Edad: {p['age']} · Altura: {p['height']} · Peso: {p['weight']}"
        self.lbl_patient_summary.configure(text=summary.strip(" ·"))

    def _can_go_next(self) -> bool:
        p = self._patient_dict()
        if not p["name"]:
            return False
        if not p["doc_num"]:
            return False
        if self.original_pil is None or self.array_bgr is None:
            return False
        # Numeric sanity (optional but nice)
        if p["age"] and _safe_int(p["age"]) is None:
            return False
        if p["height"] and _safe_float(p["height"]) is None:
            return False
        if p["weight"] and _safe_float(p["weight"]) is None:
            return False
        return True

    def _update_wizard_buttons(self) -> None:
        if self.scene_reg.winfo_ismapped():
            self.btn_next.state(["!disabled"] if self._can_go_next() else ["disabled"])

    # ===== Responsive render (3 panels) =====

    def _render_panel_reg(self, w: int, h: int) -> None:
        if self.original_pil is None:
            return
        pil = fit_box(self.original_pil, w, h, fill=0)
        self.tk_img_reg = ImageTk.PhotoImage(pil)
        self.panel_reg.configure(image=self.tk_img_reg, text="")

    def _render_panel_orig(self, w: int, h: int) -> None:
        if self.original_pil is None:
            return
        pil = fit_box(self.original_pil, w, h, fill=0)
        self.tk_img_orig = ImageTk.PhotoImage(pil)
        self.panel_orig.configure(image=self.tk_img_orig, text="")

    def _render_panel_hm(self, w: int, h: int) -> None:
        if self.heatmap_rgb is None:
            return
        pil_heat = Image.fromarray(self.heatmap_rgb).convert("RGB")
        pil = fit_box(pil_heat, w, h, fill=0)
        self.tk_img_hm = ImageTk.PhotoImage(pil)
        self.panel_hm.configure(image=self.tk_img_hm, text="")

    def _on_reg_resize(self, event: tk.Event) -> None:
        if self._job_reg is not None:
            try:
                self.after_cancel(self._job_reg)
            except Exception:
                pass
        self._job_reg = self.after(90, lambda: self._safe_render_reg(event.width, event.height))

    def _on_orig_resize(self, event: tk.Event) -> None:
        if self._job_orig is not None:
            try:
                self.after_cancel(self._job_orig)
            except Exception:
                pass
        self._job_orig = self.after(90, lambda: self._safe_render_orig(event.width, event.height))

    def _on_hm_resize(self, event: tk.Event) -> None:
        if self._job_hm is not None:
            try:
                self.after_cancel(self._job_hm)
            except Exception:
                pass
        self._job_hm = self.after(90, lambda: self._safe_render_hm(event.width, event.height))

    def _safe_render_reg(self, w: int, h: int) -> None:
        self._job_reg = None
        if w > 40 and h > 40:
            self._render_panel_reg(w, h)

    def _safe_render_orig(self, w: int, h: int) -> None:
        self._job_orig = None
        if w > 40 and h > 40:
            self._render_panel_orig(w, h)

    def _safe_render_hm(self, w: int, h: int) -> None:
        self._job_hm = None
        if w > 40 and h > 40:
            self._render_panel_hm(w, h)

    def _render_all_panels(self) -> None:
        # Scene 1
        if hasattr(self, "panel_reg") and self.panel_reg.winfo_width() > 40 and self.panel_reg.winfo_height() > 40:
            self._render_panel_reg(self.panel_reg.winfo_width(), self.panel_reg.winfo_height())

        # Scene 2
        if hasattr(self, "panel_orig") and self.panel_orig.winfo_width() > 40 and self.panel_orig.winfo_height() > 40:
            self._render_panel_orig(self.panel_orig.winfo_width(), self.panel_orig.winfo_height())
        if hasattr(self, "panel_hm") and self.panel_hm.winfo_width() > 40 and self.panel_hm.winfo_height() > 40:
            self._render_panel_hm(self.panel_hm.winfo_width(), self.panel_hm.winfo_height())

    # ===== Actions =====

    def choose_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Seleccione carpeta para guardar PDFs y CSV")
        if not folder:
            return
        self.output_dir = folder
        self.status_var.set(f"Carpeta de salida seleccionada: {self.output_dir}")
        showinfo(title="Carpeta de salida", message=f"Carpeta seleccionada:\n{self.output_dir}")

    def ensure_output_dir(self) -> bool:
        if self.output_dir and os.path.isdir(self.output_dir):
            return True
        folder = filedialog.askdirectory(title="Seleccione carpeta para guardar PDFs y CSV")
        if not folder:
            showinfo(title="Carpeta de salida", message="No se seleccionó carpeta de salida.")
            return False
        self.output_dir = folder
        return True

    def load_img_file(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
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

        # Reset prediction
        self.label = None
        self.proba = None
        self.heatmap_rgb = None
        self.pred_var.set("")
        self.proba_var.set("")
        if hasattr(self, "panel_hm"):
            self.panel_hm.configure(image="", text="Sin heatmap")
        self.tk_img_hm = None

        # Update file label
        try:
            self.lbl_img_name.configure(text=os.path.basename(filepath))
        except Exception:
            pass

        # Render responsive
        if hasattr(self, "panel_reg"):
            self.panel_reg.configure(image="", text="")
        if hasattr(self, "panel_orig"):
            self.panel_orig.configure(image="", text="")
        self.after_idle(self._render_all_panels)

        self.status_var.set("Imagen cargada. Presione 'Siguiente' para continuar.")
        self._update_wizard_buttons()

        # Enable predict if already on scene 2
        if self.scene_pred.winfo_ismapped():
            self.btn_predict.state(["!disabled"])

    def run_model(self) -> None:
        if self.array_bgr is None:
            showinfo(title="Predecir", message="Primero cargue una imagen.")
            return

        # UX: bloquear mientras corre
        self.btn_predict.state(["disabled"])
        self.status_var.set("Ejecutando predicción...")
        self.update_idletasks()

        self.label, self.proba, self.heatmap_rgb = predict_from_array(self.array_bgr)

        self.pred_var.set(pretty_label(self.label or ""))
        self.proba_var.set(f"{self.proba:.2f}%" if self.proba is not None else "")

        # Render heatmap
        if hasattr(self, "panel_hm"):
            self.panel_hm.configure(image="", text="")
        self.after_idle(self._render_all_panels)

        self.status_var.set("Predicción completada.")

        # Enable save buttons
        self.btn_save_csv.state(["!disabled"] if (self.label is not None and self.proba is not None) else ["disabled"])
        self.btn_save_pdf.state(["!disabled"] if (self.label is not None and self.heatmap_rgb is not None) else ["disabled"])

        self.btn_predict.state(["!disabled"])

    def save_results_csv(self) -> None:
        p = self._patient_dict()

        if not p["doc_num"]:
            showinfo(title="Guardar", message="Ingresa el número de documento antes de guardar en CSV.")
            return
        if not p["name"]:
            showinfo(title="Guardar", message="Ingresa el nombre del paciente antes de guardar en CSV.")
            return
        if self.label is None or self.proba is None:
            showinfo(title="Guardar", message="Primero realiza una predicción.")
            return
        if not self.ensure_output_dir():
            return

        csv_path = os.path.join(self.output_dir, "historial.csv")
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
                    p["name"],
                    p["doc_type"],
                    p["doc_num"],
                    p["sex"],
                    p["age"],
                    p["height"],
                    p["weight"],
                    pretty_label(self.label),
                    f"{self.proba:.2f}",
                    os.path.basename(self.filepath or ""),
                ]
            )

        self.status_var.set(f"CSV guardado: {csv_path}")
        showinfo(title="Guardar", message=f"Datos guardados con éxito en:\n{csv_path}")

    def create_pdf(self) -> None:
        p = self._patient_dict()

        if not p["doc_num"] or not p["name"]:
            showinfo(title="PDF", message="Complete nombre y documento antes de generar el reporte.")
            return
        if self.label is None or self.proba is None or self.heatmap_rgb is None:
            showinfo(title="PDF", message="Primero presiona 'Predecir'.")
            return
        if self.original_pil is None:
            showinfo(title="PDF", message="No se encontró la imagen original para el reporte.")
            return
        if not self.ensure_output_dir():
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_doc = "".join(ch for ch in p["doc_num"] if ch.isalnum() or ch in ("-", "_"))
        pdf_path = os.path.join(self.output_dir, f"reporte_{safe_doc}_{ts}.pdf")

        generate_pdf_report_clinical(
            pdf_path=pdf_path,
            patient=p,
            label=self.label,
            proba=self.proba,
            original_pil=self.original_pil,
            heatmap_rgb=self.heatmap_rgb,
            source_filename=os.path.basename(self.filepath or ""),
        )

        self.status_var.set(f"PDF generado: {pdf_path}")
        showinfo(title="PDF", message=f"Reporte generado con éxito:\n{pdf_path}")

    def delete(self) -> None:
        answer = askokcancel(title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING)
        if not answer:
            return

        # Patient
        self.p_name.set("")
        self.p_doc_type.set("CC")
        self.p_doc_num.set("")
        self.p_sex.set("")
        self.p_weight.set("")
        self.p_height.set("")
        self.p_age.set("")

        # Image/model
        self.filepath = None
        self.array_bgr = None
        self.original_pil = None

        self.label = None
        self.proba = None
        self.heatmap_rgb = None

        self.pred_var.set("")
        self.proba_var.set("")

        # Panels
        if hasattr(self, "panel_reg"):
            self.panel_reg.configure(image="", text="Sin imagen")
        if hasattr(self, "panel_orig"):
            self.panel_orig.configure(image="", text="Sin imagen")
        if hasattr(self, "panel_hm"):
            self.panel_hm.configure(image="", text="Sin heatmap")

        self.tk_img_reg = None
        self.tk_img_orig = None
        self.tk_img_hm = None

        try:
            self.lbl_img_name.configure(text="Ningún archivo cargado.")
        except Exception:
            pass

        # Buttons
        self.btn_predict.state(["disabled"])
        self.btn_save_csv.state(["disabled"])
        self.btn_save_pdf.state(["disabled"])
        self.status_var.set("Listo. Registre los datos del paciente y adjunte una radiografía.")
        self._update_wizard_buttons()

        showinfo(title="Borrar", message="Los datos se borraron con éxito")

    # ===== Entry =====

def run() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    run()
