from __future__ import annotations

import os
import warnings
from typing import Optional, Dict, Any

import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

from PIL import Image, ImageTk

from .gui.theme import apply_clinical_theme, PANEL_BG, card
from .gui.image_utils import fit_box, pretty_label
from .gui.state import Patient, AppState, safe_float, safe_int
from .gui.data_client_services import ReportService
from .integrator import Integrator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        apply_clinical_theme(self)

        self.title("Herramienta para la detección rápida de neumonía")
        self.geometry("1040x740")
        self.minsize(980, 680)

        self.integrator = Integrator()
        self.reports = ReportService()

        self.state = AppState()
        self.patient = Patient()

        # vars (escena 1)
        self.p_name = tk.StringVar(value="")
        self.p_doc_num = tk.StringVar(value="")
        self.p_sex = tk.StringVar(value="")
        self.p_weight = tk.StringVar(value="")
        self.p_height = tk.StringVar(value="")
        self.p_age = tk.StringVar(value="")
        self.p_doc_type = tk.StringVar(value="CC")

        # vars (escena 2)
        self.pred_var = tk.StringVar(value="")
        self.proba_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Listo. Registre los datos del paciente y adjunte una radiografía.")

        # imágenes tk
        self.tk_img_reg: Optional[ImageTk.PhotoImage] = None
        self.tk_img_orig: Optional[ImageTk.PhotoImage] = None
        self.tk_img_hm: Optional[ImageTk.PhotoImage] = None

        # debounce jobs
        self._job_reg: Optional[str] = None
        self._job_orig: Optional[str] = None
        self._job_hm: Optional[str] = None

        # menu
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Elegir carpeta de salida...", command=self.choose_output_dir)
        menubar.add_cascade(label="Archivo", menu=filemenu)
        self.config(menu=menubar)

        self._build()

        for v in (self.p_name, self.p_doc_num, self.p_sex, self.p_weight, self.p_height, self.p_age):
            v.trace_add("write", lambda *_: self._update_wizard_buttons())

    # ===== Scenes =====

    def _build_scene_registration(self) -> None:
        s = self.scene_reg

        s.grid_rowconfigure(0, weight=1)
        s.grid_columnconfigure(0, weight=1, uniform="col")
        s.grid_columnconfigure(1, weight=1, uniform="col")

        card_l_outer, card_l = card(s)
        card_l_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        card_l.grid_columnconfigure(1, weight=1)

        ttk.Label(card_l, text="Datos del paciente", style="Card.H1.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(card_l, text="Nombre").grid(row=1, column=0, sticky="w")
        self.entry_name = ttk.Entry(card_l, textvariable=self.p_name)
        self.entry_name.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="Tipo de documento").grid(row=2, column=0, sticky="w")
        combo = ttk.Combobox(card_l, textvariable=self.p_doc_type, values=("CC", "TI", "CE", "PA"), state="readonly", width=6)
        combo.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(card_l, text="Número de documento").grid(row=3, column=0, sticky="w")
        self.entry_doc = ttk.Entry(card_l, textvariable=self.p_doc_num)
        self.entry_doc.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="Edad").grid(row=4, column=0, sticky="w")
        ttk.Entry(card_l, textvariable=self.p_age).grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="Altura (cm)").grid(row=5, column=0, sticky="w")
        ttk.Entry(card_l, textvariable=self.p_height).grid(row=5, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="Peso (kg)").grid(row=6, column=0, sticky="w")
        ttk.Entry(card_l, textvariable=self.p_weight).grid(row=6, column=1, sticky="ew", pady=4)

        ttk.Label(card_l, text="Sexo").grid(row=7, column=0, sticky="w")
        ttk.Entry(card_l, textvariable=self.p_sex).grid(row=7, column=1, sticky="ew", pady=4)

        card_r_outer, card_r = card(s)
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

    def _build_scene_prediction(self) -> None:
        s = self.scene_pred

        s.grid_rowconfigure(0, weight=1)
        s.grid_columnconfigure(0, weight=1, uniform="col")
        s.grid_columnconfigure(1, weight=1, uniform="col")

        card_l_outer, card_l = card(s)
        card_l_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        card_l.grid_columnconfigure(0, weight=1)
        card_l.grid_rowconfigure(1, weight=1)

        ttk.Label(card_l, text="Imagen radiográfica", style="Card.H1.TLabel").grid(row=0, column=0, sticky="w")

        self.panel_orig = tk.Label(card_l, bd=0, bg=PANEL_BG, fg="white", text="Sin imagen")
        self.panel_orig.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.panel_orig.bind("<Configure>", self._on_orig_resize)

        card_r_outer, card_r = card(s)
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
        ttk.Entry(form, textvariable=self.pred_var, justify="center", state="readonly").grid(row=1, column=0, sticky="ew", pady=(6, 10))

        ttk.Label(form, text="Probabilidad", style="Card.H1.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.proba_var, justify="center", state="readonly").grid(row=3, column=0, sticky="ew", pady=(6, 0))

        self.lbl_patient_summary = ttk.Label(form, text="", style="Card.Muted.TLabel")
        self.lbl_patient_summary.grid(row=4, column=0, sticky="w", pady=(10, 0))

    # ===== Layout base =====

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

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

        self.body = ttk.Frame(root)
        self.body.grid(row=2, column=0, sticky="nsew")
        root.grid_rowconfigure(2, weight=1)
        self.body.grid_rowconfigure(0, weight=1)
        self.body.grid_columnconfigure(0, weight=1)

        self.scene_reg = ttk.Frame(self.body)
        self.scene_pred = ttk.Frame(self.body)

        for f in (self.scene_reg, self.scene_pred):
            f.grid(row=0, column=0, sticky="nsew")

        self._build_scene_registration()
        self._build_scene_prediction()

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
        self.btn_predict = ttk.Button(self.footer_left, text="Predecir", style="Primary.TButton", command=self.run_model)

        self.btn_back.grid(row=0, column=0, padx=(0, 10))
        self.btn_next.grid(row=0, column=1)
        self.btn_predict.grid(row=0, column=2, padx=(10, 0))
        self.btn_predict.state(["disabled"])

        self.btn_save_csv = ttk.Button(self.footer_right, text="Guardar CSV", command=self.save_results_csv)
        self.btn_save_pdf = ttk.Button(self.footer_right, text="Generar PDF", command=self.create_pdf)
        self.btn_clear = ttk.Button(self.footer_right, text="Borrar", command=self.delete, style="Danger.TButton")

        self.btn_save_csv.grid(row=0, column=0, padx=(0, 10))
        self.btn_save_pdf.grid(row=0, column=1, padx=(0, 10))
        self.btn_clear.grid(row=0, column=2)

        self.btn_save_csv.state(["disabled"])
        self.btn_save_pdf.state(["disabled"])

        status = ttk.Label(root, textvariable=self.status_var, style="Muted.TLabel")
        status.grid(row=5, column=0, sticky="ew", pady=(10, 0))

        self.go_registration()

    # ===== Navigation =====

    def go_registration(self) -> None:
        self.scene_reg.tkraise()
        self.btn_back.state(["disabled"])

        self.btn_predict.state(["disabled"])
        self.btn_save_csv.state(["disabled"])
        self.btn_save_pdf.state(["disabled"])

        self.btn_next.grid()
        self._update_wizard_buttons()

        self.status_var.set("Registre los datos del paciente y adjunte una radiografía.")
        self.after_idle(self._render_all_panels)

    def go_prediction(self) -> None:
        if not self._can_go_next():
            showinfo(title="Siguiente", message="Complete los datos requeridos y cargue una radiografía.")
            return

        self.scene_pred.tkraise()
        self.btn_back.state(["!disabled"])

        self.btn_next.grid_remove()
        self.btn_predict.state(["!disabled"] if self.state.filepath else ["disabled"])

        self._update_patient_summary()
        self.status_var.set("Ejecute la predicción y luego guarde CSV/PDF.")
        self.after_idle(self._render_all_panels)

    # ===== Helpers =====

    def _patient_dict(self) -> Dict[str, Any]:
        self.patient = Patient(
            name=self.p_name.get(),
            doc_type=self.p_doc_type.get(),
            doc_num=self.p_doc_num.get(),
            sex=self.p_sex.get(),
            age=self.p_age.get(),
            height=self.p_height.get(),
            weight=self.p_weight.get(),
        )
        return self.patient.as_dict()

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
        if self.state.original_pil is None or self.state.array_bgr is None:
            return False
        if p["age"] and safe_int(p["age"]) is None:
            return False
        if p["height"] and safe_float(p["height"]) is None:
            return False
        if p["weight"] and safe_float(p["weight"]) is None:
            return False
        return True

    def _update_wizard_buttons(self) -> None:
        if self.scene_reg.winfo_ismapped():
            self.btn_next.state(["!disabled"] if self._can_go_next() else ["disabled"])

    # ===== Responsive render =====

    def _render_panel_reg(self, w: int, h: int) -> None:
        if self.state.original_pil is None:
            return
        pil = fit_box(self.state.original_pil, w, h, fill=0)
        self.tk_img_reg = ImageTk.PhotoImage(pil)
        self.panel_reg.configure(image=self.tk_img_reg, text="")

    def _render_panel_orig(self, w: int, h: int) -> None:
        if self.state.original_pil is None:
            return
        pil = fit_box(self.state.original_pil, w, h, fill=0)
        self.tk_img_orig = ImageTk.PhotoImage(pil)
        self.panel_orig.configure(image=self.tk_img_orig, text="")

    def _render_panel_hm(self, w: int, h: int) -> None:
        if self.state.heatmap_rgb is None:
            return
        pil_heat = Image.fromarray(self.state.heatmap_rgb).convert("RGB")
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
        if hasattr(self, "panel_reg") and self.panel_reg.winfo_width() > 40 and self.panel_reg.winfo_height() > 40:
            self._render_panel_reg(self.panel_reg.winfo_width(), self.panel_reg.winfo_height())

        if hasattr(self, "panel_orig") and self.panel_orig.winfo_width() > 40 and self.panel_orig.winfo_height() > 40:
            self._render_panel_orig(self.panel_orig.winfo_width(), self.panel_orig.winfo_height())

        if hasattr(self, "panel_hm") and self.panel_hm.winfo_width() > 40 and self.panel_hm.winfo_height() > 40:
            self._render_panel_hm(self.panel_hm.winfo_width(), self.panel_hm.winfo_height())

    # ===== Actions =====

    def choose_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Seleccione carpeta para guardar PDFs y CSV")
        if not folder:
            return
        self.state.output_dir = folder
        self.status_var.set(f"Carpeta de salida seleccionada: {self.state.output_dir}")
        showinfo(title="Carpeta de salida", message=f"Carpeta seleccionada:\n{self.state.output_dir}")

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

        self.state.filepath = filepath

        self.state.array_bgr, self.state.original_pil = self.integrator.LoadImage(filepath)
        self.state.clear_prediction()

        self.pred_var.set("")
        self.proba_var.set("")
        self.btn_save_csv.state(["disabled"])
        self.btn_save_pdf.state(["disabled"])

        self.lbl_img_name.configure(text=os.path.basename(filepath))
        self.status_var.set("Imagen cargada. Presione 'Siguiente' para continuar.")
        self.after_idle(self._render_all_panels)
        self._update_wizard_buttons()

    def run_model(self) -> None:
        if not self.state.filepath:
            showinfo(title="Predecir", message="Primero cargue una imagen.")
            return

        self.btn_predict.state(["disabled"])
        self.status_var.set("Ejecutando predicción...")
        self.update_idletasks()

        if self.state.array_bgr is None:
            showinfo(title="Predecir", message="No se encontró la imagen cargada en memoria.")
            self.status_var.set("No se pudo predecir: imagen no disponible en memoria.")
            self.btn_predict.state(["!disabled"])
            return
        label, proba, heatmap_rgb = self.integrator.Run(self.state.array_bgr)

        self.state.label = label
        self.state.proba = float(proba)
        self.state.heatmap_rgb = heatmap_rgb

        self.pred_var.set(pretty_label(self.state.label or ""))
        self.proba_var.set(f"{self.state.proba:.2f}%" if self.state.proba is not None else "")

        self.after_idle(self._render_all_panels)

        self.status_var.set("Predicción completada.")
        self.btn_save_csv.state(["!disabled"])
        self.btn_save_pdf.state(["!disabled"])
        self.btn_predict.state(["!disabled"])

    def save_results_csv(self) -> None:
        p = self._patient_dict()

        if not p["doc_num"] or not p["name"]:
            showinfo(title="Guardar", message="Complete nombre y documento antes de guardar.")
            return
        if self.state.label is None or self.state.proba is None:
            showinfo(title="Guardar", message="Primero realiza una predicción.")
            return
        out_dir = self.reports.ensure_output_dir(
            current_dir=self.state.output_dir,
            ask_directory=lambda: filedialog.askdirectory(
                title="Seleccione carpeta para guardar PDFs y CSV"
            ),
            on_cancel=showinfo,
        )
        if not out_dir:
            return
        self.state.output_dir = out_dir

        csv_path = self.reports.save_csv_history(
            output_dir=self.state.output_dir,
            patient=p,
            label=self.state.label,
            proba=float(self.state.proba),
            source_filename=os.path.basename(self.state.filepath or ""),
        )

        self.status_var.set(f"CSV guardado: {csv_path}")
        showinfo(title="Guardar", message=f"Datos guardados con éxito en:\n{csv_path}")

    def create_pdf(self) -> None:
        p = self._patient_dict()

        if not p["doc_num"] or not p["name"]:
            showinfo(title="PDF", message="Complete nombre y documento antes de generar el reporte.")
            return
        if self.state.label is None or self.state.proba is None or self.state.heatmap_rgb is None:
            showinfo(title="PDF", message="Primero presiona 'Predecir'.")
            return
        if self.state.original_pil is None:
            showinfo(title="PDF", message="No se encontró la imagen original para el reporte.")
            return
        out_dir = self.reports.ensure_output_dir(
            current_dir=self.state.output_dir,
            ask_directory=lambda: filedialog.askdirectory(
                title="Seleccione carpeta para guardar PDFs y CSV"
            ),
            on_cancel=showinfo,
        )
        if not out_dir:
            return
        self.state.output_dir = out_dir

        pdf_path = self.reports.save_pdf_report(
            output_dir=self.state.output_dir,
            patient=p,
            label=self.state.label,
            proba=float(self.state.proba),
            original_pil=self.state.original_pil,
            heatmap_rgb=self.state.heatmap_rgb,
            source_filename=os.path.basename(self.state.filepath or ""),
        )

        self.status_var.set(f"PDF generado: {pdf_path}")
        showinfo(title="PDF", message=f"Reporte generado con éxito:\n{pdf_path}")

    def delete(self) -> None:
        answer = askokcancel(title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING)
        if not answer:
            return

        self.p_name.set("")
        self.p_doc_type.set("CC")
        self.p_doc_num.set("")
        self.p_sex.set("")
        self.p_weight.set("")
        self.p_height.set("")
        self.p_age.set("")

        self.state.filepath = None
        self.state.array_bgr = None
        self.state.original_pil = None
        self.state.clear_prediction()

        self.pred_var.set("")
        self.proba_var.set("")

        self.lbl_img_name.configure(text="Ningún archivo cargado.")

        self.panel_reg.configure(image="", text="Sin imagen")
        self.panel_orig.configure(image="", text="Sin imagen")
        self.panel_hm.configure(image="", text="Sin heatmap")

        self.tk_img_reg = None
        self.tk_img_orig = None
        self.tk_img_hm = None

        self.btn_predict.state(["disabled"])
        self.btn_save_csv.state(["disabled"])
        self.btn_save_pdf.state(["disabled"])
        self.status_var.set("Listo. Registre los datos del paciente y adjunte una radiografía.")
        self._update_wizard_buttons()

        showinfo(title="Borrar", message="Los datos se borraron con éxito")

if __name__ == "__main__":
    App().mainloop()