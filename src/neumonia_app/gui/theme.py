from __future__ import annotations
"""
theme.py

Definición de tema visual (estilo clínico) para Tkinter/ttk.
Incluye constantes de color/fuentes y helpers para construir "cards".
"""

import tkinter as tk
from tkinter import ttk

# ===== Estilo GUI (clínico) =====
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
    """Aplica un tema clínico (colores, fuentes y estilos ttk) sobre `root`."""
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

    style.configure("Card.TFrame", background=CARD_BG)
    style.configure("Card.TLabel", background=CARD_BG, foreground=FG_TEXT, font=FONT_BASE)
    style.configure("Card.Muted.TLabel", background=CARD_BG, foreground=MUTED, font=FONT_SMALL)
    style.configure("Card.H1.TLabel", background=CARD_BG, foreground=FG_TEXT, font=FONT_H1)

    return style


def card(parent: ttk.Frame) -> tuple[tk.Frame, ttk.Frame]:
    """Crea un card con borde suave."""
    outer = tk.Frame(parent, bg=BORDER, highlightthickness=0, bd=0)
    inner = ttk.Frame(outer, style="Card.TFrame", padding=12)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    return outer, inner