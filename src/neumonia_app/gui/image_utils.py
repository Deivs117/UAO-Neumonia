from __future__ import annotations
"""
image_utils.py

Utilidades de imagen para GUI:
- fit_box: ajustar una imagen PIL a un panel manteniendo proporción
- pretty_label: normalizar etiquetas para UI/reporte
"""

from PIL import Image, ImageOps

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def fit_box(pil_img: Image.Image, box_w: int, box_h: int, fill: int = 0) -> Image.Image:
    """
    Ajusta `pil_img` a un cuadro (box_w, box_h) preservando relación de aspecto.
    Rellena el fondo con `fill` (gris/negro) según el modo de imagen.

    Retorna una nueva imagen PIL del tamaño exacto del panel.
    """
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
    """Convierte etiquetas internas a un texto presentable para UI."""
    l = (label or "").strip().lower()
    if l == "normal":
        return "Normal"
    if l == "viral":
        return "Neumonía viral"
    if l == "bacteriana":
        return "Neumonía bacteriana"
    return (label or "").strip()