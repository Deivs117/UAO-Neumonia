from __future__ import annotations

from PIL import Image, ImageOps

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


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