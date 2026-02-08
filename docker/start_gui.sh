#!/usr/bin/env bash
set -e

# Display virtual para Tkinter
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
sleep 0.5

# Window manager liviano
fluxbox >/dev/null 2>&1 &

# VNC (sin contraseña por simplicidad)
x11vnc -display :99 -forever -shared -nopw -rfbport 5900 >/dev/null 2>&1 &

# noVNC (para verlo en el navegador)
websockify --web=/usr/share/novnc/ 6080 localhost:5900 >/dev/null 2>&1 &

# Activar venv
source /app/.venv/bin/activate

# Modelo por defecto (puedes sobreescribir con -e NEUMONIA_MODEL_PATH=...)
export NEUMONIA_MODEL_PATH="${NEUMONIA_MODEL_PATH:-/app/models/conv_MLP_84.h5}"

# Ejecutar app (tu paquete está en /app/src por PYTHONPATH)
uv run python -m neumonia_app.main
