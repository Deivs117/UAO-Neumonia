FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DISPLAY=:99 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# Tkinter + display virtual + VNC + noVNC
RUN apt-get update && apt-get install -y --no-install-recommends \
    tk \
    tcl \
    xvfb \
    fluxbox \
    x11vnc \
    novnc \
    websockify \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN pip install --no-cache-dir uv

# requirements primero (mejor cache)
COPY requirements.docker.txt /app/requirements.docker.txt

# venv + deps
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -r /app/requirements.docker.txt

# código fuente
COPY src /app/src

# carpetas de trabajo (para volúmenes)
RUN mkdir -p /app/models /app/data/input /app/data/output

# script de arranque
COPY docker/start_gui.sh /app/start_gui.sh
RUN chmod +x /app/start_gui.sh

EXPOSE 5900 6080

CMD ["/app/start_gui.sh"]
