# UAO-Neumonia — GUI (Tkinter) + Grad-CAM + Reportes (CSV/PDF) + Tests + Docker (noVNC)

Aplicación con interfaz gráfica (Tkinter) para apoyar la **clasificación de neumonía** (p. ej. *Normal / Bacteriana / Viral*) a partir de imágenes **DICOM** y formatos estándar (**JPG/PNG**).  
Incluye explicación visual **Grad-CAM**, y genera salidas para el usuario en:

- **CSV histórico** (registro de predicciones)
- **PDF clínico** (datos del paciente + imagen original + heatmap Grad-CAM + resultado)

La app puede ejecutarse:

- **Local** (Python + `uv`)
- **Docker** (recomendado para el equipo): GUI vía **Xvfb + VNC + noVNC** (se ve desde el navegador)

---

## 1) Requisitos

### Local
- Python 3.10+ (recomendado 3.10/3.11)
- `uv` instalado

### Docker
- Docker Desktop funcionando
- (Opcional) GitHub Desktop

> En Docker la GUI **no** se abre como ventana nativa; se abre en el navegador con **noVNC**.

---

## 2) Estructura del proyecto

### Código fuente (paquete)
```

src/
neumonia_app/
**init**.py
main.py
integrator.py
read_img.py
preprocess_img.py
load_model.py
grad_cam.py
gui/
state.py
theme.py
image_utils.py
data_client_services.py

```

### Tests
```

test/
test_read_img.py
test_preprocess_img.py
test_predict_basic.py

```

### Docker
```

Dockerfile
docker/start_gui.sh
docker-compose.yml
requirements.docker.txt

````

---

## 3) Carpetas locales (NO se suben a Git)

Crea estas carpetas en la raíz del repo:

- `models/` → modelo `.h5`
- `data/input/` → imágenes de prueba (DICOM/JPG/PNG)
- `data/output/` → PDFs y CSV generados

### Crear carpetas

**Windows (PowerShell)**
```powershell
mkdir models
mkdir data
mkdir data\input
mkdir data\output
````

**Linux/macOS (bash)**

```bash
mkdir -p models data/input data/output
```

---

## 4) Modelo (CRÍTICO)

### 4.1 Opción 1 (recomendada): ruta fija por variable de entorno

Debes definir la variable:

* `NEUMONIA_MODEL_PATH` → ruta absoluta al modelo `.h5`

Ejemplos:

**Windows PowerShell**

```powershell
$env:NEUMONIA_MODEL_PATH="C:\ruta\al\repo\models\mi_modelo.h5"
```

**Windows CMD**

```bat
set NEUMONIA_MODEL_PATH=C:\ruta\al\repo\models\mi_modelo.h5
```

**Linux/macOS**

```bash
export NEUMONIA_MODEL_PATH="/ruta/al/repo/models/mi_modelo.h5"
```

> Esta es la forma más robusta para local y Docker.

### 4.2 Opción 2 (fallback): búsqueda automática de `.h5`

Si **NO** defines `NEUMONIA_MODEL_PATH`, el loader buscará automáticamente un archivo `*.h5` en carpetas típicas del proyecto (ej. `models/`, `./`, etc.) y:

* si hay **uno**, usa ese
* si hay **varios**, toma el **más reciente**

**Recomendación**: en `models/` deja un solo `.h5` para evitar confusión.

---

## 5) Rutas críticas en Docker (volúmenes)

En Docker se montan típicamente:

| Host (tu PC)    | Contenedor         | Uso                |
| --------------- | ------------------ | ------------------ |
| `./src`         | `/app/src`         | Código             |
| `./models`      | `/app/models`      | Modelo `.h5`       |
| `./data/input`  | `/app/data/input`  | Imágenes de prueba |
| `./data/output` | `/app/data/output` | PDFs + CSV         |

### Dentro de la GUI (noVNC)

* Cargar imagen desde: `/app/data/input`
* Guardar PDF/CSV en: `/app/data/output`

---

## 6) Ejecución local con `uv`

### 6.1 Instalar dependencias

```bash
uv pip install -r requirements.txt
```

### 6.2 Ejecutar la GUI

Desde la raíz del repo:

```bash
uv run python -m src.neumonia_app.main
```

> Si tu entrypoint local es distinto (por ejemplo `python src/neumonia_app/main.py`), úsalo, pero mantén el `NEUMONIA_MODEL_PATH` configurado.

---

## 7) Ejecución en Docker (recomendado)

### 7.1 Levantar con Docker Compose

```bash
docker compose up --build
```

Abrir en el navegador:

* `http://localhost:6080/vnc.html`

Para detener:

```bash
docker compose down
```

### 7.2 Variable del modelo en Docker

Asegúrate de que tu `docker-compose.yml` contenga algo como:

* `NEUMONIA_MODEL_PATH=/app/models/<tu_modelo>.h5`

y que el archivo exista en `models/`.

---

## 8) Correr tests

### 8.1 Local con `uv`

Desde la raíz del repo:

**Linux/macOS**

```bash
PYTHONPATH=src uv run pytest -q
```

**Windows PowerShell**

```powershell
$env:PYTHONPATH="src"
uv run pytest -q
```

### 8.2 Docker con `uv`

Si `uv` está instalado dentro de tu imagen:

```bash
docker compose run --rm neumonia-gui sh -lc 'PYTHONPATH=src uv run pytest -q'
```

> Cambia `neumonia-gui` por el nombre real del servicio en tu `docker-compose.yml`.

---

## 9) Descripción de módulos (arquitectura y responsabilidades)

### Núcleo (Core)

* **`integrator.py`**
  Orquestador oficial del pipeline. Coordina:

  * lectura (`read_img.py`)
  * preprocesamiento (`preprocess_img.py`)
  * carga del modelo (`load_model.py`) con cache `_model`
  * inferencia + Grad-CAM (`grad_cam.py`)

* **`read_img.py`**
  Lectura de:

  * DICOM (`pydicom`) → convierte a PIL y BGR
  * JPG/PNG (`cv2`) → convierte a PIL y BGR
    Retorna siempre: `(array_bgr, img_pil)`.

* **`preprocess_img.py`**
  Preprocesamiento “puro” (sin IO):
  resize/gray/CLAHE/normalización → batch listo para Keras.

* **`load_model.py`**
  Resuelve la ruta del modelo y lo carga con `tf.keras.models.load_model(...)`.
  Prioriza:

  1. `model_path` explícito (si se pasa)
  2. variable de entorno `NEUMONIA_MODEL_PATH`
  3. búsqueda automática `*.h5` en rutas típicas

* **`grad_cam.py`**
  Servicio de predicción y Grad-CAM:

  * produce `label`, `proba_pct`, `heatmap_rgb`
  * `layer_name` tiene fallback automático si el nombre cambia (evita rompimientos por cambio de modelo)

### GUI (Interfaz)

* **`main.py`**
  Solo UI (Tkinter): pantallas, validaciones de formulario, render de imágenes, interacción con usuario.
  No implementa pipeline ML ni generación de documentos; delega.

* **`gui/state.py`**
  Modelos POO de estado (`Patient`, `AppState`) + conversores seguros (`safe_float`, `safe_int`).

* **`gui/theme.py`**
  Tema visual (colores/estilos) y helper de “cards”.

* **`gui/image_utils.py`**
  Utilidades de imagen para UI (ajustes de tamaño, formato de labels).

* **`gui/data_client_services.py`**
  Servicios de salida al usuario:

  * `ensure_output_dir` (gestiona selección/creación de carpeta)
  * `save_csv_history` (historial)
  * `save_pdf_report`/`generate_pdf_report` (PDF clínico)

---

## 10) Errores comunes

### “No se encontró un archivo de modelo (.h5)”

* Asegura que `NEUMONIA_MODEL_PATH` apunta a un `.h5` real
* o deja un único `.h5` dentro de `models/`

### Logs CUDA / cuInit en Docker

TensorFlow puede intentar inicializar GPU. Normalmente no es fatal (corre CPU).
Si quieres silenciar/forzar CPU, usa env vars en Docker:

* `CUDA_VISIBLE_DEVICES=-1`
* `TF_CPP_MIN_LOG_LEVEL=2` (o 3)

---

## 11) Licencia
Este proyecto puede ser distribuido bajo una licencia open-source.
Las opciones recomendadas para este tipo de proyecto (Deep Learning + investigación) son:

MIT License
Permite uso comercial
Permite modificación y redistribución
Muy simple y permisiva
Solo exige mantener el aviso de copyright
Apache License 2.0
Permite uso comercial
Permite modificación y redistribución
Incluye protección explícita de patentes
Más formal y usada en proyectos grandes (ej. TensorFlow)
License
MIT — see LICENSE for details.