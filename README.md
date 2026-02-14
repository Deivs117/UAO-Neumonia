# UAO-Neumonia — App GUI (Tkinter) + Docker (noVNC)

Este repositorio contiene una aplicación con interfaz gráfica (Tkinter) para apoyar la clasificación de neumonía (normal / bacteriana / viral) a partir de imágenes (DICOM / JPG), y genera reportes en PDF + registros CSV.

La app corre:
- **Local** (opcional) con Python + `uv`
- **Docker** (recomendado para el equipo) usando **Xvfb + VNC + noVNC** para ver la GUI desde el navegador

---

## 1) Requisitos (para el equipo)

- Git instalado
- Docker Desktop instalado y funcionando
- (Opcional) GitHub Desktop

> Nota: en Docker la GUI no se abre como ventana nativa de Windows; se abre por navegador en noVNC.

---

## 2) Estructura del proyecto

Código fuente (paquete):
```

src/
neumonia_app/
main.py
integrator.py
load_model.py
read_img.py
preprocess_img.py
grad_cam.py

```

Archivos Docker:
```

Dockerfile
docker/start_gui.sh
requirements.docker.txt
docker-compose.yml

````

Carpetas **locales** (NO se suben a Git):
- `models/` → aquí va el modelo `.h5`
- `data/input/` → imágenes para pruebas (DICOM/JPG)
- `data/output/` → PDFs y CSV generados

---

## 3) Preparación local (carpetas y archivos que NO están en Git)

En la raíz del repo, crear estas carpetas:

### Windows (PowerShell)
```powershell
mkdir models
mkdir data
mkdir data\input
mkdir data\output
mkdir assets
mkdir assets\samples
mkdir assets\samples\DICOM
mkdir assets\samples\JPG
````

### Linux/macOS (bash)

```bash
mkdir -p models data/input data/output assets/samples/DICOM assets/samples/JPG
```

### 3.1) Agregar el modelo `.h5` (OBLIGATORIO)

Coloca el archivo del modelo aquí:

```
models/conv_MLP_84.h5
```

> Si tu modelo se llama diferente, no hay problema, pero entonces debes actualizar la variable `NEUMONIA_MODEL_PATH` en el `docker-compose.yml` o renombrarlo.

### 3.2) Agregar imágenes de ejemplo (OPCIONAL)

Coloca algunas imágenes para pruebas:

* DICOM: `assets/samples/DICOM/*.dcm`
* JPG/PNG: `assets/samples/JPG/*.(jpg|png)`

Para que la app las vea dentro del contenedor, copia algunas también a:

* `data/input/`

Ejemplo:

```
data/input/ejemplo1.dcm
data/input/ejemplo2.jpg
```

---

## 4) Docker (recomendado)

### 4.1) ¿Cómo funciona la GUI en Docker?

Dentro del contenedor se levanta un display virtual (Xvfb) y se expone por:

* **VNC** en el puerto `5900`
* **noVNC (web)** en el puerto `6080`

Tú vas a ver la GUI en tu navegador:

* [http://localhost:6080/vnc.html](http://localhost:6080/vnc.html)

### 4.2) Conexión entre carpetas (volúmenes)

Docker monta estas carpetas del host al contenedor:

| Host (tu PC)    | Contenedor         | Uso                  |
| --------------- | ------------------ | -------------------- |
| `./src`         | `/app/src`         | Código (modo dev)    |
| `./models`      | `/app/models`      | Modelo `.h5`         |
| `./data/input`  | `/app/data/input`  | Imágenes de prueba   |
| `./data/output` | `/app/data/output` | PDFs y CSV generados |

Así, si agregas imágenes nuevas a `data/input/` **después** de construir la imagen, el contenedor las ve inmediatamente (porque es un volumen).

---

## 5) Comandos exactos para construir y correr

### Opción A — Docker Compose (recomendado)

En la raíz:

```bash
docker compose up --build
```

Luego abre:

* [http://localhost:6080/vnc.html](http://localhost:6080/vnc.html)

Para detener:

```bash
docker compose down
```

### Opción B — Docker Build + Docker Run (alternativa)

Construir la imagen:

```bash
docker build -t neumonia-gui .
```

Correr el contenedor (ejemplo PowerShell):

```powershell
docker run --rm -it `
  -p 6080:6080 -p 5900:5900 `
  -e NEUMONIA_MODEL_PATH=/app/models/conv_MLP_84.h5 `
  -v "${PWD}\src:/app/src" `
  -v "${PWD}\models:/app/models:ro" `
  -v "${PWD}\data\input:/app/data/input:ro" `
  -v "${PWD}\data\output:/app/data/output" `
  neumonia-gui
```

---

## 6) Desarrollo: ver cambios en `src` dentro del contenedor

### 6.1) Caso normal (sin autoreload)

Si editas un `.py` en `src/`, el contenedor **sí ve el archivo nuevo**, pero Python/Tkinter normalmente requiere reinicio para cargarlo.

Reiniciar el contenedor:

```bash
docker compose restart neumonia-gui
```

## 7) Uso dentro de la GUI (en noVNC)

### Cargar imagen

Desde el file picker, navega a:

* `/app/data/input`  (recomendado para pruebas)

### Guardar PDF/CSV

Selecciona como carpeta:

* `/app/data/output`

Esto se reflejará en tu PC en:

* `data/output/`

---

## 8) Errores comunes

### “No se encontró el modelo .h5…”

Verifica:

* Que exista `models/conv_MLP_84.h5`
* Que el `docker-compose.yml` esté apuntando bien a `NEUMONIA_MODEL_PATH=/app/models/conv_MLP_84.h5`

### Mensaje CUDA / cuInit error

Si ves errores tipo CUDA, normalmente es TensorFlow intentando usar GPU. No suele ser fatal: continúa en CPU.

---

## 9) Flujo Git recomendado (equipo)

* No trabajar directo en `main`
* Trabajar en tu rama `dev/<nombre>`
* Hacer Pull Request hacia `develop`

Comandos:

```bash
git fetch origin
git checkout dev/<tu-nombre>
git pull
```

Luego:

```bash
git add .
git commit -m "mensaje"
git push
```

Y abrir PR: `dev/<tu-nombre>` → `develop`

---

## 10) Notas finales

* No subir a Git: `models/`, `data/`, `assets/samples/` (datasets o archivos sensibles/pesados)
* Todo lo de Docker GUI está gestionado por: `docker/start_gui.sh`

## Licencia

Este proyecto puede ser distribuido bajo una licencia open-source.  
Las opciones recomendadas para este tipo de proyecto (Deep Learning + investigación) son:

### MIT License
- Permite uso comercial
- Permite modificación y redistribución
- Muy simple y permisiva
- Solo exige mantener el aviso de copyright

### Apache License 2.0
- Permite uso comercial
- Permite modificación y redistribución
- Incluye protección explícita de patentes
- Más formal y usada en proyectos grandes (ej. TensorFlow)

---

Cómo agregar una licencia al repositorio

Crear el archivo `LICENSE` en la raíz del proyecto:

```powershell
New-Item -ItemType File .\LICENSE

```

