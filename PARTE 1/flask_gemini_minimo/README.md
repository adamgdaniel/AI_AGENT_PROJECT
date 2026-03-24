# Flask + Gemini API: repositorio mínimo

Este repositorio contiene la **primera versión** de la app para la sesión: una app web mínima en **Flask** que hace una pregunta a **Gemini** usando una **API key**, sin `gcloud`, sin ADC y sin Google CLI.

## Qué incluye

- `app.py`: aplicación Flask mínima.
- `templates/index.html`: interfaz web simple.
- `static/styles.css`: estilos.
- `requirements.txt`: dependencias Python.
- `Dockerfile`: contenedor listo para Cloud Run.
- `.env.example`: variables de entorno de ejemplo.
- `.gitignore` y `.dockerignore`.

## Estructura

```text
flask_gemini_minimo/
├── app.py
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── .dockerignore
├── templates/
│   └── index.html
└── static/
    └── styles.css
```

## 1) Cómo conseguir tu API key sin `gcloud`

### Opción recomendada para esta práctica: Google AI Studio

1. Abre Google AI Studio.
2. Inicia sesión con la cuenta que tenga acceso a tu proyecto.
3. Si tu proyecto no aparece, entra en **Dashboard → Projects → Import projects**.
4. Busca tu proyecto de Google Cloud y selecciónalo.
5. Ve a **API Keys**.
6. Crea una nueva clave dentro de ese proyecto.
7. Copia la clave.

## 2) Configuración local

### Crear entorno virtual

```bash
python -m venv .venv
```

### Activar entorno virtual

#### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Crear tu archivo `.env`

Copia `.env.example` a `.env` y edítalo:

```bash
cp .env.example .env
```

En Windows puedes hacerlo manualmente desde el explorador o con:

```powershell
Copy-Item .env.example .env
```

Pon al menos tu clave aquí:

```env
GEMINI_API_KEY=tu_clave_real
```

## 3) Ejecutar la app en local

```bash
python app.py
```

La app quedará disponible en:

- `http://localhost:8080`

También puedes usar Flask directamente:

```bash
flask --app app run --host=0.0.0.0 --port=8080
```

## 4) Probar la app

1. Abre la web en el navegador.
2. Escribe una pregunta.
3. Pulsa **Preguntar**.
4. La respuesta aparecerá debajo.

## 5) Health check

Hay un endpoint útil para comprobar si el contenedor ha arrancado:

- `GET /health`

## 6) Preparado para Docker y Cloud Run

Construcción local opcional del contenedor:

```bash
docker build -t flask-gemini-minimo .
```

Ejecución local del contenedor:

```bash
docker run --rm -p 8080:8080 --env-file .env flask-gemini-minimo
```

## 7) Variables de entorno usadas

- `GEMINI_API_KEY`: **obligatoria**.
- `GEMINI_MODEL`: opcional. Por defecto `gemini-2.5-flash`.
- `APP_TITLE`: opcional.
- `SYSTEM_PROMPT`: opcional.
- `PORT`: la usa Cloud Run automáticamente.

## 8) Siguiente paso

En la siguiente fase:

- subiremos este repo a GitHub,
- desplegaremos a Cloud Run desde el repositorio,
- y luego añadiremos RAG con documentos.
