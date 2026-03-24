import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Eres un asistente útil, claro y didáctico para estudiantes de Big Data y Cloud. "
    "Responde en español salvo que el usuario pida otro idioma.",
)
APP_TITLE = os.getenv("APP_TITLE", "Mi primera app con Gemini")
sesion = []


def ask_gemini(user_prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Falta GEMINI_API_KEY. Define la variable de entorno antes de arrancar la app."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    full_prompt = f"""{SYSTEM_PROMPT}

Pregunta del usuario:
{user_prompt}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    return (response.text or "").strip()


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    error = None
    prompt = ""

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()

        if not prompt:
            error = "Escribe una pregunta antes de enviar."
        else:
            try:
                sesion.append("user: " + prompt)
                prompt += "{antes de responder ten en cuenta la conversacion previa: }"
                prompt += " ".join(sesion)
                answer = ask_gemini(prompt)
                if not answer:
                    error = "Gemini no ha devuelto texto en esta respuesta."
            except Exception as exc:
                error = f"Error al llamar a Gemini: {exc}"
                
    if answer:                                    
        sesion.append("system: " + answer)    

    # --- CAMBIO AÑADIDO: Faltaba devolver la plantilla al final de la función ---
    return render_template(
        "index.html",
        app_title=APP_TITLE,
        prompt=prompt,
        answer=answer,
        error=error,
        model_name=GEMINI_MODEL,
    )

@app.get("/health")
def health():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)