import os
from flask import Flask, jsonify, redirect, render_template, request, url_for
from dotenv import load_dotenv
from google import genai
from rag import build_context, build_rag_index_from_gcs, get_rag_status, mark_rag_error, retrieve_relevant_chunks

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


def get_genai_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Falta GEMINI_API_KEY. Define la variable de entorno antes de arrancar la app."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(user_prompt: str) -> str:
    client = get_genai_client()

    full_prompt = f"""{SYSTEM_PROMPT}

Pregunta del usuario:
{user_prompt}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    return (response.text or "").strip()


def ask_gemini_with_rag(user_prompt: str):
    client = get_genai_client()
    retrieved_chunks = retrieve_relevant_chunks(user_prompt)
    if not retrieved_chunks:
        return ask_gemini(user_prompt), []

    context = build_context(retrieved_chunks)
    rag_prompt = f"""{SYSTEM_PROMPT}

Estás en modo RAG.
Debes responder usando prioritariamente el contexto proporcionado.
Si el contexto no contiene la respuesta, dilo claramente.
Cuando cites de dónde sale algo, menciona el nombre del documento y la página si aparece en el contexto.
No inventes datos que no estén en el contexto.

Contexto recuperado:
{context}

Pregunta del usuario:
{user_prompt}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=rag_prompt,
    )

    return (response.text or "").strip(), retrieved_chunks


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    error = None
    prompt = ""
    use_rag = True
    rag_status = get_rag_status()
    sources = []

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        use_rag = request.form.get("use_rag") == "on"

        if not prompt:
            error = "Escribe una pregunta antes de enviar."
        else:
            try:
                if use_rag and rag_status.get("loaded"):
                    answer, sources = ask_gemini_with_rag(prompt)
                else:
                    answer = ask_gemini(prompt)
                if not answer:
                    error = "Gemini no ha devuelto texto en esta respuesta."
            except Exception as exc:
                error = f"Error al llamar a Gemini: {exc}"

    return render_template(
        "index.html",
        app_title=APP_TITLE,
        prompt=prompt,
        answer=answer,
        error=error,
        model_name=GEMINI_MODEL,
        rag_status=rag_status,
        use_rag=use_rag,
        sources=sources,
        gcs_bucket_name=os.getenv("GCS_BUCKET_NAME", ""),
        gcs_prefix=os.getenv("GCS_PREFIX", ""),
    )


@app.post("/reload-rag")
def reload_rag():
    try:
        build_rag_index_from_gcs()
    except Exception as exc:
        mark_rag_error(str(exc))
    return redirect(url_for("index"))


@app.get("/rag-status")
def rag_status():
    return jsonify(get_rag_status())


@app.get("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
