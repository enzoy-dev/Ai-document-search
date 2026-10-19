from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

from app.errors import ServiceUnavailableError

_client = None


def get_client():
    global _client
    if not GEMINI_API_KEY:
        raise ServiceUnavailableError("GEMINI_API_KEY is not configured. Set it to enable generated answers.")
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": 30000})
    return _client


def generate_answer(question: str, context: str) -> str:
    prompt = f"""Você é um assistente especializado em responder perguntas com base exclusivamente nos documentos fornecidos.

Regras:
- Use APENAS as informações do contexto abaixo.
- Se a informação não estiver no contexto, diga claramente: "Não encontrei essa informação nos documentos indexados."
- Seja direto e objetivo.
- Responda no mesmo idioma da pergunta.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

    try:
        response = get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except ServiceUnavailableError:
        raise
    except Exception as exc:
        raise ServiceUnavailableError("Gemini request failed. Check credentials, model and network access.") from exc
    if not response.text or not response.text.strip():
        raise ServiceUnavailableError("Gemini returned an empty answer. Please try again.")
    return response.text.strip()
