from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)


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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text.strip()