import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=api_key)


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
Você é um assistente de perguntas e respostas baseado em documentos.

Responda à pergunta usando somente as informações presentes no contexto abaixo.

Se a resposta não estiver no contexto, diga claramente que a informação não foi encontrada no documento.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    return response.text