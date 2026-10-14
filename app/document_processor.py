from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extrai todo o texto de um arquivo PDF.
    """

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def split_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Divide o texto em pedaços menores (chunks).
    """

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def process_document(file_path: str):
    """
    Extrai o texto e divide o documento em chunks.
    """

    text = extract_text_from_pdf(file_path)

    chunks = split_text(text)

    return {
        "text": text,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }