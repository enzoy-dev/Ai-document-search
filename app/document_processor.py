from pathlib import Path
from pypdf import PdfReader
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts).strip()


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Chunking mais inteligente:
    - tenta quebrar em parágrafos
    - se o parágrafo for grande demais, quebra por tamanho com overlap
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Se o parágrafo sozinho já é maior que o chunk_size
        if len(paragraph) > chunk_size:
            # Flush o que tinha
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Quebra o parágrafo grande
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunk = paragraph[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += chunk_size - overlap
            continue

        # Tenta juntar parágrafos
        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{paragraph}".strip() if current_chunk else paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_document(file_path: str) -> dict:
    text = extract_text_from_pdf(file_path)
    chunks = split_text(text)

    return {
        "text": text,
        "chunks": chunks,
        "total_chunks": len(chunks),
    }