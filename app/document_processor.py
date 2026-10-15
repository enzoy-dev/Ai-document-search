from pypdf import PdfReader
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_pages_from_pdf(file_path: str) -> list[dict]:
    """
    Retorna lista de {"page": número, "text": texto da página}
    """
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page": i,
                "text": text.strip()
            })

    return pages


def split_text_with_page(text: str, page: int, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Quebra o texto de uma página em chunks, mantendo o número da página.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append({"text": current_chunk.strip(), "page": page})
                current_chunk = ""

            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append({"text": piece, "page": page})
                start += chunk_size - overlap
            continue

        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{paragraph}".strip() if current_chunk else paragraph
        else:
            if current_chunk:
                chunks.append({"text": current_chunk.strip(), "page": page})
            current_chunk = paragraph

    if current_chunk:
        chunks.append({"text": current_chunk.strip(), "page": page})

    return chunks


def process_document(file_path: str) -> dict:
    pages = extract_pages_from_pdf(file_path)

    all_chunks = []
    for page_data in pages:
        page_chunks = split_text_with_page(page_data["text"], page_data["page"])
        all_chunks.extend(page_chunks)

    return {
        "chunks": all_chunks,          # agora é lista de dicts {"text", "page"}
        "total_chunks": len(all_chunks),
    }