from app.document_processor import process_document
from app.embedding_service import generate_embeddings


def test_generate_embeddings():
    # Usa um PDF que você já tenha na pasta uploads (ou cria um mock)
    result = process_document("uploads/EnzoSousaDosSantos_agenda4_TI_I.pdf")

    chunks = result["chunks"]
    assert len(chunks) > 0

    # Agora chunks são dicts, então pegamos só o texto
    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 384  # dimensão do all-MiniLM-L6-v2