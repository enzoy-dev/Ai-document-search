"""Standalone search experiment: any PDF, no Gemini request or personal filename."""
import argparse
from pathlib import Path
import tempfile

from app.document_processor import process_document
from app.embedding_service import generate_embeddings, get_model_identity
from app.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--question', required=True)
    args = parser.parse_args()
    chunks = process_document(str(args.pdf))['chunks']
    if not chunks:
        parser.error('The PDF has no extractable text')
    with tempfile.TemporaryDirectory() as directory:
        store = VectorStore(directory=directory)
        store.add(chunks, generate_embeddings([chunk['text'] for chunk in chunks]), args.pdf.name, encoder_identity=get_model_identity())
        for result in store.search(generate_embeddings([args.question])[0], encoder_identity=get_model_identity()):
            print(f"Page {result['page']} | distance {result['distance']:.4f}")
            print(result['text'])


if __name__ == '__main__':
    main()
