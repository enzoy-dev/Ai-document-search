from document_processor import process_document
from embedding_service import generate_embeddings, model
from vector_store import VectorStore


result = process_document("uploads/EnzoSousaDosSantos_agenda4_TI_I.pdf")

chunks = result["chunks"]

embeddings = generate_embeddings(chunks)

store = VectorStore()

store.add(chunks, embeddings)

question = "Sobre o que esse documento fala?"

query_embedding = model.encode(question)

results = store.search(query_embedding)

print("\nRESULTADOS:\n")

for i, item in enumerate(results, start=1):
    print(f"--- Chunk {i} ---")
    print(item["text"])
    print("Distância:", item["distance"])
    print()