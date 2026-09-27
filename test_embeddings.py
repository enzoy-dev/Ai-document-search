from document_processor import process_document
from embedding_service import generate_embeddings


result = process_document("uploads/EnzoSousaDosSantos_agenda4_TI_I.pdf")

chunks = result["chunks"]

embeddings = generate_embeddings(chunks)

print("Quantidade de chunks:", len(chunks))
print("Quantidade de embeddings:", len(embeddings))
print("Tamanho do primeiro embedding:", len(embeddings[0]))
print("Primeiros valores:", embeddings[0][:5])