from document_processor import process_document


result = process_document("uploads/EnzoSousaDosSantos_agenda4_TI_I.pdf")

print("Texto extraído:")
print(result["text"][:1000])

print("\nQuantidade de chunks:")
print(result["total_chunks"])

print("\nPrimeiro chunk:")
print(result["chunks"][0])