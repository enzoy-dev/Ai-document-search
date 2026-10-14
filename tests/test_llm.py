from app.llm import generate_answer


context = """
A atividade mostrou que prompts mais específicos produzem respostas
mais adequadas. A IA conseguiu adaptar a explicação para uma criança,
usar uma analogia e respeitar o limite de linhas.
"""

question = "O que o documento fala sobre prompts?"

answer = generate_answer(question, context)

print("\nRESPOSTA:\n")
print(answer)