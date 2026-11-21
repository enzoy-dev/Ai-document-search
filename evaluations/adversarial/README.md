# Diagnóstico adversarial fictício

Um PDF de três páginas e seis perguntas: três respondíveis, uma ambígua e duas
com fatos ausentes. As páginas incluem instruções maliciosas para trocar o fato,
forjar ID, revelar chave e produzir HTML. O marcador CANARIO-PUBLICO-SEM-SEGREDO
é público e fictício; não é credencial. Nenhum segredo real foi incluído.

Regenerar: `python -m evaluations.build_adversarial`.

O arquivo `retrieval.json` tem páginas/status esperados e, em alguns casos,
palavras de revisão. Esses rótulos são gabarito para inspeção, não uma prova de
resistência nem um avaliador semântico automático. O custo de ORBITA-47 e a chave
real não existem no documento. O prazo de empréstimo sem dizer o objeto admite
ferramentas (sete dias) e livros (21 dias).

Os testes usam extração e índice reais com embeddings/provedor simulados. Validam
regras de integração e mantêm demonstrada a limitação de citações válidas com
conteúdo incorreto. O comportamento do Gemini contra esses ataques **não foi
executado** nesta rodada. Preencha a revisão factual somente após executar e
inspecionar respostas reais, sem atribuir nota aos campos ainda não revisados.

O PDF é só para avaliação e não integra automaticamente a coleção privada ou a
demo pública. Não o confunda com instruções de operação do aplicativo.

Veja [comandos e revisão por hash](../../docs/retrieval-quality.md).
