# Comparação controlada — replay de rankings arquivados

Estes dois relatórios usam o holdout de 24 perguntas e os rankings de modelos
reais já arquivados no repositório. Os hashes dos PDFs, dataset, perguntas,
rótulos e trechos são verificados antes da comparação. Não houve download de
pesos nem nova chamada ao modelo: o acesso ao Hugging Face foi bloqueado pela
rede nesta execução.

```bash
python -m experiments.compare_retrieval --dataset evaluations/holdout/retrieval.json --archived-report evaluations/holdout/baselines/minilm.json --output evaluations/results/minilm-replay.json
python -m experiments.compare_retrieval --dataset evaluations/holdout/retrieval.json --archived-report evaluations/holdout/baselines/multilingual.json --output evaluations/results/multilingual-replay.json
```

O relatório compara o top-k semântico original, BM25 lexical atual e fusão RRF.
O pool semântico só possui os quatro vizinhos registrados. O runtime híbrido usa
um pool maior, por isso **este replay não é o benchmark completo do modo híbrido**.
Distâncias de candidatos somente lexicais ficam desconhecidas (`null`), em vez
de serem fabricadas. Distâncias já conhecidas são preservadas. Não há nova
latência de embeddings nem avaliação de respostas Gemini.

| Modelo / modo | Evidência esperada aceita (16 casos) | MRR@4 | Perguntas sem resposta rejeitadas (4 casos) |
| --- | --- | --- | --- |
| MiniLM semântico arquivado | 13/16 | 0,90625 | 1/4 |
| MiniLM + RRF em replay | 14/16 | 0,93750 | 0/4 |
| Multilíngue semântico arquivado | 16/16 | 1,00000 | 0/4 |
| Multilíngue + RRF em replay | 16/16 | 0,96875 | 0/4 |
| Somente lexical (mesmo corpus) | 14/16 | 0,796875 | 0/4 |

No MiniLM a fusão melhora cobertura/ranking, mas perde a rejeição que existia.
No multilíngue não melhora cobertura e piora ranking. Estes resultados não
justificam substituir o padrão. Há perguntas ambíguas tratadas separadamente das
quatro sem informação; aceitar trecho não prova comportamento de esclarecimento.

Relatórios completos: [MiniLM](holdout-minilm-replay.json) e
[multilíngue](holdout-multilingual-replay.json). Mantêm resultados por pergunta,
aceitação e scores, além de hashes da entrada e do relatório semântico.

São conjuntos sintéticos pequenos, já usados no desenvolvimento. Se ajustar
parâmetros por esses resultados, use outro corpus antes de alegar generalização.
A próxima verificação é executar os dois modos com os mesmos pesos reais e
avaliar respostas/recusas com revisão factual separada, incluindo os documentos
adversariais. Não é correto publicar estes números como precisão geral da IA.
