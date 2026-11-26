# Qualidade de recuperação e de resposta

## O que foi implementado

A API e a interface aceitam `retrieval_mode=semantic` (padrão) ou `hybrid`. A busca de trechos continua em GET; a geração de respostas usa POST com JSON para não colocar perguntas potencialmente sensíveis na URL.
A escolha vale para aquela requisição, sem trocar pesos nem reindexar a coleção.
As rotas privadas continuam exigindo chave; a demo continua isolada e limitada.

Exemplo, com a chave já configurada no seu terminal:

```bash
curl --get http://127.0.0.1:8000/retrieve \
  -H "X-API-Key: $API_KEY" \
  --data-urlencode 'question=Qual é a retenção do ORBITA-47?' \
  --data-urlencode 'retrieval_mode=hybrid'

curl --fail-with-body -X POST http://127.0.0.1:8000/search \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"Qual é a retenção do ORBITA-47?","retrieval_mode":"hybrid"}'
```

O modo híbrido combina ranking BM25 de termos com ranking semântico usando
Reciprocal Rank Fusion, pesos iguais e constante 60. A tokenização ignora caixa,
acentos e uma lista curta de palavras funcionais; preserva códigos com hífen.
Não há lematização, reranker ou garantia de compreensão lexical bilíngue.

O pool de cada canal tem até `max(20,5*top_k)` trechos, limitado ao corpus.
Os canais usam os IDs do mesmo snapshot transacional. O ranking lexical é
recalculado por consulta, percorrendo os trechos em memória; não é um mecanismo
pronto para milhões de documentos. A versão atual mantém a busca vetorial exata.

No modo semântico, vale o limiar de distância existente. No híbrido, um trecho
pode entrar por distância aceita **ou** por correspondência lexical BM25 positiva.
Isso amplia cobertura de termos exatos, mas também pode aceitar mais distratores
e perguntas cujo fato não existe. A API informa `acceptance_reason`, score lexical
e score de fusão. Distância/relevance continuam descrevendo apenas proximidade
semântica; score de fusão e BM25 não são probabilidade de resposta correta.

## Três decisões observáveis

As respostas de `/search` e `/demo/search` acrescentam:

| Campo | Exemplo | Significado |
| --- | --- | --- |
| `decision_stage` | `retrieval` | Nenhum trecho aceito; Gemini não foi chamado |
| `decision_reason` | `no_accepted_passages` | Falta de contexto aceito, não prova de ausência em todo o PDF |
| `decision_stage` | `generation` | Houve contexto e tentativa de geração |
| `decision_reason` | `model_abstained` | O modelo devolveu `not_found` após contexto |
| `decision_reason` | `model_requested_clarification` | O modelo pediu esclarecimento |
| `decision_reason` | `cited_response` | Resposta com IDs de fontes validados, sem prova semântica de cada afirmação |
| `generation_attempted` | `true` / `false` | Distingue os dois caminhos |

A interface diferencia a busca sem trechos aceitos da recusa após geração.
Uma pergunta vaga ainda pode ser rejeitada pela recuperação; não há uma regra
que identifique automaticamente toda ambiguidade antes da busca. O contrato e
as mensagens ficaram mais observáveis, mas a qualidade real de esclarecimento
continua exigindo avaliação com o modelo.

## Avaliar busca e geração separadamente

O avaliador real aceita `--retrieval-mode`:

```bash
python -m experiments.evaluate_retrieval --dataset evaluations/holdout/retrieval.json --retrieval-mode semantic --output evaluations/results/semantic.json
python -m experiments.evaluate_retrieval --dataset evaluations/holdout/retrieval.json --retrieval-mode hybrid --output evaluations/results/hybrid.json
```

Isso exige o modelo real em cache ou download autorizado. [O replay arquivado](../evaluations/comparisons/README.md)
é uma alternativa diagnóstica limitada, não uma execução desses comandos.

A avaliação de respostas registra `retrieved_sources` mesmo quando há recusa,
fontes citadas, etapa/motivo e se houve tentativa de geração. Há métricas de
status e citações, além de contagem de abstenção/esclarecimento após contexto.
A taxa de falha de recuperação em casos respondíveis informa seu denominador
com diagnóstico de etapa; relatos antigos sem esse campo não viram observações
comprovadas de zero falhas.

```bash
python -m experiments.evaluate_answers --dataset evaluations/adversarial/retrieval.json --retrieval-mode hybrid --allow-paid-generation --max-cases 6 --output evaluations/results/adversarial-answers.json
```

Só execute depois de configurar sua chave e avaliar custos. Não foi feita uma
chamada paga nesta entrega. Nenhuma taxa de resistência ao Gemini é afirmada.

## Revisão factual com rastreabilidade

Crie uma ficha vinculada ao hash do relatório exato:

```bash
python -m experiments.review_answers --report evaluations/results/adversarial-answers.json --output evaluations/results/human-review.json
```

Abra cada fonte, confira **todas as afirmações**, recusas e pedidos de detalhe.
Preencha `correctness`, `support`, `behavior`, `notes` e identificação do revisor.
Campos começam como `not_reviewed`; não contam como revisão concluída.

```bash
python -m experiments.review_answers --report evaluations/results/adversarial-answers.json --reviews evaluations/results/human-review.json --output evaluations/results/human-summary.json
```

O resumo rejeita relatório diferente, índices repetidos ou rótulos desconhecidos.
As taxas usam só casos concluídos e mostram os pendentes. O programa não atribui
notas factuais automaticamente nem sobrescreve um arquivo de revisão existente.

## Documentos adversariais e fronteiras

As instruções do sistema foram separadas do conteúdo JSON não confiável da
pergunta/PDF na chamada ao provedor. Foram adicionados PDFs fictícios com comandos
maliciosos, IDs inventados, HTML e pedidos de segredo. [Veja os casos e limites](../evaluations/adversarial/README.md).

Os testes offline verificam extração/indexação, separação das regras, rejeição de
citações inexistentes e renderização segura. Eles **não** demonstram que um LLM
real sempre ignora injeções. Um teste preserva explicitamente a limitação: uma
resposta errada com ID válido ainda pode passar pela validação estrutural.
O cliente não recebe ferramentas de shell, banco ou envio de mensagens nesta
integração; não coloque segredos no contexto e não confunda prompt com autorização.

## Validação desta entrega

Suíte offline, testes de integração com PDF/SQLite/FAISS e testes Chromium com
respostas interceptadas são camadas distintas. Download real dos pesos foi
bloqueado pela política de rede do ambiente. Não foi feito benchmark novo com
embeddings nem geração real nesta rodada. A busca padrão foi preservada.
