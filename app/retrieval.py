"""Evidence labels are scoped to one request, and validated against retrieved text."""

import json
import re

from app.errors import ServiceUnavailableError

NO_ANSWER = "Não encontrei essa informação nos documentos indexados."
NO_ANSWER_EN = "I couldn't find that information in the indexed documents."


def accepts_result(result, threshold):
    distance = result.get("distance")
    return (distance is not None and distance <= threshold) or (
        result.get("retrieval_mode") == "hybrid" and result.get("lexical_score", 0) > 0
    )


def evidence_from_results(results, threshold):
    sources = []
    for result in results:
        if not accepts_result(result, threshold):
            continue
        distance = result.get("distance")
        row = {
            "citation_id": f"S{len(sources) + 1}",
            "filename": result["source"],
            "page": result.get("page"),
            "text": result["text"],
            "distance": round(distance, 4) if distance is not None else None,
            "relevance": round(1 / (1 + distance), 3) if distance is not None else None,
        }
        if result.get("retrieval_mode") == "hybrid":
            row.update(
                retrieval_mode="hybrid",
                lexical_score=result.get("lexical_score", 0),
                fusion_score=result["fusion_score"],
                acceptance_reason="semantic_distance"
                if distance is not None and distance <= threshold
                else "lexical_match",
            )
        sources.append(row)
    return sources


def make_context(sources):
    return "\n\n---\n\n".join(
        f"[{source['citation_id']}] Document: {json.dumps(source['filename'], ensure_ascii=False)}; "
        f"page: {source['page']}\n{source['text']}"
        for source in sources
    )


def cited_sources(answer, sources):
    if answer.strip() in {NO_ANSWER, NO_ANSWER_EN}:
        return []
    labels = set(re.findall(r"\[(S\d+)\]", answer))
    available = {source["citation_id"] for source in sources}
    if not labels or not labels <= available:
        raise ServiceUnavailableError(
            "The generated answer has missing or invalid source citations. Please try again."
        )
    return [source for source in sources if source["citation_id"] in labels]
