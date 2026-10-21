"""Evidence labels are scoped to one request, and validated against retrieved text."""

import json
import re

from app.errors import ServiceUnavailableError

NO_ANSWER = "Não encontrei essa informação nos documentos indexados."
NO_ANSWER_EN = "I couldn't find that information in the indexed documents."


def evidence_from_results(results, threshold):
    return [
        {
            "citation_id": f"S{number}",
            "filename": result["source"],
            "page": result.get("page"),
            "text": result["text"],
            "distance": round(result["distance"], 4),
            "relevance": round(1 / (1 + result["distance"]), 3),
        }
        for number, result in enumerate(
            (r for r in results if r["distance"] <= threshold), 1
        )
    ]


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
