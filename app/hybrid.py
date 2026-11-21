"""BM25 + reciprocal-rank fusion, opt-in; ranks are not answer confidence."""

from collections import Counter
import math
import re
import unicodedata

STOPWORDS = set(
    "a ao aos as o os um uma uns umas de da das do dos em no nos na nas para por com e ou que qual quais como quando onde quem quanto quantos what which how when where who is are the of to for in on at and or does do pdf".split()
)


def tokens(text):
    normalized = "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    )
    return [
        term
        for term in re.findall(r"[^\W_]+(?:[-_][^\W_]+)*", normalized)
        if term not in STOPWORDS
    ]


def lexical_rank(question, chunks, limit=20):
    query = set(tokens(question))
    if not query or not chunks:
        return []
    documents = [
        Counter(tokens(chunk["source"] + " " + chunk["text"])) for chunk in chunks
    ]
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / len(lengths) or 1
    frequencies = {term: sum(term in doc for doc in documents) for term in query}
    rows = []
    for index, (doc, length) in enumerate(zip(documents, lengths)):
        score = 0.0
        for term in sorted(query):
            tf = doc[term]
            if tf:
                idf = math.log(
                    1
                    + (len(documents) - frequencies[term] + 0.5)
                    / (frequencies[term] + 0.5)
                )
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            rows.append({**chunks[index], "chunk_id": index, "lexical_score": score})
    return sorted(rows, key=lambda row: (-row["lexical_score"], row["chunk_id"]))[
        :limit
    ]


def fuse(semantic, lexical, top_k):
    """Equal RRF weights, constant 60. Inputs use IDs from the SAME snapshot."""
    rows = {}
    for channel, ranked in [("semantic", semantic), ("lexical", lexical)]:
        for rank, item in enumerate(ranked, 1):
            key = item["chunk_id"]
            row = rows.setdefault(
                key, {**item, "fusion_score": 0.0, "retrieval_mode": "hybrid"}
            )
            row.update(
                {
                    k: v
                    for k, v in item.items()
                    if k not in ("fusion_score", "retrieval_mode")
                    and (v is not None or k not in row)
                }
            )
            row["fusion_score"] += 1 / (60 + rank)
            row[channel + "_rank"] = rank
    return sorted(
        rows.values(), key=lambda row: (-row["fusion_score"], row["chunk_id"])
    )[:top_k]
