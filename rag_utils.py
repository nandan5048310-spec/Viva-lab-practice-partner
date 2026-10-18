
"""Lightweight, in-memory retrieval for study material."""

import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of",
    "to", "in", "on", "at", "for", "and", "or", "it",
    "this", "that", "what", "which", "how", "why",
    "does", "do", "can", "you", "me", "explain",
    "describe", "define", "about", "from", "with",
    "its", "as", "by", "be", "between", "tell",
}


def _tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]


def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into ordered, overlapping chunks."""
    if not isinstance(text, str):
        raise TypeError("Study material must be text.")

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Prefer ending at a nearby sentence boundary.
            window = text[start:end]
            boundaries = [
                match.end()
                for match in re.finditer(r"[.!?]\s+", window)
                if match.end() >= chunk_size // 2
            ]
            if boundaries:
                end = start + boundaries[-1]

        content = text[start:end].strip()

        if content:
            chunks.append({
                "id": f"chunk_{len(chunks) + 1}",
                "text": content,
            })

        if end >= len(text):
            break

        # Guaranteed forward progress, even for very short sentences.
        start = max(start + 1, end - overlap)

    return chunks


def retrieve_context(query, study_text, top_k=3):
    """Return relevant chunks ranked using keyword-weighted TF-IDF."""
    if not isinstance(query, str) or not isinstance(study_text, str):
        raise TypeError("Query and study material must be text.")

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    query_terms = set(_tokens(query))
    if not query_terms:
        return []

    chunks = chunk_text(study_text)
    if not chunks:
        return []

    documents = [Counter(_tokens(c["text"])) for c in chunks]
    n = len(documents)

    document_frequency = Counter()
    for counts in documents:
        document_frequency.update(counts.keys())

    results = []

    for chunk, counts in zip(chunks, documents):
        matching = query_terms.intersection(counts)
        if not matching:
            continue

        score = sum(
            (1 + math.log(counts[word]))
            * math.log(1 + (n + 1) / (document_frequency[word] + 1))
            for word in matching
        )

        # Reward coverage of several query terms.
        coverage = len(matching) / len(query_terms)
        score *= coverage

        # Reject weak matches with low query-term coverage.
        if coverage < 0.25:
            continue

        results.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "score": round(score, 4),
        })

    results.sort(key=lambda item: (-item["score"], item["id"]))

    return results[:top_k]
