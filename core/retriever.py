import numpy as np
from core.embedder import embed_texts
from core.vectordb import load_db
from config.settings import TOP_K


def retrieve(query, use_case=None):
    index, metadata = load_db()
    if index is None or not metadata:
        return []

    # Embed query
    query_vec = embed_texts([query])

    # 🔥 Over-fetch so we can refine later
    distances, indices = index.search(
        np.array(query_vec).astype("float32"),
        TOP_K * 3
    )

    query_lower = query.lower()

    # 🔥 Detect intent
    is_api_query = "api" in query_lower or "endpoint" in query_lower

    api_keywords = [
        "api", "endpoint", "post", "get", "put", "delete", "/api"
    ]

    results = []
    seen = set()

    for dist, idx in zip(distances[0], indices[0]):
        if idx >= len(metadata):
            continue

        meta = metadata[idx]
        text_lower = meta["text"].lower()

        # 🔹 Filter by use case
        if use_case and meta.get("use_case") != use_case:
            continue

        # 🔥 Intent-based filtering (THIS IS THE KEY)
        if is_api_query:
            if not any(k in text_lower for k in api_keywords):
                continue

        # 🔹 Deduplicate identical chunks
        key = (meta["document"], meta["text"])
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "text": meta["text"],
            "document": meta["document"],
            "use_case": meta.get("use_case"),
            "score": float(dist)
        })

        if len(results) >= TOP_K:
            break

    return results
