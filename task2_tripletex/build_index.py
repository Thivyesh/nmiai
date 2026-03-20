"""One-time script to build the semantic search index from the OpenAPI spec.

Run this once:  uv run python -m task2_tripletex.build_index

Produces:
  - task2_tripletex/search_index.npz  (embeddings + metadata)
"""

import json
from pathlib import Path

import numpy as np
import requests

SPEC_PATH = Path(__file__).parent / "openapi_cache.json"
INDEX_PATH = Path(__file__).parent / "search_index.npz"
TEI_URL = "http://localhost:8080"


def embed(texts: list[str]) -> np.ndarray:
    truncated = [t[:2000] for t in texts]  # Stay within 2048 token limit
    resp = requests.post(f"{TEI_URL}/embed", json={"inputs": truncated}, timeout=60)
    resp.raise_for_status()
    return np.array(resp.json())


def main():
    with open(SPEC_PATH) as f:
        spec = json.load(f)

    paths = spec.get("paths", {})
    entries = []  # (path, method, summary)
    texts = []  # searchable text for embedding

    for path, methods in paths.items():
        for method, details in methods.items():
            if method not in ("get", "post", "put", "delete"):
                continue
            summary = details.get("summary", "")
            param_names = " ".join(
                p.get("name", "") for p in details.get("parameters", [])
            )
            # Rich text for embedding
            text = f"{method.upper()} {path} — {summary} {param_names}"
            entries.append((path, method, summary))
            texts.append(text)

    print(f"Embedding {len(texts)} endpoints...")

    # Embed in batches (small batches to stay within TEI limits)
    all_embeddings = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.append(embed(batch))
        print(f"  Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    embeddings = np.vstack(all_embeddings)

    # Save as npz with metadata
    np.savez_compressed(
        INDEX_PATH,
        embeddings=embeddings,
        paths=np.array([e[0] for e in entries]),
        methods=np.array([e[1] for e in entries]),
        summaries=np.array([e[2] for e in entries]),
    )

    print(f"Saved index to {INDEX_PATH} ({embeddings.shape[0]} endpoints, {embeddings.shape[1]}d)")


if __name__ == "__main__":
    main()
