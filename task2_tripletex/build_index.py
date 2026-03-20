"""One-time script to build semantic search indices.

Run this once:  uv run python -m task2_tripletex.build_index

Produces:
  - task2_tripletex/search_index.npz    (API endpoint embeddings)
  - task2_tripletex/pattern_index.npz   (task pattern embeddings)
  - task2_tripletex/devdocs_index.npz   (Tripletex developer docs embeddings)
"""

import json
from pathlib import Path

import numpy as np
import requests

from task2_tripletex.task_patterns import TASK_PATTERNS

SPEC_PATH = Path(__file__).parent / "openapi_cache.json"
API_INDEX_PATH = Path(__file__).parent / "search_index.npz"
PATTERN_INDEX_PATH = Path(__file__).parent / "pattern_index.npz"
TEI_URL = "http://localhost:8080"


def embed(texts: list[str]) -> np.ndarray:
    truncated = [t[:2000] for t in texts]
    resp = requests.post(f"{TEI_URL}/embed", json={"inputs": truncated}, timeout=60)
    resp.raise_for_status()
    return np.array(resp.json())


def embed_batched(texts: list[str], batch_size: int = 8) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.append(embed(batch))
        print(f"  Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
    return np.vstack(all_embeddings)


def build_api_index():
    """Build embeddings for API endpoints."""
    with open(SPEC_PATH) as f:
        spec = json.load(f)

    paths = spec.get("paths", {})
    entries = []
    texts = []

    for path, methods in paths.items():
        for method, details in methods.items():
            if method not in ("get", "post", "put", "delete"):
                continue
            summary = details.get("summary", "")
            param_names = " ".join(
                p.get("name", "") for p in details.get("parameters", [])
            )
            text = f"{method.upper()} {path} — {summary} {param_names}"
            entries.append((path, method, summary))
            texts.append(text)

    print(f"Embedding {len(texts)} API endpoints...")
    embeddings = embed_batched(texts)

    np.savez_compressed(
        API_INDEX_PATH,
        embeddings=embeddings,
        paths=np.array([e[0] for e in entries]),
        methods=np.array([e[1] for e in entries]),
        summaries=np.array([e[2] for e in entries]),
    )
    print(f"Saved API index: {embeddings.shape[0]} endpoints, {embeddings.shape[1]}d")


def build_pattern_index():
    """Build embeddings for task patterns.

    Each section (## TASK: ...) becomes a searchable chunk.
    We also create extra embeddings for trigger phrases and multilingual terms.
    """
    sections = []
    current_title = ""
    current_lines = []

    for line in TASK_PATTERNS.split("\n"):
        if line.startswith("## TASK:") or line.startswith("## GENERAL"):
            if current_title and current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line.replace("## ", "").strip()
            current_lines = [line]
        elif line.startswith("## MULTILINGUAL"):
            if current_title and current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line.replace("## ", "").strip()
            current_lines = [line]
        elif current_title:
            current_lines.append(line)

    if current_title and current_lines:
        sections.append((current_title, "\n".join(current_lines)))

    # Create searchable texts — include title, triggers, and key phrases
    titles = []
    texts = []
    contents = []

    for title, content in sections:
        titles.append(title)
        contents.append(content)

        # Build rich searchable text from the section
        search_text = f"{title}\n"

        # Extract trigger line if present
        for line in content.split("\n"):
            if line.startswith("Trigger:"):
                search_text += line + "\n"
            elif line.startswith("Checks:"):
                search_text += line + "\n"

        # Add the first few lines of content for context
        search_text += "\n".join(content.split("\n")[:10])
        texts.append(search_text)

    # Add extra embeddings for common multilingual search queries
    extra_queries = [
        ("TASK: Create Employee", "opprett ansatt kontoadministrator administrator employee"),
        ("TASK: Create Employee", "crear empleado administrador employee admin"),
        ("TASK: Create Employee", "erstellen Mitarbeiter Administrator Kontoadministrator"),
        ("TASK: Create Customer", "opprett kunde customer klient client organization"),
        ("TASK: Create Customer", "crear cliente organisasjonsnummer org.nr"),
        ("TASK: Create Product", "opprett produkt product produktnummer price pris"),
        ("TASK: Create Invoice", "opprett faktura invoice rechnung factura betaling payment"),
        ("TASK: Create Invoice", "registrer betaling register payment zahlung registrieren pago"),
        ("TASK: Create Credit Note", "kreditnota credit note nota de crédito Gutschrift"),
        ("TASK: Create Travel Expense", "reiseregning travel expense nota de gastos Reisekostenabrechnung"),
        ("TASK: Create Travel Expense", "diett per diem dietas dagpenger utgift expense gasto"),
        ("TASK: Create Project", "opprett prosjekt project proyecto projet"),
        ("TASK: Create Department", "opprett avdeling department departamento abteilung"),
        ("TASK: Delete Entity", "slett delete eliminar löschen fjern remove"),
        ("TASK: Update Employee", "oppdater endre update actualizar ändern modify"),
        ("TASK: Register Payment", "registrer betaling payment pago zahlung registrieren"),
        ("TASK: Create Order", "opprett ordre order bestilling pedido Bestellung"),
        ("TASK: Create Contact", "opprett kontakt contact kontaktperson contacto"),
        ("GENERAL RULES", "general rules regler scoring field check verification"),
        ("MULTILINGUAL FIELD MAPPING", "field mapping felt oversettelse translation telefon phone email"),
    ]

    for target_title, query_text in extra_queries:
        # Find the matching section
        idx = next((i for i, t in enumerate(titles) if target_title in t), None)
        if idx is not None:
            titles.append(titles[idx])
            contents.append(contents[idx])
            texts.append(query_text)

    print(f"Embedding {len(texts)} pattern chunks...")
    embeddings = embed_batched(texts)

    np.savez_compressed(
        PATTERN_INDEX_PATH,
        embeddings=embeddings,
        titles=np.array(titles),
        contents=np.array(contents),
    )
    print(f"Saved pattern index: {embeddings.shape[0]} chunks, {embeddings.shape[1]}d")


DEVDOCS_PATH = Path(__file__).parent / "developer_docs.json"
DEVDOCS_INDEX_PATH = Path(__file__).parent / "devdocs_index.npz"


def build_devdocs_index():
    """Build embeddings for Tripletex developer documentation."""
    if not DEVDOCS_PATH.exists():
        print("developer_docs.json not found — skipping devdocs index")
        return

    with open(DEVDOCS_PATH) as f:
        docs = json.load(f)

    titles = []
    texts = []
    contents = []

    for doc in docs:
        if not doc["content"] or len(doc["content"]) < 50:
            continue
        titles.append(doc["title"])
        contents.append(doc["content"])
        # Searchable text: title + first 500 chars of content
        texts.append(f"{doc['title']}\n{doc['content'][:500]}")

    print(f"Embedding {len(texts)} developer docs...")
    embeddings = embed_batched(texts)

    np.savez_compressed(
        DEVDOCS_INDEX_PATH,
        embeddings=embeddings,
        titles=np.array(titles),
        contents=np.array(contents),
    )
    print(f"Saved devdocs index: {embeddings.shape[0]} docs, {embeddings.shape[1]}d")


def main():
    build_api_index()
    print()
    build_pattern_index()
    print()
    build_devdocs_index()


if __name__ == "__main__":
    main()
