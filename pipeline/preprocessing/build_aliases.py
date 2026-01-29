#!/usr/bin/env python3
"""

Creates ontology/aliases.json by querying EBI OLS4.
- Discovers candidate neuroscience terms using /api/search (rows/start paging).
- Pulls label + synonyms from term detail endpoints.
- Writes { normalized_alias: canonical_label } to ontology/aliases.json

OLS search API supports rows/start/groupField patterns. :contentReference[oaicite:2]{index=2}
OLS v2 endpoints support page/size and classes hierarchies. :contentReference[oaicite:3]{index=3}
Search responses include response/docs structure in common client code. :contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests


OLS_BASE = "https://www.ebi.ac.uk/ols4"

# v1-ish endpoints
SEARCH_ENDPOINT = f"{OLS_BASE}/api/search"
TERMS_ENDPOINT = f"{OLS_BASE}/api/terms"

# v2-ish endpoints (used for classes info, ontologies list)
V2_ONTOLOGIES_ENDPOINT = f"{OLS_BASE}/api/v2/ontologies"
V2_CLASSES_ENDPOINT_TMPL = f"{OLS_BASE}/api/v2/ontologies/{{ontology}}/classes/{{encoded_iri}}"


DEFAULT_ONTOLOGIES = [
    # good general coverage for neuro / bio
    "go",       # Gene Ontology
    "uberon",   # anatomy
    "cl",       # Cell Ontology
    "pr",       # Protein Ontology
    "chebi",    # chemicals/ions
    "hp",       # phenotypes
    "mondo",    # disease
]

NEURO_KEYWORDS = [
    # broad, intentionally high-recall
    "neuro", "neuron", "neuronal", "glia", "glial", "astrocyte", "microglia",
    "oligodendrocyte", "synapse", "synaptic", "axon", "axonal", "dendrite",
    "hippocampus", "cortex", "cortical", "thalamus", "amygdala", "cerebellum",
    "brain", "spinal cord", "ganglion", "nerve", "neuromuscular",
    "dopamine", "serotonin", "acetylcholine", "gaba", "glutamate",
    "ion channel", "neurotransmitter", "receptor", "action potential",
    "myelin", "blood brain barrier", "bbb",
]


def normalize_string(text: str) -> str:
    """Normalization (ie for Ca²⁺ -> ca2+)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold().strip()
    text = text.rstrip(".,;:!?-_")
    return text


def url_encode_iri(iri: str) -> str:
    # OLS expects IRIs URL-encoded in path segments.
    return urllib.parse.quote(iri, safe="")


def request_json(session: requests.Session, url: str, params: Optional[dict] = None, timeout: int = 30) -> dict:
    r = session.get(url, params=params, headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_search_docs(payload: dict) -> Tuple[List[dict], int]:
    """
    OLS /api/search often returns:
      { "response": { "numFound": ..., "docs": [...] } }
    (This shape is used by client implementations.) :contentReference[oaicite:5]{index=5}
    """
    resp = payload.get("response") or {}
    docs = resp.get("docs") or []
    num_found = int(resp.get("numFound") or 0)
    if not isinstance(docs, list):
        docs = []
    return docs, num_found


def extract_label(term_obj: dict) -> str:
    # common key in OLS term objects
    return term_obj.get("label") or ""


def extract_synonyms(term_obj: dict) -> List[str]:
    """
    Synonyms show up in different places depending on ontology and endpoint.
    pull from:
      - "synonyms": [...]
      - "annotation": { hasExactSynonym/hasRelatedSynonym/... : [...] }
    """
    syns: List[str] = []

    if isinstance(term_obj.get("synonyms"), list):
        syns.extend([s for s in term_obj["synonyms"] if isinstance(s, str)])

    ann = term_obj.get("annotation")
    if isinstance(ann, dict):
        for key in [
            "has_exact_synonym",
            "hasExactSynonym",
            "exact_synonym",
            "has_related_synonym",
            "hasRelatedSynonym",
            "related_synonym",
            "altLabel",
            "alternative_term",
            "abbreviation",
            "short_name",
            "shortName",
        ]:
            v = ann.get(key)
            if isinstance(v, list):
                syns.extend([x for x in v if isinstance(x, str)])
            elif isinstance(v, str):
                syns.append(v)

    # dedupe preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for s in syns:
        s2 = s.strip()
        if s2 and s2 not in seen:
            seen.add(s2)
            out.append(s2)
    return out


def term_detail_by_obo_id(session: requests.Session, ontology: str, obo_id: str) -> Optional[dict]:
    """
    /api/terms can be queried by ontology + obo_id (often easiest when available).
    """
    try:
        payload = request_json(session, TERMS_ENDPOINT, params={"ontology": ontology, "obo_id": obo_id})
    except requests.HTTPError:
        return None

    # shapes vary; try embedded terms
    if isinstance(payload, dict):
        emb = payload.get("_embedded")
        if isinstance(emb, dict) and isinstance(emb.get("terms"), list) and emb["terms"]:
            return emb["terms"][0]
        if isinstance(payload.get("terms"), list) and payload["terms"]:
            return payload["terms"][0]
        # sometimes it returns a single term-ish object
        if "label" in payload or "obo_id" in payload or "iri" in payload:
            return payload
    return None


def term_detail_by_iri_v2(session: requests.Session, ontology: str, iri: str) -> Optional[dict]:
    """
    /api/v2/ontologies/{ontology}/classes/{encoded_iri}
    v2 is referenced by OLS tooling and supports class hierarchy endpoints. :contentReference[oaicite:6]{index=6}
    """
    encoded = url_encode_iri(iri)
    url = V2_CLASSES_ENDPOINT_TMPL.format(ontology=ontology, encoded_iri=encoded)
    try:
        return request_json(session, url)
    except requests.HTTPError:
        return None


def iter_search_hits(
    session: requests.Session,
    ontology: str,
    query: str,
    rows: int,
    max_hits: int,
    polite_sleep_s: float,
) -> Iterable[dict]:
    """
    Page through /api/search using start/rows. :contentReference[oaicite:7]{index=7}
    """
    start = 0
    yielded = 0

    while yielded < max_hits:
        params = {
            "q": query,
            "ontology": ontology,
            "rows": rows,
            "start": start,
            # grouping reduces duplicates across fields; common in examples :contentReference[oaicite:8]{index=8}
            "groupField": "iri",
        }
        payload = request_json(session, SEARCH_ENDPOINT, params=params)
        docs, num_found = extract_search_docs(payload)

        if not docs:
            break

        for d in docs:
            yield d
            yielded += 1
            if yielded >= max_hits:
                break

        start += rows
        if start >= num_found:
            break

        if polite_sleep_s > 0:
            time.sleep(polite_sleep_s)


def looks_neuro(term_label: str, synonyms: List[str], keywords: List[str]) -> bool:
    hay = " ".join([term_label] + synonyms).casefold()
    return any(k.casefold() in hay for k in keywords)


def build_aliases(
    out_path: Path,
    ontologies: List[str],
    keywords: List[str],
    rows: int = 50,
    max_hits_per_keyword: int = 500,
    polite_sleep_s: float = 0.05,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Returns (alias_map, stats)
    """
    alias_map: Dict[str, str] = {}
    stats = {
        "ontologies": len(ontologies),
        "keywords": len(keywords),
        "search_docs_seen": 0,
        "terms_fetched": 0,
        "aliases_written": 0,
        "collisions_skipped": 0,
        "terms_skipped_not_neuro": 0,
    }

    seen_term_keys: Set[Tuple[str, str]] = set()  # (ontology, obo_id or iri)

    with requests.Session() as session:
        for onto in ontologies:
            for kw in keywords:
                for doc in iter_search_hits(
                    session=session,
                    ontology=onto,
                    query=kw,
                    rows=rows,
                    max_hits=max_hits_per_keyword,
                    polite_sleep_s=polite_sleep_s,
                ):
                    stats["search_docs_seen"] += 1
                    if stats["search_docs_seen"] % 50 == 0:
                        print(f"[progress] {stats['search_docs_seen']} docs seen, {stats['terms_fetched']} terms fetched, {len(alias_map)} aliases")

                    iri = doc.get("iri")
                    obo_id = doc.get("obo_id") or doc.get("short_form") or doc.get("shortForm")

                    key = (onto, obo_id or iri or "")
                    if not key[1] or key in seen_term_keys:
                        continue
                    seen_term_keys.add(key)

                    # fetch details (prefer obo_id if present)
                    term_obj = None
                    if obo_id:
                        term_obj = term_detail_by_obo_id(session, onto, obo_id)
                    if term_obj is None and iri:
                        term_obj = term_detail_by_iri_v2(session, onto, iri)

                    if term_obj is None:
                        continue

                    stats["terms_fetched"] += 1
                    label = extract_label(term_obj)
                    if not label:
                        continue

                    syns = extract_synonyms(term_obj)

                    # extra guard: keep only neuro-ish terms
                    if not looks_neuro(label, syns, keywords):
                        stats["terms_skipped_not_neuro"] += 1
                        continue

                    # write label + synonyms -> canonical label
                    for s in [label] + syns:
                        k = normalize_string(s)
                        if not k:
                            continue
                        if k in alias_map and alias_map[k] != label:
                            stats["collisions_skipped"] += 1
                            continue
                        alias_map[k] = label

    stats["aliases_written"] = len(alias_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(alias_map, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return alias_map, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ontology/aliases.json", help="Output path (default: ontology/aliases.json)")
    ap.add_argument("--ontologies", nargs="*", default=DEFAULT_ONTOLOGIES,
                    help=f"Ontology short names (default: {', '.join(DEFAULT_ONTOLOGIES)})")
    ap.add_argument("--rows", type=int, default=50, help="Search page size for /api/search (rows)")
    ap.add_argument("--max-hits-per-keyword", type=int, default=500,
                    help="Cap results per (ontology, keyword) to avoid huge runs")
    ap.add_argument("--sleep", type=float, default=0.05, help="Polite sleep between search pages (seconds)")
    ap.add_argument("--keywords-file", default=None,
                    help="Optional text file of keywords (one per line). If omitted, built-in neuro keywords are used.")
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.keywords_file:
        kws = []
        for line in Path(args.keywords_file).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                kws.append(s)
        keywords = kws if kws else NEURO_KEYWORDS
    else:
        keywords = NEURO_KEYWORDS

    alias_map, stats = build_aliases(
        out_path=out_path,
        ontologies=args.ontologies,
        keywords=keywords,
        rows=args.rows,
        max_hits_per_keyword=args.max_hits_per_keyword,
        polite_sleep_s=args.sleep,
    )

    print(f"Wrote {len(alias_map)} aliases -> {out_path}")
    print("Stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
