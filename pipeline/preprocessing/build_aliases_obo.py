#!/usr/bin/env python3
"""
Build aliases.json from local OBO ontology files using pronto.
Much faster than API crawling - parses full ontologies locally.

Usage:
    python build_aliases_obo.py --out ontology/aliases.json

Downloads OBO files if not present, then extracts all term synonyms.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Set
import requests

try:
    import pronto
except ImportError:
    print("Install pronto: pip install pronto")
    raise

# OBO file URLs (use direct URLs that don't redirect)
OBO_SOURCES = {
    "go": "https://current.geneontology.org/ontology/go.obo",
    "uberon": "https://raw.githubusercontent.com/obophenotype/uberon/master/uberon.obo",
    "cl": "https://raw.githubusercontent.com/obophenotype/cell-ontology/master/cl.obo",
    "chebi": "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo",
    "hp": "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo",
    "mondo": "http://purl.obolibrary.org/obo/mondo.obo",
    "pr": "https://proconsortium.org/download/current/pro_reasoned.obo",
}

NEURO_KEYWORDS = [
    "calcium", "alzheimer's", "neuro", "neuron", "neuronal", "glia", "glial", "astrocyte", "microglia",
    "oligodendrocyte", "synapse", "synaptic", "axon", "axonal", "dendrite",
    "hippocampus", "cortex", "cortical", "thalamus", "amygdala", "cerebellum",
    "brain", "spinal cord", "ganglion", "nerve", "neuromuscular",
    "dopamine", "serotonin", "acetylcholine", "gaba", "glutamate",
    "ion channel", "neurotransmitter", "receptor", "action potential",
    "myelin", "blood brain barrier", "bbb",
]


def normalize_string(text: str) -> str:
    """Normalize text for matching (Ca2+ -> ca2+, etc.)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold().strip()
    text = text.rstrip(".,;:!?-_")
    return text


def looks_neuro(label: str, synonyms: List[str], keywords: List[str]) -> bool:
    """Check if term is neuroscience-related."""
    hay = " ".join([label] + synonyms).casefold()
    return any(k.casefold() in hay for k in keywords)


def download_obo(name: str, url: str, obo_dir: Path) -> Path:
    """Download OBO file if not present."""
    obo_path = obo_dir / f"{name}.obo"
    if obo_path.exists():
        # Verify it's not an HTML redirect page
        with open(obo_path, 'r', errors='ignore') as f:
            first_line = f.readline()
            if first_line.strip().startswith('<!DOCTYPE') or first_line.strip().startswith('<html'):
                print(f"[invalid] {name}.obo is HTML, re-downloading")
                obo_path.unlink()
            else:
                print(f"[skip] {name}.obo already exists")
                return obo_path

    print(f"[download] {name}.obo from {url}")
    obo_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=300, allow_redirects=True)
        resp.raise_for_status()
        obo_path.write_bytes(resp.content)
    except Exception as e:
        print(f"[error] Failed to download {name}.obo: {e}")
        return None
    return obo_path


def parse_obo(obo_path: Path, keywords: List[str], max_label_len: int) -> Dict[str, str]:
    """Parse OBO file and extract term -> synonym mappings."""
    print(f"[parse] {obo_path.name}")

    try:
        ont = pronto.Ontology(str(obo_path))
    except Exception as e:
        print(f"[error] Failed to parse {obo_path.name}: {e}")
        return {}

    alias_map: Dict[str, str] = {}
    terms_seen = 0
    terms_matched = 0

    for term in ont.terms():
        terms_seen += 1
        if terms_seen % 5000 == 0:
            print(f"  [progress] {terms_seen} terms processed, {len(alias_map)} aliases")

        label = term.name
        if not label:
            continue

        # Get all synonyms
        synonyms = [syn.description for syn in term.synonyms]

        # Filter to neuro-relevant terms
        if keywords and not looks_neuro(label, synonyms, keywords):
            continue

        terms_matched += 1

        # Add label and all synonyms -> canonical label
        for text in [label] + synonyms:
            k = normalize_string(text)
            if not k:
                continue
            if len(k) > max_label_len:
                continue
            # First canonical label wins (no overwrites)
            if k not in alias_map:
                alias_map[k] = label

    print(f"  [done] {terms_seen} terms, {terms_matched} matched, {len(alias_map)} aliases")
    return alias_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ontology/aliases.json", help="Output path")
    ap.add_argument("--obo-dir", default="obo_cache", help="Directory for OBO files")
    ap.add_argument("--ontologies", nargs="*", default=list(OBO_SOURCES.keys()),
                    help=f"Ontologies to process (default: {', '.join(OBO_SOURCES.keys())})")
    ap.add_argument("--no-filter", action="store_true",
                    help="Don't filter by neuro keywords (get ALL terms)")
    ap.add_argument("--max-label-len", type=int, default=60,
                    help="Skip terms longer than this (default: 60)")
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip downloading, use existing OBO files only")
    args = ap.parse_args()

    obo_dir = Path(args.obo_dir)
    out_path = Path(args.out)
    keywords = [] if args.no_filter else NEURO_KEYWORDS

    # Download OBO files
    obo_files: List[Path] = []
    for name in args.ontologies:
        if name not in OBO_SOURCES:
            print(f"[warn] Unknown ontology: {name}")
            continue

        obo_path = obo_dir / f"{name}.obo"
        if args.skip_download:
            if obo_path.exists():
                obo_files.append(obo_path)
            else:
                print(f"[skip] {name}.obo not found")
        else:
            result = download_obo(name, OBO_SOURCES[name], obo_dir)
            if result:
                obo_files.append(result)

    # Also check for OBO files in current directory
    for name in args.ontologies:
        local_obo = Path(f"{name}.obo")
        if local_obo.exists() and local_obo not in obo_files:
            print(f"[found] {local_obo} in current directory")
            obo_files.append(local_obo)

    # Parse all OBO files
    combined_aliases: Dict[str, str] = {}
    for obo_path in obo_files:
        aliases = parse_obo(obo_path, keywords, args.max_label_len)
        for k, v in aliases.items():
            if k not in combined_aliases:
                combined_aliases[k] = v

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined_aliases, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    print(f"\nWrote {len(combined_aliases)} aliases -> {out_path}")


if __name__ == "__main__":
    main()
