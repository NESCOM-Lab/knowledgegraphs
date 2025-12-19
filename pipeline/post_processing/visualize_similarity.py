#!/usr/bin/env python3
"""
Visualize Node Similarity Distribution
=======================================

Creates a simple histogram showing similarity distribution with bins.

Usage:
    python visualize_similarity.py
    python visualize_similarity.py --save similarity.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from scipy.spatial.distance import pdist
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Visualize similarity distribution")
    parser.add_argument("--save", type=str, help="Save plot to file")
    args = parser.parse_args()

    # Load config
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not neo4j_password:
        print("ERROR: NEO4J_PASSWORD not set")
        sys.exit(1)

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

    # Extract nodes
    print("Extracting nodes...")
    query = """
    MATCH (n)
    WHERE NOT n:Document
    RETURN COALESCE(n.id, n.name, labels(n)[0]) AS name
    """

    with driver.session(database=neo4j_database) as session:
        result = session.run(query)
        names = [record["name"] for record in result if record["name"]]

    driver.close()

    if not names:
        print("No nodes found")
        return

    print(f"Extracted {len(names)} nodes")

    # Embed
    print("Embedding nodes...")
    embeddings_model = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    embeddings = []
    batch_size = 100
    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        batch_emb = embeddings_model.embed_documents(batch)
        embeddings.extend(batch_emb)
        if (i + batch_size) % 1000 == 0:
            print(f"  Progress: {i + batch_size}/{len(names)}")

    embeddings = np.array(embeddings)
    print(f"Embedded {len(names)} nodes")

    # Compute similarities
    print("Computing similarities...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    distances = pdist(normalized, metric='cosine')
    similarities = 1 - distances

    print(f"Computed {len(similarities):,} pairwise similarities")

    # Create histogram
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Define bins
    bins = [0.0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.0]
    bin_labels = ['0.0-0.70', '0.70-0.75', '0.75-0.80', '0.80-0.85',
                  '0.85-0.90', '0.90-0.95', '0.95-0.99', '0.99-1.0']

    # Count how many pairs fall in each bin
    counts, _ = np.histogram(similarities, bins=bins)

    # Create bar positions
    x_pos = np.arange(len(bin_labels))

    # Color the bars
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(counts)))
    bars = ax.bar(x_pos, counts, color=colors, edgecolor='black', alpha=0.8, width=0.8)

    # Add count labels on bars
    for i, (count, bar) in enumerate(zip(counts, bars)):
        height = bar.get_height()
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count):,}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlabel('Cosine Similarity Range', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Node Pairs', fontsize=13, fontweight='bold')
    ax.set_title(f'Similarity Distribution ({len(names):,} nodes, {len(similarities):,} pairs)',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"\nSaved to: {args.save}")

    plt.show()

    # Print stats
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total nodes: {len(names):,}")
    print(f"Total pairs: {len(similarities):,}")
    print(f"\nSimilarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
    print(f"Mean: {similarities.mean():.3f}")
    print(f"Median: {np.median(similarities):.3f}")
    print(f"\nBin counts:")
    for i, (count, low, high) in enumerate(zip(counts, edges[:-1], edges[1:])):
        pct = (count / len(similarities)) * 100
        print(f"  {low:.2f}-{high:.2f}: {int(count):,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
