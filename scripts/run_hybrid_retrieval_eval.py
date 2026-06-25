#!/usr/bin/env python3
"""Run golden-query acceptance checks for Milvus hybrid retrieval."""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.milvus_hybrid import HybridHit, MilvusHybridRetriever


@dataclass(frozen=True)
class GoldenQuery:
    name: str
    query: str
    expected_fragments: List[str]
    min_hits_top5: int = 1


GOLDEN_QUERIES = [
    GoldenQuery(
        name="model specs",
        query="SW-2100 technical specifications AMOLED 72 hour battery IP68 price",
        expected_fragments=["SW-2100-Flagship"],
    ),
    GoldenQuery(
        name="price",
        query="smartwatch priced at 999 dollars sports TFT display",
        expected_fragments=["SW-1500-Sport"],
    ),
    GoldenQuery(
        name="waterproof higher than IP67",
        query="waterproof rating higher than IP67 IP68 IP69K products",
        expected_fragments=["SW-2100-Flagship", "SW-1500-Sport", "SW-2200-Premium"],
        min_hits_top5=2,
    ),
    GoldenQuery(
        name="2023 all regions revenue",
        query="2023 annual revenue all regions retail stores East South North Southwest",
        expected_fragments=[
            "2023-Market-Layout/East-China-Region",
            "2023-Market-Layout/South-China-Region",
            "2023-Market-Layout/North-China-Region",
            "2023-Market-Layout/Southwest-Region",
        ],
        min_hits_top5=4,
    ),
    GoldenQuery(
        name="supplier display",
        query="supplier supplies AMOLED OLED screens eight year partnership",
        expected_fragments=["Display-Supplier-CrystalVision"],
    ),
    GoldenQuery(
        name="supplier audio",
        query="speaker microphone modules audio supplier acoustic engineering",
        expected_fragments=["Audio-Supplier-SoundTech"],
    ),
    GoldenQuery(
        name="research display",
        query="research team flexible screens micro display micro OLED micro LED",
        expected_fragments=["Display-Tech-Team"],
    ),
    GoldenQuery(
        name="negation display types",
        query="display types besides AMOLED and OLED LCD TFT",
        expected_fragments=["SW-1800-Business", "SW-1500-Sport"],
        min_hits_top5=2,
    ),
    GoldenQuery(
        name="extreme earbuds battery",
        query="Bluetooth audio device longest battery life forty hours",
        expected_fragments=["AE-Max-Master"],
    ),
    GoldenQuery(
        name="multi hop market comparison",
        query="compare 2024 and 2023 South China annual revenue retail stores growth",
        expected_fragments=[
            "2024-Market-Layout/South-China-Region",
            "2023-Market-Layout/South-China-Region",
        ],
        min_hits_top5=2,
    ),
]


def hit_count(hits: List[HybridHit], expected_fragments: List[str], top_n: int = 5) -> int:
    matched = set()
    for hit in hits[:top_n]:
        haystack = f"{hit.path}\n{hit.source_path}"
        for fragment in expected_fragments:
            if fragment in haystack:
                matched.add(fragment)
    return len(matched)


def format_hits(hits: List[HybridHit], top_n: int = 5) -> str:
    return ", ".join(hit.path for hit in hits[:top_n]) or "<none>"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=8, help="Number of results to request.")
    parser.add_argument(
        "--skip-single-comparison",
        action="store_true",
        help="Only verify hybrid top5 hits; skip dense-only and sparse-only comparisons.",
    )
    parser.add_argument(
        "--milvus-uri",
        help=(
            "Override the configured Milvus URI. Useful for Milvus Lite local files, "
            "for example .cache/deep_rag_milvus.db."
        ),
    )
    parser.add_argument(
        "--local-embeddings",
        action="store_true",
        help="Use deterministic local hash embeddings for development/CI validation.",
    )
    parser.add_argument(
        "--ranker",
        choices=["rrf", "weighted"],
        help="Override the hybrid ranker for this evaluation run.",
    )
    parser.add_argument("--dense-weight", type=float, help="Dense weight for weighted ranker.")
    parser.add_argument("--sparse-weight", type=float, help="Sparse weight for weighted ranker.")
    args = parser.parse_args()

    if args.milvus_uri:
        uri_path = Path(args.milvus_uri)
        if uri_path.suffix == ".db":
            settings.milvus_uri = str(uri_path.resolve())
            settings.milvus_token = ""
        else:
            settings.milvus_uri = args.milvus_uri

    if args.local_embeddings:
        settings.embedding_provider = "local_hash"

    if args.ranker:
        settings.hybrid_ranker = args.ranker
    if args.dense_weight is not None:
        settings.hybrid_dense_weight = args.dense_weight
    if args.sparse_weight is not None:
        settings.hybrid_sparse_weight = args.sparse_weight

    retriever = MilvusHybridRetriever()
    failures = []

    for golden in GOLDEN_QUERIES:
        hybrid = await retriever.search(golden.query, top_k=args.top_k)
        hybrid_hits = hit_count(hybrid, golden.expected_fragments)
        print(f"[{golden.name}] hybrid top5 hits={hybrid_hits}: {format_hits(hybrid)}")

        if hybrid_hits < golden.min_hits_top5:
            failures.append(
                f"{golden.name}: hybrid top5 expected >= {golden.min_hits_top5}, got {hybrid_hits}"
            )

        if not args.skip_single_comparison:
            dense = await retriever.search_dense(golden.query, top_k=args.top_k)
            sparse = await retriever.search_sparse(golden.query, top_k=args.top_k)
            dense_hits = hit_count(dense, golden.expected_fragments)
            sparse_hits = hit_count(sparse, golden.expected_fragments)
            print(f"  dense top5 hits={dense_hits}: {format_hits(dense)}")
            print(f"  sparse top5 hits={sparse_hits}: {format_hits(sparse)}")

            if hybrid_hits < max(dense_hits, sparse_hits):
                failures.append(
                    f"{golden.name}: hybrid top5 hits {hybrid_hits} weaker than "
                    f"dense/sparse max {max(dense_hits, sparse_hits)}"
                )

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll hybrid retrieval acceptance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
