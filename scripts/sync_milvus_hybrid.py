#!/usr/bin/env python3
"""Synchronize Knowledge-Base-Chunks into the Milvus hybrid retrieval collection."""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.milvus_hybrid import MilvusHybridRetriever, collect_chunk_documents


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop and recreate the Milvus collection before syncing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the number of changed/deleted documents without upserting embeddings.",
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
    args = parser.parse_args()

    if args.milvus_uri:
        uri_path = Path(args.milvus_uri)
        if uri_path.suffix == ".db":
            uri_path.parent.mkdir(parents=True, exist_ok=True)
            settings.milvus_uri = str(uri_path.resolve())
            settings.milvus_token = ""
        else:
            settings.milvus_uri = args.milvus_uri

    if args.local_embeddings:
        settings.embedding_provider = "local_hash"

    documents = collect_chunk_documents()
    retriever = MilvusHybridRetriever()

    if args.drop:
        print(f"Dropping collection: {retriever.collection_name}")
        retriever.drop_collection()

    if args.dry_run:
        existing = await retriever.list_existing_documents()
        incoming = {doc.chunk_id: doc for doc in documents}
        changed = [
            doc
            for doc in documents
            if existing.get(doc.chunk_id) != doc.content_hash
        ]
        stale_ids = [chunk_id for chunk_id in existing if chunk_id not in incoming]
        print("Milvus hybrid sync dry run")
        print(f"documents: {len(documents)}")
        print(f"changed: {len(changed)}")
        print(f"deleted: {len(stale_ids)}")
        print(f"unchanged: {len(documents) - len(changed)}")
        return 0

    stats = await retriever.sync_documents(documents)
    print("Milvus hybrid sync complete")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
