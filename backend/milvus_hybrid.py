import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from backend.config import settings
from backend.embedding_provider import EmbeddingProvider


@dataclass(frozen=True)
class HybridDocument:
    chunk_id: str
    path: str
    source_path: str
    search_text: str
    content_hash: str
    mtime: float


@dataclass(frozen=True)
class HybridHit:
    rank: int
    path: str
    source_path: str
    score: Optional[float]
    chunk_id: Optional[str] = None


def document_id_for_path(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_path_for_chunk(chunk_path: Path, chunks_base: Path) -> str:
    relative = chunk_path.relative_to(chunks_base)
    if re.fullmatch(r"\d+-\d+\.md", chunk_path.name) and chunk_path.parent != chunks_base:
        source = chunk_path.parent.parent / f"{chunk_path.parent.name}.md"
        return source.relative_to(chunks_base).as_posix()
    return relative.as_posix()


def load_summary_cache(summary_file: Path) -> Dict:
    summary_demo = summary_file.parent / "summary_demo.json"
    if not summary_demo.exists():
        return {}
    with open(summary_demo, "r", encoding="utf-8") as f:
        return json.load(f)


def summary_for_chunk(relative_path: str, source_path: str, summary_cache: Dict) -> str:
    entry = summary_cache.get(source_path) or summary_cache.get(relative_path)
    if isinstance(entry, str):
        return entry
    if not isinstance(entry, list):
        return ""

    match = re.fullmatch(r".*/(\d+)-(\d+)\.md", relative_path)
    if not match:
        return ""

    start, end = int(match.group(1)), int(match.group(2))
    for chunk in entry:
        if chunk.get("start") == start and chunk.get("end") == end:
            return chunk.get("summary", "")
    return ""


def collect_chunk_documents(
    chunks_base: Path = None,
    summary_file: Path = None,
    max_text_length: int = None,
) -> List[HybridDocument]:
    base = Path(chunks_base or settings.knowledge_base_chunks)
    summary_path = Path(summary_file or settings.knowledge_base_file_summary)
    max_length = max_text_length or settings.milvus_max_text_length

    if not base.exists():
        raise RuntimeError(f"Knowledge base chunks path not found: {base}")

    summary_cache = load_summary_cache(summary_path)
    documents: List[HybridDocument] = []

    for md_file in sorted(base.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        rel_path = md_file.relative_to(base).as_posix()
        source_path = source_path_for_chunk(md_file, base)
        summary = summary_for_chunk(rel_path, source_path, summary_cache)
        search_text = f"{rel_path}\n{summary}\n\n{text}".strip()
        if len(search_text) > max_length:
            search_text = search_text[:max_length]

        documents.append(
            HybridDocument(
                chunk_id=document_id_for_path(rel_path),
                path=rel_path,
                source_path=source_path,
                search_text=search_text,
                content_hash=content_hash(text),
                mtime=md_file.stat().st_mtime,
            )
        )

    return documents


def clamp_top_k(top_k: Optional[int]) -> int:
    if top_k is None:
        return settings.hybrid_top_k
    return max(1, min(int(top_k), settings.hybrid_max_top_k))


def escape_expr_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_path_filter(file_paths: Optional[List[str]]) -> Optional[str]:
    terms = []

    for raw_path in file_paths or []:
        raw = (raw_path or "").strip().replace("\\", "/")
        if raw in ("", ".", "/"):
            return None

        path = raw.lstrip("/")
        if ".." in Path(path).parts:
            raise ValueError(f"Illegal path in retrieval filter: {raw_path}")

        if path.endswith("/"):
            prefix = escape_expr_string(path.rstrip("/") + "/")
            terms.append(f'path like "{prefix}%"')
            terms.append(f'source_path like "{prefix}%"')
        else:
            exact = escape_expr_string(path)
            terms.append(f'path == "{exact}"')
            terms.append(f'source_path == "{exact}"')

    if not terms:
        return None
    return "(" + " or ".join(terms) + ")"


class MilvusHybridRetriever:
    def __init__(self, client=None, embedder: EmbeddingProvider = None):
        self.client = client
        self.embedder = embedder or EmbeddingProvider()
        self.collection_name = settings.milvus_collection

    def _import_pymilvus(self):
        try:
            from pymilvus import (
                AnnSearchRequest,
                DataType,
                Function,
                FunctionType,
                MilvusClient,
                RRFRanker,
                WeightedRanker,
            )
        except ImportError as exc:
            raise RuntimeError(
                "pymilvus is not installed. Install dependencies with "
                "`pip install -r requirements.txt` before using Milvus hybrid retrieval."
            ) from exc

        return {
            "AnnSearchRequest": AnnSearchRequest,
            "DataType": DataType,
            "Function": Function,
            "FunctionType": FunctionType,
            "MilvusClient": MilvusClient,
            "RRFRanker": RRFRanker,
            "WeightedRanker": WeightedRanker,
        }

    def _client(self):
        if self.client is None:
            milvus = self._import_pymilvus()
            self._patch_milvus_lite_windows_manifest()
            kwargs = {"uri": settings.milvus_uri}
            if settings.milvus_token:
                kwargs["token"] = settings.milvus_token
            self.client = milvus["MilvusClient"](**kwargs)
        return self.client

    def _patch_milvus_lite_windows_manifest(self) -> None:
        """Milvus Lite on Windows may save manifests with os.rename.

        Windows fails when the target manifest already exists. os.replace has the
        atomic overwrite behavior used by POSIX rename, so patch only the local
        Lite manifest module and only for .db URIs on Windows.
        """
        if os.name != "nt" or not str(settings.milvus_uri).lower().endswith(".db"):
            return

        try:
            import milvus_lite.storage.manifest as manifest_module
        except Exception:
            return

        if getattr(manifest_module.os, "rename", None) is not os.replace:
            manifest_module.os.rename = os.replace

    def ensure_collection(self) -> None:
        milvus = self._import_pymilvus()
        client = self._client()
        if client.has_collection(self.collection_name):
            self._load_collection()
            return

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="chunk_id",
            datatype=milvus["DataType"].VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(field_name="path", datatype=milvus["DataType"].VARCHAR, max_length=1024)
        schema.add_field(
            field_name="source_path",
            datatype=milvus["DataType"].VARCHAR,
            max_length=1024,
        )
        schema.add_field(
            field_name="search_text",
            datatype=milvus["DataType"].VARCHAR,
            max_length=settings.milvus_max_text_length,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=milvus["DataType"].FLOAT_VECTOR,
            dim=settings.embedding_dim,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=milvus["DataType"].SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name="content_hash",
            datatype=milvus["DataType"].VARCHAR,
            max_length=64,
        )
        schema.add_field(field_name="mtime", datatype=milvus["DataType"].DOUBLE)
        schema.add_function(
            milvus["Function"](
                name="search_text_bm25",
                input_field_names=["search_text"],
                output_field_names=["sparse_vector"],
                function_type=milvus["FunctionType"].BM25,
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type=settings.milvus_dense_index_type,
            metric_type=settings.milvus_dense_metric_type,
            params={
                "M": settings.milvus_hnsw_m,
                "efConstruction": settings.milvus_hnsw_ef_construction,
            },
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"bm25_k1": settings.milvus_bm25_k1, "bm25_b": settings.milvus_bm25_b},
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        self._load_collection()

    def _load_collection(self) -> None:
        client = self._client()
        try:
            client.load_collection(self.collection_name)
        except Exception as exc:
            message = str(exc).lower()
            if "loaded" not in message and "already" not in message:
                raise

    def drop_collection(self) -> None:
        client = self._client()
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)

    async def list_existing_documents(self) -> Dict[str, str]:
        client = self._client()
        if not client.has_collection(self.collection_name):
            return {}

        rows = await asyncio.to_thread(
            client.query,
            collection_name=self.collection_name,
            filter='chunk_id != ""',
            output_fields=["chunk_id", "content_hash"],
            limit=16384,
        )
        return {
            row["chunk_id"]: row.get("content_hash", "")
            for row in rows
            if row.get("chunk_id")
        }

    async def sync_documents(self, documents: List[HybridDocument]) -> Dict[str, int]:
        self.ensure_collection()
        existing = await self.list_existing_documents()
        incoming = {doc.chunk_id: doc for doc in documents}
        changed = [
            doc
            for doc in documents
            if existing.get(doc.chunk_id) != doc.content_hash
        ]
        stale_ids = [chunk_id for chunk_id in existing if chunk_id not in incoming]

        upserted = 0
        for batch in _batches(changed, settings.embedding_batch_size):
            vectors = await self.embedder.embed_texts([doc.search_text for doc in batch])
            rows = [
                {
                    "chunk_id": doc.chunk_id,
                    "path": doc.path,
                    "source_path": doc.source_path,
                    "search_text": doc.search_text,
                    "content_hash": doc.content_hash,
                    "mtime": doc.mtime,
                    "dense_vector": vector,
                }
                for doc, vector in zip(batch, vectors)
            ]
            await asyncio.to_thread(
                self._client().upsert,
                collection_name=self.collection_name,
                data=rows,
            )
            upserted += len(rows)

        deleted = 0
        for batch in _batches(stale_ids, 128):
            ids = ", ".join(f'"{escape_expr_string(chunk_id)}"' for chunk_id in batch)
            await asyncio.to_thread(
                self._client().delete,
                collection_name=self.collection_name,
                filter=f"chunk_id in [{ids}]",
            )
            deleted += len(batch)

        return {
            "documents": len(documents),
            "upserted": upserted,
            "deleted": deleted,
            "unchanged": len(documents) - upserted,
        }

    async def search(
        self,
        query: str,
        file_paths: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[HybridHit]:
        return await self._search(query, file_paths=file_paths, top_k=top_k, modes=("dense", "sparse"))

    async def search_dense(
        self,
        query: str,
        file_paths: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[HybridHit]:
        return await self._search(query, file_paths=file_paths, top_k=top_k, modes=("dense",))

    async def search_sparse(
        self,
        query: str,
        file_paths: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[HybridHit]:
        return await self._search(query, file_paths=file_paths, top_k=top_k, modes=("sparse",))

    async def _search(
        self,
        query: str,
        file_paths: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        modes: tuple = ("dense", "sparse"),
    ) -> List[HybridHit]:
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("Milvus hybrid retrieval requires a non-empty query.")

        limit = clamp_top_k(top_k)
        path_filter = build_path_filter(file_paths)
        self._load_collection()

        milvus = self._import_pymilvus()
        req_kwargs = {"expr": path_filter} if path_filter else {}
        requests = []

        if "dense" in modes:
            dense_vector = (await self.embedder.embed_texts([clean_query]))[0]
            requests.append(
                milvus["AnnSearchRequest"](
                    data=[dense_vector],
                    anns_field="dense_vector",
                    param={
                        "metric_type": settings.milvus_dense_metric_type,
                        "params": {"ef": settings.milvus_hnsw_ef},
                    },
                    limit=limit,
                    **req_kwargs,
                )
            )

        if "sparse" in modes:
            requests.append(
                milvus["AnnSearchRequest"](
                    data=[clean_query],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25", "params": {}},
                    limit=limit,
                    **req_kwargs,
                )
            )

        if not requests:
            raise ValueError("At least one Milvus search mode is required.")

        ranker = self._build_ranker(milvus, modes)

        results = await asyncio.to_thread(
            self._client().hybrid_search,
            collection_name=self.collection_name,
            reqs=requests,
            ranker=ranker,
            limit=limit,
            output_fields=["path", "source_path", "content_hash"],
        )

        return parse_milvus_hits(results)

    def _build_ranker(self, milvus: Dict, modes: tuple):
        if settings.hybrid_ranker.lower() == "weighted":
            weights = []
            for mode in modes:
                if mode == "dense":
                    weights.append(settings.hybrid_dense_weight)
                elif mode == "sparse":
                    weights.append(settings.hybrid_sparse_weight)
            return milvus["WeightedRanker"](*weights)

        return milvus["RRFRanker"](k=settings.hybrid_rrf_k)


def parse_milvus_hits(results) -> List[HybridHit]:
    if not results:
        return []

    first_result = results[0] if isinstance(results, list) else results
    hits = []

    for rank, hit in enumerate(first_result, start=1):
        entity = _hit_entity(hit)
        path = entity.get("path")
        if not path:
            continue

        hits.append(
            HybridHit(
                rank=rank,
                path=path,
                source_path=entity.get("source_path", path),
                score=_hit_score(hit),
                chunk_id=_hit_id(hit),
            )
        )

    return hits


def _hit_entity(hit) -> Dict:
    if isinstance(hit, dict):
        return hit.get("entity") or hit
    entity = getattr(hit, "entity", None)
    if isinstance(entity, dict):
        return entity
    if hasattr(hit, "get"):
        try:
            entity = hit.get("entity")
            if isinstance(entity, dict):
                return entity
        except Exception:
            pass
    return {}


def _hit_score(hit) -> Optional[float]:
    if isinstance(hit, dict):
        value = hit.get("score", hit.get("distance"))
    else:
        value = getattr(hit, "score", getattr(hit, "distance", None))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _hit_id(hit) -> Optional[str]:
    if isinstance(hit, dict):
        return hit.get("id") or hit.get("chunk_id")
    value = getattr(hit, "id", None)
    return str(value) if value is not None else None


def _batches(values: Iterable, batch_size: int):
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
