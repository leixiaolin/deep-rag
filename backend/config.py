from pydantic_settings import BaseSettings
from pydantic import AliasChoices, Field
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Core configuration class - all settings loaded from .env with defaults"""
    
    api_provider: str = Field(default="google")
    tool_calling_mode: str = Field(default="function")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=8192)
    knowledge_base: str = Field(
        default="./Knowledge-Base",
        validation_alias=AliasChoices("KNOWLEDGE_BASE", "KNOWLEDGE_BASE_PATH"),
    )
    knowledge_base_chunks: str = Field(
        default="./Knowledge-Base-Chunks",
        validation_alias=AliasChoices("KNOWLEDGE_BASE_CHUNKS", "KNOWLEDGE_BASE_CHUNKS_PATH"),
    )
    knowledge_base_file_summary: str = Field(default="./Knowledge-Base-File-Summary/summary.txt")
    retrieval_backend: str = Field(default="direct")
    milvus_uri: str = Field(default="http://localhost:19530")
    milvus_token: str = Field(default="")
    milvus_collection: str = Field(default="deep_rag_chunks_v1")
    milvus_dense_metric_type: str = Field(default="COSINE")
    milvus_dense_index_type: str = Field(default="HNSW")
    milvus_hnsw_m: int = Field(default=16)
    milvus_hnsw_ef_construction: int = Field(default=200)
    milvus_hnsw_ef: int = Field(default=64)
    milvus_bm25_k1: float = Field(default=1.2)
    milvus_bm25_b: float = Field(default=0.75)
    milvus_max_text_length: int = Field(default=65000)
    embedding_provider: str = Field(default="openai")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dim: int = Field(default=1536)
    embedding_batch_size: int = Field(default=32)
    embedding_timeout: float = Field(default=120.0)
    hybrid_top_k: int = Field(default=8)
    hybrid_max_top_k: int = Field(default=30)
    hybrid_rrf_k: int = Field(default=100)
    hybrid_ranker: str = Field(default="rrf")
    hybrid_dense_weight: float = Field(default=0.3)
    hybrid_sparse_weight: float = Field(default=0.7)

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_provider_config(self, provider: str) -> dict:
        """Dynamically get provider configuration from .env"""
        prefix = provider.upper()
        
        config = {
            "api_key": os.getenv(f"{prefix}_API_KEY"),
            "base_url": os.getenv(f"{prefix}_BASE_URL"),
            "model": os.getenv(f"{prefix}_MODEL"),
            "headers": {}
        }
        
        headers_str = os.getenv(f"{prefix}_HEADERS")
        if headers_str:
            import json
            try:
                config["headers"] = json.loads(headers_str)
            except json.JSONDecodeError:
                pass
        
        return config

    def get_embedding_config(self) -> dict:
        """Get OpenAI-compatible embedding configuration.

        EMBEDDING_* values override the provider selected by EMBEDDING_PROVIDER.
        This lets chat use one provider while embeddings use another.
        """
        provider = (self.embedding_provider or self.api_provider).lower()
        config = self.get_provider_config(provider)
        config["api_key"] = os.getenv("EMBEDDING_API_KEY") or config.get("api_key")
        config["base_url"] = os.getenv("EMBEDDING_BASE_URL") or config.get("base_url")
        config["model"] = os.getenv("EMBEDDING_MODEL") or self.embedding_model or config.get("model")

        headers_str = os.getenv("EMBEDDING_HEADERS")
        if headers_str:
            import json
            try:
                config["headers"].update(json.loads(headers_str))
            except json.JSONDecodeError:
                pass

        return config
    
    def list_available_providers(self) -> list:
        """List all configured providers by scanning {PROVIDER}_MODEL environment variables"""
        providers = []
        for key in os.environ:
            if key.endswith('_MODEL'):
                provider = key[:-6].lower()
                if provider == "embedding":
                    continue
                providers.append(provider)
        return sorted(providers)

settings = Settings()
