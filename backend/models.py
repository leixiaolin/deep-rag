from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    provider: Optional[str] = None
    model: Optional[str] = None

class FileRetrievalRequest(BaseModel):
    file_paths: List[str] = Field(default_factory=list, example=[
        "Product-Line-A-Smartwatch-Series/SW-2100-Flagship.md",
        "2023-Market-Layout/"
    ])
    query: Optional[str] = Field(
        None,
        description="Optional retrieval query for Milvus hybrid sparse+dense search.",
    )
    top_k: Optional[int] = Field(
        None,
        description="Optional number of hybrid retrieval results. Backend clamps to a safe range.",
    )

class FileRetrievalResponse(BaseModel):
    content: str

class SummaryRetrievalRequest(BaseModel):
    path: str = Field(".", description="Path to explore. '.' for root, a category name, or a file path.")
    depth: int = Field(1, description="How many levels to expand.")

class KnowledgeBaseInfo(BaseModel):
    summary: str
    file_tree: Dict[str, Any]

class ProviderConfig(BaseModel):
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    providers: List[str]
