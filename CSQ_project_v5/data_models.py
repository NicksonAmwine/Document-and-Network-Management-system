from dataclasses import dataclass
from pydantic import BaseModel
from typing import List

class QueryExpansion(BaseModel):
    queries: List[str]

@dataclass
class DocumentChunk:
    """Enhanced document chunk with source tracking."""
    content: str
    chunk_id: str
    chunk_index: int
    source_document: str
    document_type: str
    last_modified: str

@dataclass
class RetrievalResult:
    """Result from similarity search with source tracking."""
    content: str
    source_document: str
    relevance_score: float
    chunk_id: str