from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    source_id: str
    source_type: str
    title: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class DocumentChunk:
    chunk_id: str
    source_id: str
    source_type: str
    title: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class WebSearchResult:
    title: str
    content: str
    url: str


@dataclass
class AnswerSource:
    source_type: str
    reference: str
