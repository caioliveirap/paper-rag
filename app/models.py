from datetime import datetime
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    section: str
    page_numbers: list[int] = []
    token_count: int
    chunk_index: int


class Paper(BaseModel):
    paper_id: str
    filename: str
    title: str
    authors: list[str] = []
    abstract: str = ""
    total_chunks: int
    ingested_at: datetime


class RetrievedChunk(Chunk):
    relevance_score: float


class QueryResult(BaseModel):
    question: str
    answer: str
    sources: list[RetrievedChunk]
    model_used: str
