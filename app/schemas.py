from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class BookResult(BaseModel):
    title: str
    authors: str
    publication_year: int | None
    language: str | None


class RecommendationResponse(BaseModel):
    results: List[BookResult]
