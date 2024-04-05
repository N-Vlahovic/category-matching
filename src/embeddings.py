from __future__ import annotations
import numpy as np
import torch
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List

import categories


# nomic-ai/nomic-embed-text-v1.5
MODEL: SentenceTransformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# MODEL: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')


def encode(query: str | List[str], convert_to_tensor: bool = True) -> np.ndarray | torch.Tensor:
    return MODEL.encode(query, convert_to_tensor=convert_to_tensor)


class EmbeddingResponse(BaseModel):
    query: List[str] | None = None
    category: categories.Category | None = None
    embedding: List[List[float]]

    @classmethod
    def from_embedding(cls, embedding: Embedding) -> EmbeddingResponse:
        return cls(
            query=embedding.query,
            category=embedding.category,
            embedding=[[b for b in a] for a in embedding.embedding]
        )


class Embedding(BaseModel):
    query: List[str] | None = None
    category: categories.Category | None = None
    embedding: np.ndarray | torch.Tensor = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


def get_embedding(query: List[str]) -> EmbeddingResponse:
    return EmbeddingResponse.from_embedding(Embedding(query=query, embedding=encode(query)))


def get_category_embeddings() -> List[Embedding]:
    return [Embedding(
        category=_,
        embedding=encode([_.name]),
    ) for _ in categories.CORPUS]


EMBEDDINGS: List[Embedding] = get_category_embeddings()
EMBEDDING_RESPONSES: List[EmbeddingResponse] = [EmbeddingResponse.from_embedding(_) for _ in EMBEDDINGS]


def get_corpus_embedding() -> torch.Tensor:
    return torch.Tensor(
        [_.embedding for _ in EMBEDDING_RESPONSES]
    ).reshape([len(categories.CORPUS), EMBEDDINGS[0].embedding.shape[1]]).to(device='cuda')
