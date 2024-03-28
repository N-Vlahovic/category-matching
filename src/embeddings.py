from __future__ import annotations
import numpy as np
import torch
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List

import categories


MODEL: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def encode(query: str | List[str], convert_to_tensor: bool = True) -> np.ndarray | torch.Tensor:
    return MODEL.encode(query, convert_to_tensor=convert_to_tensor)


class EmbeddingResponse(BaseModel):
    category: categories.Category
    embedding: List[List[float]]

    @classmethod
    def from_embedding(cls, embedding: Embedding) -> EmbeddingResponse:
        return cls(
            category=embedding.category,
            embedding=[[b for b in a] for a in embedding.embedding]
        )


class Embedding(BaseModel):
    category: categories.Category
    embedding: np.ndarray | torch.Tensor = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


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
    ).reshape([3, EMBEDDINGS[0].embedding.shape[1]]).to(device='mps')
