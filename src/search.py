from pydantic import BaseModel
from sentence_transformers import util
import torch
from typing import List

import categories
import embeddings


corpus_embeddings: torch.Tensor = embeddings.get_corpus_embedding()


class SemanticSearchRes(BaseModel):
    category: categories.Category
    score: float


class SemanticSearch(BaseModel):
    query: str
    results: List[SemanticSearchRes]
    top_k: int


def core_search(query: str, top_k: int = 3) -> torch.topk:
    query_embedding = embeddings.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    return torch.topk(cos_scores, k=top_k)


def search(query: str, top_k: int = 3) -> SemanticSearch:
    top_results = core_search(query, top_k)
    return SemanticSearch(
        query=query,
        top_k=top_k,
        results=[
            SemanticSearchRes(
                category=categories.CORPUS[idx],
                score=score,
            ) for score, idx in zip(top_results[0], top_results[1])
        ]
    )


if __name__ == '__main__':
    Q = [
        'single board computer',
        'screwdriver',
        'machine operator',
    ]
    for q in Q:
        print(search(q))
        print()
    print()
