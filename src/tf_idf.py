from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import List

import categories


class TfIdfSearchRes(BaseModel):
    category: categories.Category
    score: float


class TfIdfSearch(BaseModel):
    query: str
    results: List[TfIdfSearchRes]
    top_k: int


def get_tf_idf_scores(query: str, top_k: int) -> TfIdfSearch:
    vectorizer = TfidfVectorizer()
    corpus = [query] + [_.name for _ in categories.CORPUS]
    tfidf = vectorizer.fit_transform(corpus)
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-(top_k + 2):-1][1:]
    related_docs_scores = cosine_similarities[related_docs_indices]
    results = [TfIdfSearchRes(
        score=s,
        category=categories.CORPUS[i - 1],
    ) for i, s in zip(related_docs_indices, related_docs_scores)]
    return TfIdfSearch(
        query=query,
        top_k=top_k,
        results=results,
    )
