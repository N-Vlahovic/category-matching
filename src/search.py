from pydantic import BaseModel
from typing import List, Optional, Tuple

import categories
import semantic_search
import tf_idf


AbstractSearchRes = semantic_search.SemanticSearchRes | tf_idf.TfIdfSearchRes


class SearchRes(BaseModel):
    category: categories.Category
    semantic_score: Optional[float]
    tfidf_score: Optional[float]
    weighted_score: Optional[float]


class Search(BaseModel):
    query: str
    top_k: int
    results: List[SearchRes]


def coalesce_score(score: float | None, fallback: float) -> float:
    return fallback if score is None or score == 0 else score


def helper(s_res: AbstractSearchRes, others: List[AbstractSearchRes]) -> Tuple[float, float, float]:
    this_score = s_res.score
    other_score = None
    try:
        other_score = next(filter(lambda _: _.category == s_res.category, others)).score
    except StopIteration:
        pass
    weighted_score = .5 * coalesce_score(this_score, other_score) + .5 * coalesce_score(other_score, this_score)
    return this_score, other_score, weighted_score


def search(query: str, top_k: int) -> Search:
    semantic_results = semantic_search.semantic_search(query, top_k).results
    tfidf_results = tf_idf.get_tf_idf_scores(query, top_k).results
    results = []
    for s_res in semantic_results:
        category = s_res.category
        semantic_score, tfidf_score, weighted_score = helper(s_res, tfidf_results)
        results.append(SearchRes(
            category=category,
            semantic_score=semantic_score,
            tfidf_score=tfidf_score,
            weighted_score=weighted_score,
        ))
    for s_res in [_ for _ in tfidf_results if _.category not in map(lambda _: _.category, semantic_results)]:
        category = s_res.category
        tfidf_score, semantic_score, weighted_score = helper(s_res, tfidf_results)
        results.append(SearchRes(
            category=category,
            semantic_score=semantic_score,
            tfidf_score=tfidf_score,
            weighted_score=weighted_score,
        ))
    results = sorted(results, key=lambda _: _.weighted_score, reverse=True)
    return Search(
        query=query,
        top_k=top_k,
        results=results[:top_k],
    )
