from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import categories
import embeddings
import semantic_search
import search
import tf_idf


app: FastAPI = FastAPI()


class Home(BaseModel):
    status: str


@app.get('/')
async def home() -> Home:
    return Home(status='ok')


@app.get('/category')
async def get_categories() -> List[categories.Category]:
    return categories.CORPUS


@app.get('/embedding')
async def get_embeddings() -> List[embeddings.EmbeddingResponse]:
    return embeddings.EMBEDDING_RESPONSES


@app.get('/semantic-search')
async def semantic_search(query: str, top_k: int = 1) -> semantic_search.SemanticSearch:
    return semantic_search.semantic_search(query, top_k)


@app.get('/search')
async def consolidated_search(query: str, top_k: int = 1) -> search.Search:
    return search.search(query, top_k)


@app.get('/tfidf-search')
async def tfidf_search(query: str, top_k: int = 1) -> tf_idf.TfIdfSearch:
    return tf_idf.get_tf_idf_scores(query, top_k)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, port=8080)
