from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import categories
import embeddings
import search


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


@app.get('/search')
async def semantic_search(query: str, top_k: int = 1) -> search.SemanticSearch:
    return search.search(query, top_k)
