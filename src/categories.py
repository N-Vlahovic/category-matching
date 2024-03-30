from pydantic import BaseModel
from typing import List


class Category(BaseModel):
    name: str


class CategoryRes(BaseModel):
    status: str


CORPUS: List[Category] = [Category(name=_) for _ in [
    'semi-conductors',
    'bolts & screws',
    'labour',
    'computer hardware',
]]


def add_category(
        name: str,
) -> CategoryRes:
    global CORPUS
    CORPUS.append(Category(name=name))
    return CategoryRes(status='ok')


def rm_category(name: str) -> CategoryRes:
    global CORPUS
    CORPUS = [_ for _ in CORPUS if _.name != name]
    return CategoryRes(status='ok')


def set_categories(names: List[str]) -> CategoryRes:
    global CORPUS
    CORPUS = [Category(name=_) for _ in names]
    return CategoryRes(status='ok')
