from pydantic import BaseModel
from typing import List


class Category(BaseModel):
    name: str


CORPUS: List[Category] = [Category(name=_) for _ in [
    'semi-conductors',
    'bolts & screws',
    'labour',
]]
