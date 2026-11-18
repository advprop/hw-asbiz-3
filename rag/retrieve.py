import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag.embed import embed
from rag.index import EMBEDDINGS
from rag.parse import load_documents, load_pages
from rag.route import rank_documents

RRF_K = 60
DOC_BONUS = 1.0  # насколько сильно поднимаем страницы из отчёта за нужный год


def tokenize(text):
    words = re.findall(r"[a-zа-я]+|\d+", text.lower().replace("ё", "е"))
    # грубый стемминг: у длинных слов оставляем первые 6 букв
    return [w[:6] if w.isalpha() else w for w in words if len(w) > 1]


def ranks(scores):
    order = np.argsort(-scores)
    result = np.empty(len(scores))
    result[order] = np.arange(1, len(scores) + 1)
    return result


class Retriever:
    # после создания только читает данные, поэтому можно вызывать из многих потоков

    def __init__(self, use_embeddings=True):
        self.documents = {d["doc_id"]: d for d in load_documents()}
        self.pages = load_pages()
        self.doc_of_page = np.array([p["doc_id"] for p in self.pages])
        self.position = {(p["doc_id"], p["page"]): i for i, p in enumerate(self.pages)}
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in self.pages])
        self.embeddings = None
        if use_embeddings:
            self.embeddings = np.load(EMBEDDINGS).astype(np.float32)
            if len(self.embeddings) != len(self.pages):
                raise ValueError("индекс не совпадает со страницами, пересоберите: python -m rag.index")

    def text(self, doc_id, page):
        return self.pages[self.position[(doc_id, page)]]["text"]

    def search(self, query, top_k=8, doc_ids=None, route_by=None):
        # route_by - исходный вопрос: по нему выбираем отчёты, даже если ищем по другому запросу
        weights = rank_documents(route_by or query, self.documents)
        if doc_ids:
            weights = {d: weights.get(d, 0.0) for d in doc_ids if d in self.documents}
        if not weights:
            weights = {d: 0.0 for d in self.documents}
        allowed = np.isin(self.doc_of_page, list(weights))

        score = 1.0 / (RRF_K + ranks(np.where(allowed, self.bm25.get_scores(tokenize(query)), -np.inf)))
        if self.embeddings is not None:
            dense = self.embeddings @ np.array(embed(query), dtype=np.float32)
            score = score + 1.0 / (RRF_K + ranks(np.where(allowed, dense, -np.inf)))
        bonus = np.array([weights.get(d, 0.0) for d in self.doc_of_page])
        score = np.where(allowed, score * (1 + DOC_BONUS * bonus), -np.inf)

        found = []
        for i in np.argsort(-score)[:top_k]:
            if np.isfinite(score[i]):
                page = self.pages[i]
                found.append({"doc_id": page["doc_id"], "page": page["page"], "text": page["text"], "score": float(score[i])})
        return found
