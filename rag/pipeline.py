import json
import re

from rag.index import doc_title
from rag.llm import chat
from rag.retrieve import Retriever
from rag.route import rank_documents

TOP_K = 8
EXTRA_K = 5
MAX_SEARCHES = 2
FALLBACK_PAGES = 2

SYSTEM = """Ты отвечаешь на вопросы по финансовой отчётности банков. Тебе дают вопрос и страницы из PDF-отчётов, найденные поиском. Отвечай только по этим страницам, ничего не выдумывай.

Верни один JSON-объект без Markdown:
{"thoughts": "...", "action": "answer" или "search", "query": "...", "doc_id": "...", "answer": "...", "pages": [{"doc_id": "...", "page": 1}]}

thoughts — коротко: на какой странице и в какой строке и колонке таблицы стоит нужное значение, в каких единицах таблица (смотри заголовок: тысячи, миллионы, миллиарды рублей), и все вычисления по шагам.

Если на страницах есть всё нужное, action="answer":
- answer — ответ на русском. Назови значение с единицами измерения и датой или периодом. Если нужно считать (разница, доля, отношение, рост), приведи исходные числа и результат. Не добавляй лишних фактов, о которых не спрашивали.
- Числа бери точно как в отчёте, не округляй. Число в скобках в отчёте отрицательное (расход, убыток, отток).
- Следи за колонкой: нужная дата или период, а не сравнительные данные за другой год.
- pages — только страницы, из которых взяты факты ответа. Если одно и то же значение есть на нескольких данных страницах (например, в отчёте и в примечании), укажи все такие страницы.

Если нужных данных на страницах нет или их не хватает (например, нужен другой год, другой банк или другое примечание), action="search":
- query — новый поисковый запрос: название показателя или раздела так, как оно пишется в отчётности.
- doc_id — в каком отчёте искать, из списка доступных отчётов. Можно не указывать.
Искать можно не больше двух раз. Если поиск исчерпан, дай лучший ответ по тому, что есть; если данных всё равно нет, так и напиши, а pages оставь пустым."""


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def group_pages(pairs):
    # [(doc_id, page), ...] -> [{"doc_id": ..., "pages": [...]}] как ждёт eval
    grouped = {}
    for doc_id, page in pairs:
        grouped.setdefault(doc_id, set()).add(page)
    return [{"doc_id": doc_id, "pages": sorted(pages)} for doc_id, pages in grouped.items()]


class RAG:
    def __init__(self):
        self.retriever = Retriever()

    def format_pages(self, pages):
        return "\n\n".join(
            f'<page doc_id="{p["doc_id"]}" page="{p["page"]}">\n{p["text"]}\n</page>' for p in pages
        )

    def document_list(self, question):
        documents = self.retriever.documents
        weights = rank_documents(question, documents)
        lines = [f"{doc_id} — {doc_title(documents[doc_id])}" for doc_id in sorted(weights or documents)]
        return "\n".join(lines)

    def cited_pages(self, reply, shown):
        cited = []
        for item in reply.get("pages") or []:
            if not isinstance(item, dict):
                continue
            try:
                key = (str(item.get("doc_id")), int(item.get("page")))
            except (TypeError, ValueError):
                continue
            # модель может сослаться только на то, что ей показали
            if key in shown and key not in cited:
                cited.append(key)
        return cited

    def rag(self, question):
        found = self.retriever.search(question, top_k=TOP_K)
        shown = {(p["doc_id"], p["page"]): p for p in found}
        first_hits = list(shown)
        messages = [{
            "role": "user",
            "content": f"Вопрос: {question}\n\nДоступные отчёты:\n{self.document_list(question)}\n\n"
            f"Найденные страницы:\n{self.format_pages(found)}",
        }]
        searches = 0
        steps = []
        reply = None
        for _ in range(MAX_SEARCHES + 2):
            text, _usage = chat(SYSTEM, messages)
            reply = parse_json(text)
            if reply is None:
                messages += [
                    {"role": "assistant", "content": text or "(пусто)"},
                    {"role": "user", "content": "Ответ не разобрался как JSON. Верни один JSON-объект по формату."},
                ]
                reply = {"action": "answer", "answer": text, "pages": []}
                continue
            if reply.get("action") != "search" or searches >= MAX_SEARCHES:
                break

            searches += 1
            query = str(reply.get("query") or question)
            doc_id = reply.get("doc_id")
            doc_ids = [doc_id] if doc_id in self.retriever.documents else None
            extra = self.retriever.search(query, top_k=EXTRA_K + len(shown), doc_ids=doc_ids, route_by=question)
            extra = [p for p in extra if (p["doc_id"], p["page"]) not in shown][:EXTRA_K]
            steps.append({"query": query, "doc_id": doc_id, "found": [(p["doc_id"], p["page"]) for p in extra]})
            for page in extra:
                shown[(page["doc_id"], page["page"])] = page
            note = "Больше искать нельзя, дай ответ." if searches >= MAX_SEARCHES else ""
            body = self.format_pages(extra) if extra else "Новых страниц не нашлось."
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Результаты поиска «{query}»:\n{body}\n\n{note}".strip()},
            ]

        answer = str(reply.get("answer") or "").strip() or "Не удалось найти ответ в отчётности."
        cited = self.cited_pages(reply, shown)
        if not cited:
            cited = first_hits[:FALLBACK_PAGES]
        return {"answer": answer, "context": group_pages(cited), "searches": steps, "thoughts": reply.get("thoughts", "")}
