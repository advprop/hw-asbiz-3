import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from rag.embed import embed
from rag.parse import ROOT, load_documents, load_pages

EMBEDDINGS = ROOT / "artifacts" / "embeddings.npy"
PARTIAL = ROOT / "artifacts" / "embeddings_partial.jsonl"
WORKERS = 8


def doc_title(document):
    _, year, period = document["doc_id"].rsplit("_", 2)
    period = "за 6 месяцев" if period == "6m" else "годовая"
    return f"{document['organization']}, отчётность {period} {year} года"


def page_for_embedding(page, titles):
    return f"{titles[page['doc_id']]}, страница {page['page']}\n{page['text']}"


def load_partial():
    done = {}
    if PARTIAL.exists():
        with PARTIAL.open() as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # последняя строка могла не дописаться
                done[(row["doc_id"], row["page"])] = row["embedding"]
    return done


def main():
    titles = {d["doc_id"]: doc_title(d) for d in load_documents()}
    pages = load_pages()

    # если скрипт перезапустили, готовые страницы не считаем заново
    done = load_partial()
    todo = [p for p in pages if (p["doc_id"], p["page"]) not in done]
    print(f"осталось {len(todo)} из {len(pages)}", file=sys.stderr)

    lock = threading.Lock()
    started = time.time()
    with PARTIAL.open("a") as output, ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(embed, page_for_embedding(p, titles)): p for p in todo}
        for number, future in enumerate(as_completed(futures), start=1):
            page = futures[future]
            vector = future.result()
            with lock:
                done[(page["doc_id"], page["page"])] = vector
                output.write(json.dumps({"doc_id": page["doc_id"], "page": page["page"], "embedding": vector}) + "\n")
                output.flush()
            if number % 200 == 0:
                print(f"{number}/{len(todo)} за {time.time() - started:.0f} с", file=sys.stderr)

    matrix = np.array([done[(p["doc_id"], p["page"])] for p in pages], dtype=np.float16)
    np.save(EMBEDDINGS, matrix)
    print(matrix.shape)


if __name__ == "__main__":
    main()
