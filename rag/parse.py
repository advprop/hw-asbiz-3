import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ROOT / "data" / "documents.jsonl"
REPORTS = ROOT / "data" / "reports"
PAGES = ROOT / "artifacts" / "pages.jsonl.gz"
MIN_TEXT_CHARS = 200  # если текста меньше, это скан
OCR_ZOOM = 2.5


def rows_to_text(words, gap):
    # слова на одной высоте собираем в одну строку
    words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, current, center = [], [], None
    for word in words:
        middle, height = (word[1] + word[3]) / 2, word[3] - word[1]
        if current and abs(middle - center) <= max(height, 1e-6) * 0.5:
            current.append(word)
            center += (middle - center) / len(current)
        else:
            if current:
                rows.append(current)
            current, center = [word], middle
    if current:
        rows.append(current)
    lines = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        parts = [row[0][4]]
        for previous, word in zip(row, row[1:]):
            # большой пробел между словами - это соседние колонки
            parts.append("  |  " if word[0] - previous[2] > gap else " ")
            parts.append(word[4])
        lines.append("".join(parts))
    return "\n".join(lines)


def native_text(page):
    words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
    return rows_to_text(words, gap=12)


def ocr_text(page):
    from ocrmac import ocrmac
    from PIL import Image

    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(OCR_ZOOM, OCR_ZOOM), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    found = ocrmac.OCR(
        image, language_preference=["ru-RU", "en-US"], recognition_level="accurate"
    ).recognize()
    words = []
    for text, _confidence, (x, y, width, height) in found:
        # у Vision координаты в долях страницы и идут снизу вверх
        top = (1 - y - height) * page.rect.height
        left = x * page.rect.width
        words.append((left, top, left + width * page.rect.width, top + height * page.rect.height, text))
    return rows_to_text(words, gap=12)


def parse_document(document):
    result = []
    with pymupdf.open(REPORTS / document["path"]) as pdf:
        for number, page in enumerate(pdf, start=1):
            text, method = native_text(page), "text"
            if len(text.strip()) < MIN_TEXT_CHARS:
                recognized = ocr_text(page)
                if len(recognized.strip()) > len(text.strip()):
                    text, method = recognized, "ocr"
            result.append({"doc_id": document["doc_id"], "page": number, "method": method, "text": text})
    return result


def load_documents():
    with DOCUMENTS.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_pages():
    with gzip.open(PAGES, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def main():
    documents = load_documents()
    PAGES.parent.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool, gzip.open(PAGES, "wt", encoding="utf-8") as output:
        for document, pages in zip(documents, pool.map(parse_document, documents)):
            for page in pages:
                output.write(json.dumps(page, ensure_ascii=False) + "\n")
            recognized = sum(page["method"] == "ocr" for page in pages)
            print(f"{document['doc_id']}: {len(pages)} стр., распознано {recognized}", file=sys.stderr)


if __name__ == "__main__":
    main()
