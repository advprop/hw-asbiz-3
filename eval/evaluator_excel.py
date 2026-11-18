import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from time import perf_counter

from openpyxl import Workbook

from eval.eval_functions import score_answer, score_context
from rag.limits import retry_pause
from rag.llm import limiter
from rag.pipeline import RAG

METRICS = ("context_recall", "context_precision", "answer_correctness", "time")
COLUMNS = ("question_id", "question", "gold_context", "gold_answer", "context", "answer", *METRICS)
JUDGE_ATTEMPTS = 6


def read_items(dataset_path):
    with Path(dataset_path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def judge(gold_answer, answer):
    # судья сидит на том же ключе, что и RAG, поэтому ждём общий лимит и повторяем при сбоях
    error = None
    for attempt in range(JUDGE_ATTEMPTS):
        limiter.wait()
        try:
            return score_answer(gold_answer, answer)
        except Exception as exc:
            error = exc
            limiter.back_off(retry_pause(attempt + 1))
    raise RuntimeError(f"судья не ответил: {error}")


def evaluate_one(system, item):
    start = perf_counter()
    try:
        prediction = system.rag(item["question"])
    except Exception as exc:
        # один упавший вопрос не должен ронять весь прогон, он просто получает нули
        print(f"{item['question_id']}: ошибка RAG: {exc}", file=sys.stderr)
        prediction = {"answer": "", "context": []}
    elapsed = perf_counter() - start

    context_scores = score_context(item["gold_context"], prediction["context"])
    if prediction["answer"]:
        answer_score = judge(item["gold_answer"], prediction["answer"])
    else:
        answer_score = {"answer_correctness": 0, "reason": "нет ответа"}
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "gold_context": json.dumps(item["gold_context"], ensure_ascii=False),
        "gold_answer": item["gold_answer"],
        "context": json.dumps(prediction["context"], ensure_ascii=False),
        "answer": prediction["answer"],
        **context_scores,
        "answer_correctness": answer_score["answer_correctness"],
        "time": elapsed,
        "judge_reason": answer_score["reason"],
    }


def load_checkpoint(path):
    done = {}
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[row["question_id"]] = row
    return done


def write_excel(results, output_path):
    report = Workbook()
    summary = report.active
    summary.title = "Итоги"
    summary.append(METRICS)
    summary.append([mean(row[key] for row in results) for key in METRICS])
    details = report.create_sheet("По вопросам")
    details.append(COLUMNS)
    for row in results:
        details.append([row[key] for key in COLUMNS])
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"  # чтобы ответ, начинающийся с "=", не стал формулой
    report.save(output_path)


def evaluate(dataset_path, output_path=None, workers=1, limit=None, resume=False):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path or dataset_path.with_name(dataset_path.stem + "_results.xlsx"))
    if output_path.resolve() == dataset_path.resolve():
        raise ValueError("Выберите для отчёта отдельный файл.")
    items = read_items(dataset_path)[:limit]
    if not items:
        raise ValueError("В датасете нет вопросов.")

    # готовые вопросы пишем в jsonl по одному: прогон можно продолжить через --resume
    checkpoint = output_path.with_suffix(".jsonl")
    done = load_checkpoint(checkpoint) if resume else {}
    todo = [item for item in items if item["question_id"] not in done]

    system = RAG()
    lock = threading.Lock()
    started = time.time()
    with checkpoint.open("a" if resume else "w", encoding="utf-8") as log, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(evaluate_one, system, item) for item in todo]
        for number, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            with lock:
                done[row["question_id"]] = row
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
            print(
                f"{number}/{len(todo)} {row['question_id']} recall={row['context_recall']:.2f} "
                f"precision={row['context_precision']:.2f} correct={row['answer_correctness']} "
                f"time={row['time']:.1f} ({time.time() - started:.0f} с)",
                file=sys.stderr,
            )

    results = [done[item["question_id"]] for item in items]
    write_excel(results, output_path)
    for key in METRICS:
        print(f"{key}: {mean(row[key] for row in results):.4f}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="куда сохранить Excel")
    parser.add_argument("--workers", type=int, default=1, help="сколько вопросов обрабатывать одновременно")
    parser.add_argument("--limit", type=int, help="взять только первые N вопросов")
    parser.add_argument("--resume", action="store_true", help="продолжить прерванный прогон")
    args = parser.parse_args()
    print(evaluate(args.dataset_path, args.output, args.workers, args.limit, args.resume))
