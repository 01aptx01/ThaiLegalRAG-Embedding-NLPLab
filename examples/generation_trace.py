"""Generate answers through the production pipeline and return its raw rows."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from thai_legal_rag.generation import run_generation
from thai_legal_rag.retrieval import METHODS


def run_answer_examples(
    corpus: dict[str, dict[str, Any]],
    questions: pd.DataFrame,
    retrieval_rows: list[dict[str, Any]],
    config: Any,
    revision: str,
    device: str,
    output_dir: Path,
    *,
    bfloat16: bool = False,
) -> list[dict[str, Any]]:
    """Call the production generator and return unchanged JSONL result rows."""
    import torch

    output_dir = Path(output_dir)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("generator revision must be a pinned 40-character SHA")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"requested device is unavailable: {device}")
    required = {"question_id", "question", "gold_ids", "reference_answer"}
    if required - set(questions.columns) or questions.empty:
        raise ValueError("questions must include question_id, question, gold_ids, reference_answer")
    if questions["question_id"].astype(str).duplicated().any():
        raise ValueError("question IDs must be unique")
    if questions["reference_answer"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("every example must include a reference answer")
    question_table = questions.copy()
    question_table["question_id"] = question_table["question_id"].astype(str)
    question_table = question_table.set_index("question_id", drop=False)
    question_ids = list(question_table.index)

    if not retrieval_rows:
        raise ValueError("retrieval rows are empty")
    methods = list(dict.fromkeys(row.get("method") for row in retrieval_rows))
    if any(method not in METHODS for method in methods):
        raise ValueError("retrieval rows contain an unknown method")
    expected_pairs = {(qid, method) for qid in question_ids for method in methods}
    received_pairs: list[tuple[str, str]] = []
    generation_rows: list[dict[str, Any]] = []
    for number, row in enumerate(retrieval_rows, 1):
        qid, method = str(row.get("question_id", "")), row.get("method")
        label = f"retrieval row {number}"
        if qid not in question_table.index or method not in methods:
            raise ValueError(f"{label}: unexpected question or method")
        source = question_table.loc[qid]
        if str(row.get("question", "")) != str(source["question"]):
            raise ValueError(f"{label}: question text does not match {qid}")
        gold_ids = list(row.get("gold_ids", []))
        expected_gold = list(source["gold_ids"])
        if not gold_ids or set(gold_ids) != set(expected_gold):
            raise ValueError(f"{label}: gold IDs do not match {qid}")
        retrieved_ids = list(row.get("retrieved_ids", []))
        if len(retrieved_ids) != len(set(retrieved_ids)) or set(retrieved_ids) - set(corpus):
            raise ValueError(f"{label}: retrieved IDs are duplicated or missing from the corpus")
        received_pairs.append((qid, method))
        generation_rows.append({
            "method": method, "question_id": qid, "question": str(source["question"]),
            "gold_ids": expected_gold, "retrieved_ids": retrieved_ids,
        })
    counts = Counter(received_pairs)
    if set(received_pairs) != expected_pairs or any(count != 1 for count in counts.values()):
        raise ValueError("retrieval rows must contain exactly one row per question and enabled method")

    method_order = {method: position for position, method in enumerate(METHODS)}
    question_order = {qid: position for position, qid in enumerate(question_ids)}
    generation_rows.sort(key=lambda row: (question_order[row["question_id"]], method_order[row["method"]]))
    indexed_questions = question_table[["reference_answer"]].copy()
    indexed_questions.index.name = "question_id"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if bfloat16:
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support bfloat16")
        dtype = torch.bfloat16
    run_generation(
        corpus, indexed_questions, pd.DataFrame(generation_rows), config,
        revision, device, dtype, output_dir,
    )

    result_path = output_dir / "generation_rows.jsonl"
    with result_path.open(encoding="utf-8") as stream:
        results = [json.loads(line) for line in stream if line.strip()]
    result_pairs = [(str(row["question_id"]), row["method"]) for row in results]
    result_counts = Counter(result_pairs)
    if set(result_pairs) != expected_pairs or any(count != 1 for count in result_counts.values()):
        raise RuntimeError("production generator output is missing or duplicates a question/method pair")
    retrieval_by_pair = {(row["question_id"], row["method"]): row for row in generation_rows}
    for result in results:
        key = (str(result["question_id"]), result["method"])
        retrieved = retrieval_by_pair[key]["retrieved_ids"][:config.top_k]
        prompt_ids, omitted_ids = result["prompt_ids"], result["omitted_ids"]
        if [sid for sid in retrieved if sid in set(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"{key}: prompt IDs do not follow retrieval order")
        if [sid for sid in retrieved if sid not in set(prompt_ids)] != omitted_ids:
            raise RuntimeError(f"{key}: omitted IDs do not match retrieval output")
    return results
