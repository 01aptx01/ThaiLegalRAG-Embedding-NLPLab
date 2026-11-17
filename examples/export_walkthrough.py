"""Export a few named questions from a trusted, locally built inference index."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import normalize

from thai_legal_rag.config import RunConfig
from thai_legal_rag.data import search_text
from thai_legal_rag.inference_index import InferenceIndex
from thai_legal_rag.pipeline import _versions
from thai_legal_rag.retrieval import METHODS, rank_exact, score_ranking

from dataset_trace import load_example_data
from embedding_trace import describe_embeddings, query_vector
from generation_trace import run_answer_examples
from render_cases import render_cases


ROOT = Path(__file__).resolve().parents[1]
PILOT_REVISION = "654e1a10afc44ecaee36fd739bc90078dcc62388"
TEST_REVISION = "9f75697075e0f5691e3b821037b908c05f1821a7"


def revision(value: str) -> str:
    if not re.fullmatch(r"[a-fA-F0-9]{40}", value):
        raise argparse.ArgumentTypeError("expected a 40-character commit SHA")
    return value.lower()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def dense_row(matrix) -> np.ndarray:
    return np.asarray(matrix.toarray() if issparse(matrix) else matrix).reshape(-1)


def trace_ranking(index, question, method, query, config, positions):
    matrix = index.matrices[method]
    ranked_ids = rank_exact(query, matrix, index.ids, len(index.ids))[0]
    gold = set(question.gold_ids)
    ranks = {sid: rank for rank, sid in enumerate(ranked_ids, 1) if sid in gold}
    scores = dense_row(query @ matrix.T)
    top = ranked_ids[:config.rank_k]
    at_5 = score_ranking(gold, top, 5)
    return {
        "question_id": question.question_id,
        "question": question.question,
        "method": index.methods[method],
        "method_key": method,
        "gold_ids": question.gold_ids,
        "retrieved_ids": top,
        "ranked": [
            {"rank": rank, "statute_id": sid, "cosine": float(scores[positions[sid]]), "is_gold": sid in gold}
            for rank, sid in enumerate(top, 1)
        ],
        "gold_ranks": {sid: ranks.get(sid) for sid in question.gold_ids},
        "metrics": {
            "recall_at_5": float(at_5["recall"]),
            "hit_at_5": at_5["hit"],
            "ndcg_at_5": float(at_5["ndcg"]),
            "rr_at_10": float(score_ranking(gold, top, 10)["rr"]),
        },
        "empty_retrieval": not ranked_ids,
    }


def cosine_example(query, statute, sid: str) -> dict:
    # The stored vectors have already been L2-normalized by the index builder.
    # Densify only the two selected rows for a readable coordinate breakdown.
    query_row, statute_row = dense_row(query), dense_row(statute)
    products = query_row * statute_row
    greatest = np.argsort(-np.abs(products), kind="stable")[:4]
    return {
        "statute_id": sid,
        "query_l2": float(np.linalg.norm(query_row)),
        "statute_l2": float(np.linalg.norm(statute_row)),
        "normalized_dot_product": float(dense_row(query @ statute.T)[0]),
        "largest_coordinate_products": [
            {"coordinate": int(i), "query_value": float(query_row[i]),
             "statute_value": float(statute_row[i]), "product": float(products[i])}
            for i in greatest if products[i] != 0
        ],
    }


def sample_index(index, questions, size: int, bge_revision: str, config: RunConfig):
    """Build a small four-way demonstration from the same candidate statutes."""
    import torch
    from sentence_transformers import SentenceTransformer

    if tuple(index.methods) != ("r1", "r2", "r3"):
        raise ValueError("sample mode requires an R1–R3 index built with --no-dense")
    selected = {sid for gold_ids in questions["gold_ids"] for sid in gold_ids}
    for question in questions["question"]:
        for method in index.methods:
            selected.update(index.search(question, method, limit=config.rank_k))
    if not len(selected) <= size < len(index.ids):
        raise ValueError(f"sample size must be between {len(selected)} and {len(index.ids) - 1}")
    pool = [sid for sid in index.ids if sid not in selected]
    remaining = size - len(selected)
    selected.update(np.random.default_rng(config.seed).choice(pool, remaining, replace=False).tolist())
    positions = [i for i, sid in enumerate(index.ids) if sid in selected]
    index.ids = [index.ids[i] for i in positions]
    index.matrices = {method: matrix[positions] for method, matrix in index.matrices.items()}
    dtype = torch.float16 if index.device == "cuda" else torch.float32
    index.encoder = SentenceTransformer(config.bge_id, revision=bge_revision, device=index.device,
                                        model_kwargs={"torch_dtype": dtype})
    index.encoder.max_seq_length = config.bge_max_tokens
    texts = [search_text(index.corpus[sid]["section_content"]) for sid in index.ids]
    index.matrices["r4"] = normalize(index.encoder.encode(
        texts, batch_size=config.bge_batch_size, show_progress_bar=True,
    ).astype(np.float32))
    index.methods["r4"] = METHODS[3]
    index.manifest.update(bge_revision=bge_revision, bge_dtype=str(dtype), bge_max_tokens=config.bge_max_tokens)


def export(args, destination: Path) -> None:
    index = InferenceIndex.load(args.index_dir, args.device)
    revisions = {"corpus": index.corpus_revision, "pilot": args.pilot_revision, "test": args.test_revision}
    corpus, questions, datasets = load_example_data(revisions, args.question_id, destination / "audit")
    if list(corpus) != index.ids or any(
        any(corpus[sid][key] != value for key, value in index.corpus[sid].items()) for sid in index.ids
    ):
        raise ValueError("index content or order does not match the pinned statute corpus")
    config = RunConfig()
    if args.sample_statutes is not None:
        sample_index(index, questions, args.sample_statutes, args.bge_revision, config)
    rankings, embeddings = [], []
    positions = {sid: i for i, sid in enumerate(index.ids)}
    for question in questions.itertuples(index=False):
        vectors = {key: query_vector(index, question.question, key) for key in index.methods}
        rows = [trace_ranking(index, question, key, vectors[key], config, positions) for key in index.methods]
        rankings.extend(rows)
        # Use the same examples across methods: up to two gold statutes and a distractor.
        samples = list(question.gold_ids[:2])
        distractor = next((sid for row in rows for sid in row["retrieved_ids"] if sid not in question.gold_ids), None)
        if distractor is not None:
            samples.append(distractor)
        for row in rows:
            key = row["method_key"]
            row["cosine_examples"] = [
                cosine_example(vectors[key], index.matrices[key][positions[sid]:positions[sid] + 1], sid)
                for sid in samples
            ]
            embeddings.append({
                "question_id": question.question_id, "method": index.methods[key], "method_key": key,
                "statutes": [{"statute_id": sid, "law_name": corpus[sid]["law_name"],
                              "section_num": corpus[sid]["section_num"]} for sid in samples],
                "embedding": describe_embeddings(index, question.question, key, vectors[key], samples),
            })

    # The BGE encoder and the generator are not needed in memory together.
    index.encoder = None
    answers = []
    if args.generator_revision is not None:
        answers = run_answer_examples(corpus, questions, rankings, config, args.generator_revision,
                                      index.device, destination / "generation")
        write_json(destination / "answers.json", answers)
        prompt_ids = sorted({sid for answer in answers for sid in answer["prompt_ids"]})
        datasets["prompt_statutes"] = {sid: {
            field: corpus[sid][field] for field in ("law_name", "section_num", "section_content")
        } for sid in prompt_ids}

    manifest = {
        "format_version": 1,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "subset" if args.sample_statutes else "full_corpus",
        "corpus_size": len(corpus),
        "candidate_count": len(index.ids),
        "candidate_ids": index.ids if args.sample_statutes else None,
        "candidate_selection": (
            "gold plus R1–R3 full-index top 10 per question, then seeded sampling without replacement"
            if args.sample_statutes else "all pinned statutes in source order"
        ),
        "dataset_revisions": revisions,
        "question_ids": args.question_id,
        "methods": list(index.methods.values()),
        "device": index.device,
        "bge_revision": index.manifest.get("bge_revision"),
        "bge_model": index.manifest.get("bge_model"),
        "bge_max_tokens": index.manifest.get("bge_max_tokens"),
        "bge_dtype": index.manifest.get("bge_dtype"),
        "generator": config.generator_id if answers else None,
        "generator_revision": args.generator_revision,
        "generation_status": "completed" if answers else "skipped",
        "seed": config.seed,
        "rank_k": config.rank_k,
        "top_k": config.top_k,
        "max_input_tokens": config.max_input_tokens,
        "max_new_tokens": config.max_new_tokens,
        "versions": _versions(),
    }
    write_json(destination / "dataset.json", datasets)
    write_json(destination / "embeddings.json", embeddings)
    write_json(destination / "rankings.json", rankings)
    write_json(destination / "manifest.json", manifest)
    report = render_cases(corpus, questions, rankings, answers, manifest, dataset=datasets, embeddings=embeddings)
    (destination / "cases.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True, type=Path, help="Trusted local inference index")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for exported examples")
    parser.add_argument("--question-id", required=True, action="append", help="Pinned pilot/test row ID; repeat for each case")
    parser.add_argument("--pilot-revision", type=revision, default=PILOT_REVISION)
    parser.add_argument("--test-revision", type=revision, default=TEST_REVISION)
    parser.add_argument("--sample-statutes", type=int, help="Score a teaching subset using an R1–R3 index")
    parser.add_argument("--bge-revision", type=revision, help="Pinned BGE revision for sample mode")
    parser.add_argument("--generator-revision", type=revision, help="Enable Typhoon at this pinned revision")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    if len(set(args.question_id)) != len(args.question_id):
        parser.error("question IDs must be unique")
    if (args.sample_statutes is None) != (args.bge_revision is None):
        parser.error("--sample-statutes and --bge-revision must be used together")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        export(args, temporary)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
