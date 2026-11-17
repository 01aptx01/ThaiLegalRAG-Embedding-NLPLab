"""Load pinned source rows and build human-readable, traceable data examples."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi

from thai_legal_rag.data import DATASETS, load_sources, prepare_data, search_text


def _excerpt(value: Any, limit: int = 700) -> str:
    """Return a readable excerpt while preserving the original source in data."""
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _truncated(value: Any, limit: int) -> bool:
    return len(str(value or "").strip()) > limit


def _raw_question_example(row: dict[str, Any], split: str) -> dict[str, Any]:
    """Keep the source fields that explain how the gold statutes were labelled."""
    result: dict[str, Any] = {
        "question": row.get("question", ""),
    }
    context_field = "positive_contexts" if split == "pilot" else "relevant_laws"
    contexts = row.get(context_field) or []
    source_contexts = []
    for item in contexts:
        item = item or {}
        if split == "pilot":
            metadata = item.get("metadata") or {}
            source_contexts.append({
                "metadata": {
                    "law_title": metadata.get("law_title"),
                    "section": metadata.get("section"),
                },
                "context_excerpt": _excerpt(item.get("context")),
                "context_is_excerpt": _truncated(item.get("context"), 700),
            })
        else:
            source_contexts.append({
                "law_name": item.get("law_name"),
                "section_num": item.get("section_num"),
                "section_content_excerpt": _excerpt(item.get("section_content")),
                "section_content_is_excerpt": _truncated(item.get("section_content"), 700),
            })
    result[context_field] = source_contexts
    answer_field = "positive_answer" if split == "pilot" else "reference_answer"
    result[answer_field] = _excerpt(row.get(answer_field), limit=1200)
    result[f"{answer_field}_is_excerpt"] = _truncated(row.get(answer_field), 1200)
    return result


def _statute_example(statute_id: str, row: dict[str, Any], match_method: str) -> dict[str, Any]:
    content = row.get("section_content", "")
    return {
        "statute_id": statute_id,
        "matched_by": match_method,
        "law_name": row.get("law_name"),
        "section_num": row.get("section_num"),
        "raw_section_content_excerpt": _excerpt(content),
        "normalized_search_text_excerpt": _excerpt(search_text(content)),
    }


def load_example_data(
    revisions: dict[str, str], question_ids: list[str], audit_dir: Path
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Load pinned datasets and return examples linked through the real matcher.

    ``question_ids`` accepts IDs such as ``pilot-04144`` and ``test-00016``.
    Every eligible question is passed through the production matcher first, then
    the requested rows are returned in the caller's order. The audit files in
    ``audit_dir`` cover both source splits in one set.
    """
    if not question_ids:
        raise ValueError("question_ids must contain at least one pilot- or test-question ID")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("question_ids must not contain duplicates")

    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    raw, resolved_revisions = load_sources(HfApi(), revisions)

    corpus, pilot_questions = prepare_data(
        raw,
        "pilot",
        pilot_questions=max(1, len(raw["pilot"])),
        seed=42,
        run_dir=audit_dir,
    )
    # The production helper writes combined audit CSVs on every call. Preserve
    # the single canonical set above and isolate its second call in a temp dir.
    with tempfile.TemporaryDirectory(prefix=".prepare-test-", dir=audit_dir) as temp_dir:
        _, test_questions = prepare_data(
            raw,
            "test",
            pilot_questions=max(1, len(raw["test"])),
            seed=42,
            run_dir=Path(temp_dir),
        )
    prepared = {"pilot": pilot_questions, "test": test_questions}

    raw_by_split = {
        "pilot": {f"pilot-{index:05d}": row for index, row in enumerate(raw["pilot"], 1)},
        "test": {f"test-{index:05d}": row for index, row in enumerate(raw["test"], 1)},
    }
    accepted_by_id = {
        row["question_id"]: (split, row)
        for split, questions in prepared.items()
        for row in questions.to_dict(orient="records")
    }

    unknown = [question_id for question_id in question_ids if not any(question_id in rows for rows in raw_by_split.values())]
    if unknown:
        raise ValueError(f"question IDs do not exist in the pinned datasets: {unknown}")
    ineligible = [question_id for question_id in question_ids if question_id not in accepted_by_id]
    if ineligible:
        raise ValueError(f"question IDs were excluded by the production matching rules: {ineligible}")

    selected_rows = []
    question_examples = []
    mapping_audit = pd.read_csv(audit_dir / "mapping_audit.csv", keep_default_na=False)

    for question_id in question_ids:
        split, prepared_row = accepted_by_id[question_id]
        selected_rows.append(prepared_row)
        raw_row = raw_by_split[split][question_id]
        mapping_rows = mapping_audit.loc[mapping_audit["question_id"].eq(question_id)]
        statute_examples = []
        for statute_id in prepared_row["gold_ids"]:
            match = mapping_rows.loc[mapping_rows["statute_id"].eq(statute_id)]
            match_method = match.iloc[0]["status"] if not match.empty else "unknown"
            statute_examples.append(_statute_example(statute_id, corpus[statute_id], match_method))

        question_examples.append({
            "question_id": question_id,
            "dataset_id": DATASETS[split][0],
            "split": DATASETS[split][1],
            "dataset_revision": resolved_revisions[split],
            "dataset_card_url": f"https://huggingface.co/datasets/{DATASETS[split][0]}/tree/{resolved_revisions[split]}",
            "source_row_number": int(question_id.rsplit("-", 1)[1]),
            "raw_record": _raw_question_example(raw_row, split),
            "normalized_question": search_text(prepared_row["question"]),
            "gold_ids": list(prepared_row["gold_ids"]),
            "gold_law": prepared_row["gold_law"],
            "reference_answer": prepared_row["reference_answer"],
            "matched_statutes": statute_examples,
        })

    # Select one genuinely excluded source row and retain its recorded reason.
    excluded_example = None
    audit = pd.read_csv(audit_dir / "question_audit.csv", keep_default_na=False)
    excluded = audit.loc[audit["reason"].ne("included")]
    if not excluded.empty:
        audit_row = excluded.iloc[0]
        question_id = str(audit_row["question_id"])
        split = question_id.split("-", 1)[0]
        excluded_example = {
            "question_id": question_id,
            "dataset_id": DATASETS[split][0],
            "split": DATASETS[split][1],
            "dataset_revision": resolved_revisions[split],
            "dataset_card_url": f"https://huggingface.co/datasets/{DATASETS[split][0]}/tree/{resolved_revisions[split]}",
            "exclusion_reason": str(audit_row["reason"]),
            "raw_record": _raw_question_example(raw_by_split[split][question_id], split),
        }

    selected = pd.DataFrame(selected_rows).set_index("question_id").loc[question_ids].reset_index()
    corpus_example_id = selected.iloc[0]["gold_ids"][0]
    dataset_trace = {
        "sources": {
            name: {
                "dataset_id": DATASETS[name][0],
                "split": DATASETS[name][1],
                "revision": resolved_revisions[name],
                "dataset_card_url": f"https://huggingface.co/datasets/{DATASETS[name][0]}/tree/{resolved_revisions[name]}",
                "rows_loaded": len(raw[name]),
            }
            for name in DATASETS
        },
        "license_notes": {
            "VISAI-AI/nitibench-statute": "MIT, https://huggingface.co/datasets/VISAI-AI/nitibench-statute",
            "VISAI-AI/nitibench": "MIT, https://huggingface.co/datasets/VISAI-AI/nitibench",
            "airesearch/WangchanX-Legal-ThaiCCL-RAG": "MIT, https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG",
        },
        "corpus_example": {
            "dataset_id": DATASETS["corpus"][0],
            "split": DATASETS["corpus"][1],
            "revision": resolved_revisions["corpus"],
            "dataset_card_url": f"https://huggingface.co/datasets/{DATASETS['corpus'][0]}/tree/{resolved_revisions['corpus']}",
            "row_count": len(corpus),
            "statute": _statute_example(corpus_example_id, corpus[corpus_example_id], "gold match"),
        },
        "selected_questions": question_examples,
        "excluded_example": excluded_example,
        "audit_files": [
            "audit/statute_ids.csv",
            "audit/question_audit.csv",
            "audit/mapping_audit.csv",
            "audit/duplicate_keys.jsonl",
        ],
    }
    return corpus, selected, dataset_trace
