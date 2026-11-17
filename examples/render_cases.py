"""Render a readable, auditable Markdown report from real pipeline outputs."""

from __future__ import annotations

import math
import html
import re
from collections import defaultdict
from typing import Any, Mapping


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _quoted(value: Any) -> str:
    """Quote untrusted multiline text as Markdown while preserving its lines."""
    text = _text(value).replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return "> *(ไม่มีข้อความ)*"
    lines = []
    for line in text.split("\n"):
        line = html.escape(line, quote=True)
        line = re.sub(r"([\\`*_{}\[\]()#+\-.!|:])", r"\\\1", line)
        lines.append("> " + line)
    return "\n".join(lines)


def _cell(value: Any) -> str:
    """Keep dataset and model text inside a Markdown table cell."""
    text = _text(value).replace("\r", " ").replace("\n", " ").strip()
    text = html.escape(text, quote=True)
    text = re.sub(r"([\\`*_{}\[\]()#+\-.!|:])", r"\\\1", text)
    return text or "—"


def _ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _statute_label(corpus: Mapping[str, Mapping[str, Any]], statute_id: str) -> str:
    statute = corpus.get(statute_id, {})
    name = _text(statute.get("law_name"))
    section = _text(statute.get("section_num"))
    if name or section:
        return f"{statute_id} · {name} มาตรา {section}".strip()
    return statute_id


def _metric(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _ranking_table(
    corpus: Mapping[str, Mapping[str, Any]],
    rankings: list[dict[str, Any]],
    methods: list[str],
) -> list[str]:
    by_method = {str(row["method"]): row for row in rankings}
    lines = ["| อันดับ | " + " | ".join(_cell(method) for method in methods) + " |",
             "| ---: | " + " | ".join("---" for _ in methods) + " |"]
    for rank in range(1, 6):
        cells = [str(rank)]
        for method in methods:
            record = by_method.get(method)
            ranked = record.get("ranked", []) if record else []
            item = next((entry for entry in ranked if int(entry.get("rank", 0)) == rank), None)
            if item is None:
                cells.append("—")
                continue
            label = _statute_label(corpus, str(item["statute_id"]))
            gold = " · gold" if item.get("is_gold") else ""
            cosine = _metric(item.get("cosine"), 4)
            cells.append(f"{_cell(label)} · cosine `{cosine}`{gold}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _cosine_examples(
    corpus: Mapping[str, Mapping[str, Any]],
    rankings: list[dict[str, Any]],
    methods: list[str],
) -> list[str]:
    lines = ["#### ตัวอย่างผลคูณที่รวมเป็น cosine", "",
             "เวกเตอร์ query และมาตราถูก normalize ไว้แล้ว ค่า dot product จึงตรงกับ cosine similarity ที่ใช้จัดอันดับ ส่วนนี้แสดงพิกัดที่มีผลคูณสัมบูรณ์สูงสุดไม่เกิน 4 พิกัด", "",
             "| วิธีค้น | มาตราที่เทียบ | ‖q‖₂ | ‖d‖₂ | q·d | ผลคูณตามพิกัด |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    by_method = {str(row["method"]): row for row in rankings}
    for method in methods:
        record = by_method.get(method)
        examples = record.get("cosine_examples", []) if record else []
        example = examples[0] if examples else None
        if example is None:
            lines.append(f"| {_cell(method)} | — | — | — | — | ไม่มีข้อมูลตัวอย่าง |")
            continue
        coordinates = example.get("largest_coordinate_products", [])
        products = ", ".join(
            f"{int(item['coordinate'])}: {_metric(item['query_value'], 4)} × "
            f"{_metric(item['statute_value'], 4)} = {_metric(item['product'], 4)}"
            for item in coordinates
        ) or "ไม่มีพิกัดที่ไม่เป็นศูนย์"
        lines.append("| " + " | ".join([
            _cell(method), _cell(_statute_label(corpus, str(example.get("statute_id", "")))),
            _metric(example.get("query_l2"), 4), _metric(example.get("statute_l2"), 4),
            _metric(example.get("normalized_dot_product"), 4), _cell(products),
        ]) + " |")
    return lines


def _dataset_embedding_example(
    question_id: str,
    dataset: Mapping[str, Any] | None,
    embeddings: list[dict[str, Any]] | None,
) -> list[str]:
    question = next((row for row in (dataset or {}).get("selected_questions", [])
                     if str(row.get("question_id")) == question_id), None)
    lines = ["### จากแถวข้อมูลสู่เวกเตอร์", ""]
    if question is None:
        lines.extend(["ไม่มี raw record ของคำถามนี้ใน dataset trace", ""])
    else:
        source = question.get("raw_record", {})
        lines.extend([
            f"แหล่งข้อมูล `{_cell(question.get('dataset_id'))}` · split `{_cell(question.get('split'))}` · row {question.get('source_row_number', '—')} · revision `{_cell(question.get('dataset_revision'))}`",
            "",
            "**ข้อความคำถามดิบ**", "", _quoted(source.get("question")), "",
            f"**หลัง normalize** {_quoted(question.get('normalized_question'))}", "",
            "**รายการมาตราจากแถวข้อมูลที่จับคู่ได้**", "",
            "| ข้อมูลต้นทาง | รหัสใน corpus | วิธีจับคู่ |",
            "| --- | --- | --- |",
        ])
        contexts = source.get("positive_contexts", source.get("relevant_laws", [])) or []
        matched = question.get("matched_statutes", [])
        for position, statute in enumerate(matched):
            raw_context = contexts[position] if position < len(contexts) else {}
            law = raw_context.get("metadata", {}).get("law_title") or raw_context.get("law_name") or statute.get("law_name")
            section = raw_context.get("metadata", {}).get("section") or raw_context.get("section_num") or statute.get("section_num")
            lines.append("| " + " | ".join([
                _cell(f"{law} มาตรา {section}"),
                _cell(f"{statute.get('statute_id')} · {statute.get('law_name')} มาตรา {statute.get('section_num')}"),
                _cell(statute.get("matched_by")),
            ]) + " |")
        if not matched:
            lines.append("| ไม่มีรายการมาตราที่จับคู่ได้ | — | — |")
        lines.extend(["", "ข้อความในแถวต้นทางอาจใช้รูปเลขหรือช่องว่างต่างจาก corpus โปรแกรมจับคู่ด้วยชื่อกฎหมายและเลขมาตราที่ normalize แล้ว และบันทึกรหัส corpus ที่ได้ไว้ข้างบน", ""])

    by_method = [row for row in (embeddings or []) if str(row.get("question_id")) == question_id]
    if not by_method:
        lines.extend(["ยังไม่มี embedding trace สำหรับคำถามนี้", ""])
        return lines

    lines.extend(["**ตัวอย่าง token และพิกัดเวกเตอร์**", "",
                  "แสดงสองพิกัดแรกจากตัวอย่างที่ export แต่ละ embedding มีมิติและค่าครบใน [embeddings.json](embeddings.json)", "",
                  "| วิธี | Token ที่ใช้ | ตัวอย่างพิกัด 1 | ตัวอย่างพิกัด 2 |",
                  "| --- | --- | --- | --- |"])
    for row in by_method:
        embedding = row.get("embedding", {})
        tokens = embedding.get("tokens")
        if tokens is None:
            tokens = embedding.get("token_preview", [])
        token_text = ", ".join(_text(token) for token in tokens[:8])
        if len(tokens) > 8:
            token_text += ", …"
        coordinates = embedding.get("coordinates", [])[:2]
        values = []
        for item in coordinates:
            dimension = item.get("index", item.get("coordinate", "?"))
            if "term" in item:
                detail = f"{item['term']}: raw {_metric(item.get('raw_value'), 4)}, norm {_metric(item.get('normalized_value'), 4)}"
            elif "raw_mean" in item:
                detail = f"raw mean {_metric(item.get('raw_mean'), 4)}, norm {_metric(item.get('normalized_value'), 4)}"
            else:
                detail = f"value {_metric(item.get('value', item.get('normalized_value')), 4)}"
            values.append(f"มิติ {dimension} · {detail}")
        values.extend(["—"] * (2 - len(values)))
        lines.append("| " + " | ".join([
            _cell(row.get("method")), _cell(token_text) or "—",
            _cell(values[0]), _cell(values[1]),
        ]) + " |")
    lines.extend(["", "R1 เป็นค่าการมีคำในข้อความ, R2 เป็นน้ำหนัก TF-IDF, R3 เป็นค่าเฉลี่ยเวกเตอร์คำ, ส่วน R4 เป็นค่าพิกัดจาก encoder ซึ่งไม่แทนคำทีละคำ", ""])
    return lines


def _method_answer_section(
    corpus: Mapping[str, Mapping[str, Any]], answer: Mapping[str, Any]
) -> list[str]:
    method = _text(answer.get("method"))
    prompt_ids = _ids(answer.get("prompt_ids"))
    top_k_ids = _ids(answer.get("top_k_ids"))
    omitted_ids = _ids(answer.get("omitted_ids"))
    cited_ids = _ids(answer.get("cited_ids"))
    invalid_ids = set(_ids(answer.get("invalid_ids")))
    lines = [f"#### {_cell(method)}", "",
             f"- Retrieved top-k: {', '.join(_cell(sid) for sid in top_k_ids) or 'ไม่มี'}",
             f"- ตัวบทที่เข้า prompt: {', '.join(_cell(sid) for sid in prompt_ids) or 'ไม่มี'}",
             f"- ตัวบทที่ไม่ได้ใส่เพราะงบ token: {', '.join(_cell(sid) for sid in omitted_ids) or 'ไม่มี'}",
             "- คำอ้างในคำตอบ: " + (", ".join(
                 f"{_cell(sid)}{' (ไม่พบใน prompt)' if sid in invalid_ids else ''}" for sid in cited_ids
             ) or "ไม่มี"),
             f"- Gold อยู่ใน top-k/prompt: {bool(answer.get('gold_in_top_k'))} / {bool(answer.get('gold_in_prompt'))}",
             f"- Abstain: {bool(answer.get('abstained'))} · input/output tokens: "
             f"{answer.get('input_tokens', '—')} / {answer.get('output_tokens', '—')} · "
             f"เวลา generate: {_metric(answer.get('generation_seconds'), 2)} วินาที",
             "", "**คำตอบที่โมเดลสร้าง**", "", _quoted(answer.get("answer")), "",
             "**ตัวบทที่ส่งให้โมเดล**", ""]
    if not prompt_ids:
        lines.append("ไม่มีตัวบทถูกส่งเข้า prompt")
    for sid in prompt_ids:
        statute = corpus.get(sid)
        if statute is None:
            lines.extend([f"- **{_cell(sid)}** · ไม่พบรหัสนี้ในคลังที่ export", ""])
            continue
        label = _statute_label(corpus, sid)
        text = _text(statute.get("section_content"))
        excerpt = text[:1000]
        if len(text) > len(excerpt):
            excerpt += "…"
        lines.extend([f"- **{_cell(label)}** · อ่านฉบับเต็มได้ใน [dataset.json](dataset.json) ด้วย ID {_cell(sid)}",
                      "", _quoted(excerpt), ""])
    return lines


def render_cases(
    corpus: Mapping[str, Mapping[str, Any]],
    questions: Any,
    rankings: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    dataset: Mapping[str, Any] | None = None,
    embeddings: list[dict[str, Any]] | None = None,
) -> str:
    """Create a Markdown case report comparing retrieval and generated answers.

    Text copied from datasets, laws, and model output is emitted as quoted
    Markdown. Scores describe retrieval order only. Generated answer quality
    must be assessed by reading the answer against the cited statute text.
    """
    required = {"question_id", "question", "gold_ids", "reference_answer"}
    missing = required - set(getattr(questions, "columns", []))
    if missing:
        raise ValueError(f"questions are missing columns {sorted(missing)}")

    question_rows = list(questions.to_dict(orient="records"))
    ranking_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rankings:
        ranking_by_question[str(row["question_id"])].append(row)
    answer_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answers:
        answer_by_question[str(row["question_id"])].append(row)

    declared_methods = [_text(method) for method in manifest.get("methods", [])]
    present_methods = list(dict.fromkeys(
        _text(row.get("method")) for row in rankings if row.get("method")
    ))
    methods = declared_methods or present_methods
    for method in present_methods:
        if method not in methods:
            methods.append(method)

    lines = [
        "# ตัวอย่างข้อมูลและผลการทำงาน Thai Legal RAG", "",
        "> รายงานนี้สร้างจากข้อมูลและผลที่ exporter บันทึกไว้ ตรวจค่ารายละเอียดได้จากไฟล์ JSON ข้างเคียง", "",
        "## ขอบเขตของรอบนี้", "",
        f"- คลังตัวบทมี {int(manifest.get('corpus_size', len(corpus))):,} รายการ · scope: {_cell(manifest.get('scope', '—'))}",
        *([f"- จำนวน candidate ที่ใช้จัดอันดับ: {int(manifest.get('candidate_count', manifest.get('corpus_size', len(corpus)))):,} รายการ"] if manifest.get("scope") == "subset" else []),
        *([f"- วิธีเลือก candidate: {_cell(manifest.get('candidate_selection', 'ไม่มีรายละเอียด'))}"] if manifest.get("scope") == "subset" else []),
        f"- source commit: {_cell(manifest.get('source_commit', '—'))}",
        f"- วิธีค้นที่ประกาศใน manifest: {', '.join(_cell(method) for method in methods) or 'ไม่มี'}",
        f"- สถานะการสร้างคำตอบ: {_cell(manifest.get('generation_status', 'skipped'))}",
        "- ไฟล์ข้อมูลดิบ: [dataset.json](dataset.json) · [embeddings.json](embeddings.json) · [rankings.json](rankings.json) · [manifest.json](manifest.json)",
        *(["- ผลคำตอบ: [answers.json](answers.json)"] if manifest.get("generation_status") == "completed" else []), "",
        "คำเตือน: gold คือป้ายกำกับจากชุดข้อมูล อาจไม่ครอบคลุมทุกมาตราที่ตอบได้ ส่วนกรณีใช้ candidate subset ระบบคัดมาตรา gold และผลค้นอันดับต้นจากวิธี sparse ไว้ก่อน แล้วจึงจัดอันดับภายใน candidate ที่เลือก คะแนนจึงอธิบายผลเฉพาะขอบเขตนี้ ไม่ใช่ benchmark ของคลังเต็ม", "",
    ]

    if not question_rows:
        lines.extend(["## ตัวอย่างคำถาม", "", "รอบนี้ไม่มีคำถามที่ส่งออก", ""])

    for question in question_rows:
        qid = str(question["question_id"])
        query_rankings = ranking_by_question.get(qid, [])
        query_answers = answer_by_question.get(qid, [])
        lines.extend([f"## {_cell(qid)}", "", "**คำถามจากชุดข้อมูล**", "", _quoted(question.get("question")), "",
                      "**มาตราที่ชุดข้อมูลกำกับไว้ (gold)**", ""])
        gold_ids = _ids(question.get("gold_ids"))
        if gold_ids:
            lines.extend([f"- {_cell(_statute_label(corpus, sid))}" for sid in gold_ids])
        else:
            lines.append("ไม่มี gold statute")
        lines.extend(["", *_dataset_embedding_example(qid, dataset, embeddings)])
        lines.extend(["", "### จัดอันดับจาก cosine similarity", "",
                      *_ranking_table(corpus, query_rankings, methods), "",
                      "ตารางแสดงอันดับสูงสุด 5 รายการต่อวิธี และติดป้าย gold เมื่อรหัสตรงกับชุดข้อมูล", "",
                      *_cosine_examples(corpus, query_rankings, methods), "",
                      "| วิธีค้น | Recall@5 | Hit@5 | nDCG@5 | RR@10 |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        by_method = {str(row["method"]): row for row in query_rankings}
        for method in methods:
            record = by_method.get(method)
            if record is None:
                lines.append(f"| {_cell(method)} | — | — | — | — |")
                continue
            metrics = record.get("metrics") or {}
            lines.append("| " + " | ".join([
                _cell(method), _metric(metrics.get("recall_at_5")),
                _metric(metrics.get("hit_at_5")), _metric(metrics.get("ndcg_at_5")),
                _metric(metrics.get("rr_at_10")),
            ]) + " |")
        lines.extend(["", "### คำตอบอ้างอิงจากชุดข้อมูล", "", _quoted(question.get("reference_answer")), ""])

        if not query_answers:
            lines.extend(["### คำตอบจาก Generator", "", "ยังไม่ได้รัน generator สำหรับตัวอย่างนี้", ""])
        else:
            lines.extend(["### คำตอบจาก Generator", "",
                          "คำตอบแต่ละวิธีใช้คำถามและรุ่น generator เดียวกัน โดยเปลี่ยนรายการตัวบทตามผลค้น วิธีนี้ช่วยเทียบผลที่ได้ แต่ยังต้องอ่านคำตอบเทียบตัวบทเพื่อประเมินใจความ", ""])
            for method in methods:
                answer = next((row for row in query_answers if str(row.get("method")) == method), None)
                if answer is None:
                    lines.extend([f"#### {_cell(method)}", "", "ไม่มีผลคำตอบสำหรับวิธีนี้ในไฟล์ผล", ""])
                else:
                    lines.extend(_method_answer_section(corpus, answer))
            for answer in query_answers:
                if str(answer.get("method")) not in methods:
                    lines.extend(_method_answer_section(corpus, answer))
        lines.extend(["---", ""])

    if not question_rows:
        lines.append("## หมายเหตุการอ่านผล")
        lines.append("")
        lines.append("ไม่พบตัวอย่างคำถามในชุดส่งออก จึงไม่มีอันดับหรือคำตอบให้ตรวจ")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
