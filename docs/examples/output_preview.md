# ดูตารางจากผลรันจริง

เปิดตารางจากผลรันที่บันทึกไว้ใน `docs/examples/run/` ใช้ Python กับ pandas เพื่ออ่านผล รันเซลล์ตามลำดับจากโฟลเดอร์ repository หรือ `docs/examples/` ได้เลย

[Notebook พร้อมผลที่รันแล้ว](output_preview.ipynb) · [รายงานทั้งสองข้อ](run/cases.md) · [วิธีสร้างผลชุดใหม่](README.md#เตรียมดัชนีและส่งออกตัวอย่าง)

## เปิดไฟล์ผลรัน

```python
from pathlib import Path
import json

from IPython.display import display
import pandas as pd

cwd = Path.cwd()
run_dir = next(
    (path for path in (cwd / "run", cwd / "docs/examples/run")
     if (path / "manifest.json").is_file()),
    None,
)
if run_dir is None:
    raise FileNotFoundError("เปิด notebook จากโฟลเดอร์ repository หรือ docs/examples/")

def read_json(name):
    return json.loads((run_dir / name).read_text(encoding="utf-8"))

manifest = read_json("manifest.json")
print(f"คำถาม {len(manifest['question_ids'])} ข้อ | วิธีค้น {len(manifest['methods'])} วิธี")
print(f"ชุดสาธิต {manifest['candidate_count']:,} จากคลัง {manifest['corpus_size']:,} มาตรา")
```

**ผลลัพธ์**

```text
คำถาม 2 ข้อ | วิธีค้น 4 วิธี
ชุดสาธิต 72 จากคลัง 5,127 มาตรา
```

## ข้อมูลต้นทางและแถวที่เลือก

ตารางแรกแสดงจำนวนแถวของแต่ละแหล่ง ส่วนตารางที่สองแสดงตัวบท คำถาม pilot/test และข้อที่ถูกคัดออกในรอบสาธิต

```python
# แสดงแถวตัวอย่างจากรอบส่งออก
example_data = read_json("dataset.json")
sources = pd.DataFrame([
    {"ส่วน": name, "dataset": source["dataset_id"], "split": source["split"], "จำนวนแถว": source["rows_loaded"]}
    for name, source in example_data["sources"].items()
])
print(f"ตารางแหล่งข้อมูล: {sources.shape[0]} rows × {sources.shape[1]} columns")
display(sources)

def short_text(text):
    text = " ".join(str(text).split())
    return text if len(text) <= 110 else text[:109] + "…"

preview_rows = []
statute = example_data["corpus_example"]["statute"]
preview_rows.append({"ส่วน": "corpus", "รหัสคำถาม": "—", "มาตรา": statute["section_num"],
                     "statute_id": statute["statute_id"], "ข้อความย่อ": short_text(statute["raw_section_content_excerpt"]),
                     "ผลจับคู่": statute["matched_by"]})
for question in example_data["selected_questions"]:
    for statute in question["matched_statutes"]:
        source = statute["source_mapping"]
        law_section = f"{source['law_name']} มาตรา {source['section']}"
        preview_rows.append({"ส่วน": question["question_id"].split("-")[0], "รหัสคำถาม": question["question_id"],
                             "มาตรา": law_section, "statute_id": statute["statute_id"],
                             "ข้อความย่อ": short_text(question["raw_record"]["question"]),
                             "ผลจับคู่": statute["matched_by"]})
excluded = example_data["excluded_example"]
if excluded:
    preview_rows.append({"ส่วน": "คัดออก", "รหัสคำถาม": excluded["question_id"], "มาตรา": "—",
                         "statute_id": "—", "ข้อความย่อ": short_text(excluded["raw_record"]["question"]),
                         "ผลจับคู่": excluded["exclusion_reason"]})
dataset_preview = pd.DataFrame(preview_rows)
print(f"ตารางตัวอย่างที่เลือก: {dataset_preview.shape[0]} rows × {dataset_preview.shape[1]} columns")
display(dataset_preview)
```

**ผลลัพธ์**

```text
ตารางแหล่งข้อมูล: 3 rows × 4 columns
ตารางตัวอย่างที่เลือก: 4 rows × 6 columns
```

| # | ส่วน | dataset | split | จำนวนแถว |
| --- | --- | --- | --- | --- |
| 0 | corpus | VISAI-AI/nitibench-statute | ccl | 5127 |
| 1 | pilot | airesearch/WangchanX-Legal-ThaiCCL-RAG | train | 8211 |
| 2 | test | VISAI-AI/nitibench | ccl | 3729 |

| # | ส่วน | รหัสคำถาม | มาตรา | statute_id | ข้อความย่อ | ผลจับคู่ |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | corpus | — | 218/18 | S00755 | พระราชบัญญัติหลักทรัพย์และตลาดหลักทรัพย์ พ.ศ. ... | gold match |
| 1 | pilot | pilot-04144 | พระราชบัญญัติหลักทรัพย์และตลาดหลักทรัพย์ พ.ศ. ... | S00755 | ผู้จัดการกองทุนต้องปฏิบัติอย่างไร? | matched_key |
| 2 | test | test-00016 | ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 1262 | S02926 | ถ้ามีมติของที่ประชุมใหญ่ให้อำนาจผู้ชำระบัญชีให... | matched_key |
| 3 | คัดออก | pilot-00169 | — | — | การเสียภาษีอากรนั้นจะถือว่าสมบูรณ์เมื่อใด | test_overlap |

## พิกัดเวกเตอร์จากคำถามและตัวบท

แสดงสองพิกัดต่อวิธีของคำถามเดียวกันและมาตรา S00755 ตัวเลขในตารางปัดแสดงสี่ตำแหน่ง ส่วน NaN หมายถึงไฟล์ผลไม่มีค่านั้นสำหรับวิธีดังกล่าว

```python
# แสดงพิกัดจริงของ query และมาตราอ้างอิงเดียวกันในสี่วิธี
embedding_rows = read_json("embeddings.json")
question_id, statute_id = "pilot-04144", "S00755"
selected = [row for row in embedding_rows if row["question_id"] == question_id]
preview_rows = []

for result in selected:
    trace = result["embedding"]
    statute = next(row for row in trace["statute_vectors"] if row["statute_id"] == statute_id)
    statute_coordinates = {row["index"]: row for row in statute["coordinates"]}
    for query_coordinate in trace["coordinates"][:2]:
        index = query_coordinate["index"]
        statute_coordinate = statute_coordinates[index]
        is_sparse = "term" in query_coordinate
        preview_rows.append({
            "วิธี": result["method"], "มิติ": trace["dimension"], "พิกัด": index,
            "คำใน vocabulary": query_coordinate.get("term"),
            "query ก่อนปรับ": query_coordinate.get("raw_value", query_coordinate.get("raw_mean")),
            "query หลังปรับ": query_coordinate.get("normalized_value", query_coordinate.get("value")),
            "S00755 หลังปรับ": statute_coordinate["value"],
            "S00755 ก่อนปรับ": statute_coordinate.get("raw_value") if is_sparse else None,
        })

embedding_preview_df = pd.DataFrame(preview_rows)
print("PyThaiNLP newmm:", ", ".join(selected[0]["embedding"]["tokens"]))
print("BGE-M3 tokenizer:", " ".join(selected[-1]["embedding"]["token_preview"]))
print(f"จำนวนแถว {len(embedding_preview_df)} | จำนวนคอลัมน์ {len(embedding_preview_df.columns)}")
display(embedding_preview_df.round(4).head(8))
```

**ผลลัพธ์**

```text
PyThaiNLP newmm: ผู้จัดการ, กองทุน, ต้อง, ปฏิบัติ, อย่างไร, ?
BGE-M3 tokenizer: <s> ▁ ผู้จัดการ กองทุน ต้อง ปฏิบัติ อย่างไร ? </s>
จำนวนแถว 8 | จำนวนคอลัมน์ 8
```

| # | วิธี | มิติ | พิกัด | คำใน vocabulary | query ก่อนปรับ | query หลังปรับ | S00755 หลังปรับ | S00755 ก่อนปรับ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | R1 Binary BoW | 7046 | 2046 | กองทุน | 1.0000 | 0.4472 | 0.1459 | 1.0000 |
| 1 | R1 Binary BoW | 7046 | 3379 | ต้อง | 1.0000 | 0.4472 | 0.1459 | 1.0000 |
| 2 | R2 TF-IDF | 7046 | 2046 | กองทุน | 4.5318 | 0.4501 | 0.7788 | 40.7865 |
| 3 | R2 TF-IDF | 7046 | 3379 | ต้อง | 1.8720 | 0.1859 | 0.0357 | 1.8720 |
| 4 | R3 Word2Vec | 150 | 1 | NaN | -0.4683 | -0.2227 | -0.3080 | NaN |
| 5 | R3 Word2Vec | 150 | 57 | NaN | 0.4606 | 0.2191 | 0.1579 | NaN |
| 6 | R4 BGE-M3 | 1024 | 386 | NaN | NaN | 0.2314 | 0.2372 | NaN |
| 7 | R4 BGE-M3 | 1024 | 297 | NaN | NaN | -0.1468 | -0.1567 | NaN |

## Cosine และอันดับมาตรา

ตารางแรกตามมาตราอ้างอิงเดียวกันในสี่วิธี ตารางถัดไปแสดงห้าอันดับแรกของ R1 ด้วย `.head()`

```python
# ดูอันดับของมาตราอ้างอิงเดียวกันจากทั้งสี่วิธีค้น
ranking_rows = read_json("rankings.json")
manifest = read_json("manifest.json")
question_id, gold_id = "pilot-04144", "S00755"
results = [row for row in ranking_rows if row["question_id"] == question_id]

comparison_df = pd.DataFrame([{
    "วิธีค้น": row["method"],
    "มาตราอ้างอิง": gold_id,
    "อันดับ": row["gold_ranks"][gold_id],
    "cosine": round(next(
        item["normalized_dot_product"] for item in row["cosine_examples"]
        if item["statute_id"] == gold_id
    ), 4),
    "Recall@5": row["metrics"]["recall_at_5"],
    "nDCG@5": round(row["metrics"]["ndcg_at_5"], 3),
    "RR@10": round(row["metrics"]["rr_at_10"], 3),
} for row in results])
print(f"คำถาม {question_id} | มาตรา gold {gold_id}")
print(f"ขอบเขตการจัดอันดับ {manifest['candidate_count']:,} จาก {manifest['corpus_size']:,} มาตรา")
print(f"ผลลัพธ์ {comparison_df.shape[0]} แถว | {comparison_df.shape[1]} คอลัมน์")
display(comparison_df)

# แสดงห้าอันดับแรกของ R1 ตามข้อมูลที่บันทึกไว้
r1 = next(row for row in results if row["method"] == "R1 Binary BoW")
top_five_df = pd.DataFrame(r1["ranked"])[["rank", "statute_id", "cosine", "is_gold"]].head()
top_five_df = top_five_df.rename(columns={
    "rank": "อันดับ", "statute_id": "รหัสมาตรา", "cosine": "cosine", "is_gold": "มาตรา gold",
})
print(f"R1 Binary BoW ผลค้น 5 อันดับแรก | {top_five_df.shape[0]} แถว | {top_five_df.shape[1]} คอลัมน์")
display(top_five_df.round({"cosine": 4}))
print(f"มาตรา {gold_id} อยู่ที่อันดับ {r1['gold_ranks'][gold_id]}")
```

**ผลลัพธ์**

```text
คำถาม pilot-04144 | มาตรา gold S00755
ขอบเขตการจัดอันดับ 72 จาก 5,127 มาตรา
ผลลัพธ์ 4 แถว | 7 คอลัมน์
R1 Binary BoW ผลค้น 5 อันดับแรก | 5 แถว | 4 คอลัมน์
มาตรา S00755 อยู่ที่อันดับ 6
```

| # | วิธีค้น | มาตราอ้างอิง | อันดับ | cosine | Recall@5 | nDCG@5 | RR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | R1 Binary BoW | S00755 | 6 | 0.1957 | 0.0 | 0.000 | 0.167 |
| 1 | R2 TF-IDF | S00755 | 3 | 0.4601 | 1.0 | 0.500 | 0.333 |
| 2 | R3 Word2Vec | S00755 | 2 | 0.8271 | 1.0 | 0.631 | 0.500 |
| 3 | R4 BGE-M3 | S00755 | 1 | 0.6671 | 1.0 | 1.000 | 1.000 |

| # | อันดับ | รหัสมาตรา | cosine | มาตรา gold |
| --- | --- | --- | --- | --- |
| 0 | 1 | S04712 | 0.2902 | False |
| 1 | 2 | S03802 | 0.2176 | False |
| 2 | 3 | S04711 | 0.2148 | False |
| 3 | 4 | S00754 | 0.2121 | False |
| 4 | 5 | S00861 | 0.2000 | False |

## คำตอบจริงที่ Typhoon สร้าง

เปิดข้อ test-00016 แล้วดูข้อความเต็มตามที่โมเดลส่งออก คู่กับรหัสมาตราใน prompt และรหัสที่คำตอบอ้าง

```python
# เลือกผลสร้างคำตอบของคำถาม test-00016 จากไฟล์ผลที่บันทึกไว้
question_id = "test-00016"
dataset_trace = read_json("dataset.json")
question = next(
    row for row in dataset_trace["selected_questions"]
    if row["question_id"] == question_id
)
answer_rows = [
    row for row in read_json("answers.json")
    if row["question_id"] == question_id
]
preview_manifest = read_json("manifest.json")
answer_limit = preview_manifest["max_new_tokens"]

if not answer_rows:
    raise ValueError(f"ไม่พบผลคำตอบของ {question_id} ใน answers.json")

print(f"คำถาม {question_id}: {question['raw_record']['question']}")
print(f"คำตอบอ้างอิง: {question['reference_answer']}")

answer_comparison_df = pd.DataFrame([{
    "วิธีค้น": row["method"],
    "มาตราที่ส่งให้โมเดล": ", ".join(row["prompt_ids"]) or "ไม่มี",
    "รหัสที่คำตอบอ้าง": ", ".join(row["cited_ids"]) or "ไม่มี",
    "output tokens": row["output_tokens"],
    f"ชนเพดาน {answer_limit} tokens": bool(row["output_token_limit_reached"]),
    "คำตอบที่สร้าง": row["answer"],
} for row in answer_rows])

print(f"ผลจากโมเดลจริง {len(answer_comparison_df)} แถว × {len(answer_comparison_df.columns)} คอลัมน์")
with pd.option_context("display.max_colwidth", None):
    display(answer_comparison_df)
```

**ผลลัพธ์**

```text
คำถาม test-00016: ถ้ามีมติของที่ประชุมใหญ่ให้อำนาจผู้ชำระบัญชีให้ชำระบัญชีแยกกันได้ต้องนำความไปจดทะเบียนภายในกี่วัน
คำตอบอ้างอิง: ต้องนำความจดทะเบียนภายในสิบสี่วันนับแต่วันลงมติ
ผลจากโมเดลจริง 4 แถว × 6 คอลัมน์
```

| # | วิธีค้น | มาตราที่ส่งให้โมเดล | รหัสที่คำตอบอ้าง | output tokens | ชนเพดาน 128 tokens | คำตอบที่สร้าง |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | R1 Binary BoW | S02926, S02922, S02934, S02814, S01693 | S02926 | 128 | True | คำถาม (ภาษาไทย)<br>ถ้ามีมติของที่ประชุมใหญ่ให้อำนาจผู้ชำระบัญชีให้ชำระบัญชีแยกกันได้ต้องนำความไปจดทะเบียนภายในกี่วัน<br><br>ตัวบท (ภาษาไทย)<br>[S02926] ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 1262<br>ถ้ามีมติของที่ประชุมใหญ่หรือคำบังคับของศาลให้อำนาจผู้ชำระบัญชีให้ทำการแยกก |
| 1 | R2 TF-IDF | S02926, S02934, S02925, S04706, S04353 | S02926 | 128 | True | คำถาม (ภาษาไทย)<br>ถ้ามีมติของที่ประชุมใหญ่ให้อำนาจผู้ชำระบัญชีให้ชำระบัญชีแยกกันได้ต้องนำความไปจดทะเบียนภายในกี่วัน<br><br>ตัวบท (ภาษาไทย)<br>[S02926] ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 1262<br>ถ้ามีมติของที่ประชุมใหญ่หรือคำบังคับของศาลให้อำนาจผู้ชำระบัญชีให้ทำการแยกก |
| 2 | R3 Word2Vec | S02926, S02934, S02922, S02932, S02918 | S02926 | 128 | True | คำถาม (ภาษาไทย)<br>ถ้ามีมติของที่ประชุมใหญ่ให้อำนาจผู้ชำระบัญชีให้ชำระบัญชีแยกกันได้ต้องนำความไปจดทะเบียนภายในกี่วัน<br><br>ตัวบท (ภาษาไทย)<br>[S02926] ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 1262<br>ถ้ามีมติของที่ประชุมใหญ่หรือคำบังคับของศาลให้อำนาจผู้ชำระบัญชีให้ทำการแยกก |
| 3 | R4 BGE-M3 | S02926, S02934, S04355, S02922, S02896 | ไม่มี | 128 | True | คำตอบ<br>ตามมาตรา 1262 ของประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 1270 ของประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 1239 ของประมวลกฎหมายแพ่งและพาณิชย์ และมาตรา 176 ของพระราชบัญญัติบริษัทมหาชนจำกัด พ.ศ. 2535 มาตรา 176 ผู้ชำระบัญชีต้องนำความจดทะเบียนภายในสิบสี่วันนับแต่วันลงมติหรือออกคำบังคับ<br><br>คำถาม |
