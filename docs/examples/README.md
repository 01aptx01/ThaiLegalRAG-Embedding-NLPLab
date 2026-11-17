# ตัวอย่างและการรันซ้ำ

รอบตัวอย่างใช้คำถาม `pilot-04144` และ `test-00016` เปรียบเทียบทั้งสี่วิธีค้น เปิดตารางที่รันไว้ใน Notebook ด้วย Python kernel ที่มี `pandas` และ `ipykernel` โดยทำงานจาก repository หรือ `docs/examples/` [Notebook พร้อมตาราง](output_preview.ipynb) · [รายงานผลทั้งสองข้อ](run/cases.md) · [ตัวอย่างการคำนวณ](../../README.md#ตัวอย่างผลรันจริง)

## เตรียมดัชนีและส่งออกตัวอย่าง

เริ่มจากติดตั้งโปรเจกต์และสร้างดัชนีตาม [คู่มือเริ่มใช้งาน](../getting-started.md) สร้างดัชนีจาก revision เดิมและตั้ง `PYTHONHASHSEED=42` ไว้ก่อน build เมื่อต้องการทำซ้ำตัวอย่างที่สร้างไว้

การส่งออกแบบเต็มต้องใช้ดัชนีที่มีทุกวิธี หากต้องการสร้างคำตอบ ให้เตรียม Typhoon ด้วย revision ที่ตรึงไว้ด้วย สำหรับผู้ที่ยังไม่มี BGE index สามารถเริ่มจากดัชนี R1–R3 ที่สร้างด้วย `--no-dense` แล้วใช้ตัวอย่างย่อย 72 มาตราด้านล่าง

```bash
thai-legal-rag-web build-index --help
python examples/export_walkthrough.py --help
```

คำสั่งแรกแสดงตัวเลือกสำหรับสร้างดัชนี คำสั่งที่สองแสดงตัวเลือกส่งออกตัวอย่าง ตัวอย่างการส่งออกคำถามสองข้อ:

```bash
python examples/export_walkthrough.py \
  --index-dir path/to/index \
  --output-dir path/to/new-example-run \
  --question-id pilot-04144 \
  --question-id test-00016
```

ให้แทน `path/to/index` ด้วยดัชนีที่สร้างไว้ และระบุ `--output-dir` เป็นโฟลเดอร์ใหม่ที่ยังไม่มีอยู่ ต้องระบุ `--question-id` อย่างน้อยหนึ่งครั้งและอาจระบุซ้ำเพื่อเลือกหลายข้อ ตัวอย่างนี้เลือกหนึ่งข้อจาก pilot และอีกข้อจาก test ตัวส่งออกใช้ CPU โดยปริยาย สำหรับดัชนีเต็มที่มี BGE เลือก `--device` ให้ตรงกับอุปกรณ์และชนิดตัวเลขของดัชนี หากต้องการส่งออกรุ่น pilot/test หรือรุ่น Generator ที่กำหนดเอง ให้ใช้ `--pilot-revision`, `--test-revision` หรือ `--generator-revision` ตาม SHA ที่ต้องการทำซ้ำ

### เดโม 4 วิธีค้น โดยไม่ต้องสร้าง BGE index ล่วงหน้า

คำสั่งนี้ใช้ revision เดียวกับรายงานตัวอย่าง เตรียมดัชนี R1–R3 จากคลังเต็ม แล้วคำนวณ BGE-M3 เฉพาะมาตราที่ใช้สาธิต

```bash
PYTHONHASHSEED=42 thai-legal-rag-web build-index \
  --output inference_index --device cpu --no-dense \
  --corpus-revision 069e774f015a50e801dbe2055a29be580c54a83c

python examples/export_walkthrough.py \
  --index-dir inference_index \
  --output-dir path/to/new-subset-run \
  --question-id pilot-04144 \
  --question-id test-00016 \
  --sample-statutes 72 \
  --bge-revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --generator-revision 7513a054da229b9dde7c8cd1a7da57baa36ac59b \
  --generator-bfloat16 --max-new-tokens 128
```

รอบนี้รวมมาตราอ้างอิงกับสิบอันดับแรกของ R1–R3 ต่อข้อ แล้วสุ่มเติมด้วย seed 42 ให้ครบ 72 มาตรา ทั้งสี่วิธีจัดอันดับบนชุดเดียวกัน ส่วน R1–R3 ใช้เวกเตอร์ที่เตรียมจากคลังเต็ม คำตอบ Typhoon ใช้ `bfloat16` และเพดาน 128 tokens [รุ่นข้อมูลและขอบเขตผลรัน](run/manifest.json)

## เปิดไฟล์ผลรัน

| ไฟล์ | ใช้ดูอะไร |
| --- | --- |
| [`manifest.json`](run/manifest.json) | รุ่นข้อมูลและโมเดล ขอบเขตตัวบท และตัวเลือกการรัน |
| [`dataset.json`](run/dataset.json) | แถวคำถาม มาตราอ้างอิง และตัวบทฉบับเต็มตามรหัสใน `prompt_statutes` |
| [`embeddings.json`](run/embeddings.json) | การตัดคำ ขนาดเวกเตอร์ และพิกัดตัวอย่างของคำถามกับตัวบท |
| [`rankings.json`](run/rankings.json) | คะแนน cosine อันดับมาตรา และคะแนนรายข้อ |
| [`generation/generation_rows.jsonl`](run/generation/generation_rows.jsonl) | ข้อความคำตอบ ข้อมูล prompt และรหัสอ้างอิง ภายในโฟลเดอร์ `generation/` จะมีตารางสรุปและแบบฟอร์มตรวจคำตอบเมื่อเปิด generator |
| [`audit/audit_scope.json`](run/audit/audit_scope.json) | ขอบเขตแถวตรวจสอบที่เก็บไว้ คู่กับไฟล์การจับคู่และคัดคำถามใน `audit/` |
| [`cases.md`](run/cases.md) | รายงานรายคำถามสำหรับเทียบอันดับ ผลค้น และคำตอบของแต่ละวิธี |

อ้างอิงกติกาของแต่ละขั้นใน [คู่มือข้อมูล](../data.md) · [คู่มือ retrieval](../retrieval.md) · [คู่มือ generation](../generation.md) · [คู่มือการประเมิน](../evaluation.md)
