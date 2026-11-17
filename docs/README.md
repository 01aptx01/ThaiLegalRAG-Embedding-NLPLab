# คู่มือโปรเจกต์

## ตามผลรันจริงตั้งแต่ข้อมูลถึงคำตอบ

ดูโค้ดและตารางผลลัพธ์ที่แสดงใต้แต่ละเซลล์ ตั้งแต่ข้อมูลต้นทางไปจนถึงคำตอบ Typhoon [โค้ดและตารางผลบน GitHub](examples/output_preview.md) · [Notebook พร้อมผลที่รันแล้ว](examples/output_preview.ipynb)

เปิดข้อ `pilot-04144` เพื่อดูว่ามาตราที่ชุดข้อมูลระบุขึ้นมาอยู่ลำดับใดในแต่ละวิธี จากนั้นเปิด `test-00016` เพื่อเทียบคำตอบเมื่อทั้งสี่วิธีค้นพบมาตราอ้างอิงเป็นอันดับหนึ่ง [รายงานสองข้อ](examples/run/cases.md) · [คู่มือรันซ้ำ](examples/README.md)

| ช่วงที่ต้องการดู | เอกสารและผลรัน |
| --- | --- |
| คำถามต้นทางและการจับคู่รหัส | [ตัวอย่างแถวข้อมูล](data.md#ตัวอย่างแถวข้อมูลจริง) · [แถวจาก dataset](examples/run/dataset.json) |
| เวกเตอร์และการคำนวณ cosine | [วิธีอ่านพิกัดและผลคูณ](retrieval.md#ตัวอย่างเวกเตอร์และ-cosine-จากการรันจริง) · [เวกเตอร์](examples/run/embeddings.json) · [คะแนนและอันดับ](examples/run/rankings.json) |
| มาตราที่เข้า prompt และคำตอบ Typhoon | [ตัวอย่างอ่านคำตอบ](generation.md#ตัวอย่างอ่านผลจากคำถามเดียวกัน) · [คำตอบดิบ](examples/run/answers.json) |
| คะแนน retrieval และเกณฑ์ให้คนตรวจ | [ตัวอย่างอ่านคะแนน](evaluation.md#ตัวอย่างอ่านคะแนนจากสองข้อจริง) · [แบบฟอร์มตรวจคำตอบ](examples/run/generation/review_template.csv) |

## เอกสารตามหัวข้อ

| เอกสาร | ใช้อ่านตอน |
| --- | --- |
| [เริ่มใช้งาน](getting-started.md) | ติดตั้ง เปิดเว็บครั้งแรก หรือแก้ปัญหาเครื่องรัน |
| [ข้อมูลและการจับคู่มาตรา](data.md) | ตรวจแหล่งข้อมูล วิธีตั้งรหัส และคำถามที่ถูกคัดออก |
| [ตัวอย่างตั้งแต่ข้อมูลถึงคำตอบ](examples/README.md) | ไล่ตามแถวข้อมูลจริง เวกเตอร์สี่แบบ อันดับ cosine และคำตอบ Typhoon |
| [การค้นมาตรา](retrieval.md) | ทำความเข้าใจวิธีค้นทั้งสี่แบบ หรือสร้างดัชนีไว้ใช้กับเว็บ |
| [การสร้างคำตอบ](generation.md) | ตรวจว่าตัวบทใดถูกส่งให้ Typhoon และคำตอบอ้างมาตราอะไร |
| [การทดลองและการประเมินผล](evaluation.md) | รันชุดคำถาม อ่านไฟล์ผล หรือกรอกคะแนนคนตรวจ |
| [หน้าเว็บและ API](inference-web.md) | ใช้แต่ละหน้า หรือเชื่อมต่อกับ API ด้วยโค้ด |

## สองทางเข้าในโค้ด

**การทดลอง** เริ่มที่ `thai-legal-rag run` ใน [cli.py](../src/thai_legal_rag/cli.py) แล้ว [pipeline.py](../src/thai_legal_rag/pipeline.py) จะโหลดข้อมูล เลือกคำถาม รันทั้งสี่วิธี เขียนผล และสร้าง ZIP ของรอบนั้น หากต้องกรอกคะแนนคนตรวจภายหลัง ให้ใช้ `thai-legal-rag review` กับโฟลเดอร์ผลเดิม

**หน้าเว็บ** ใช้ `thai-legal-rag-web build-index` เพื่อสร้างไฟล์ค้นคืนจากคลัง แล้ว `thai-legal-rag-web serve` โหลดไฟล์เหล่านั้นไว้รอคำถามใหม่ [web.py](../src/thai_legal_rag/web.py) เสิร์ฟหน้าเว็บจากไฟล์ที่ build แล้วและรับคำถามผ่าน API การถามบนเว็บจะไม่เพิ่มแถวในไฟล์ benchmark

| ตำแหน่ง | หน้าที่ |
| --- | --- |
| [frontend/src/main.ts](../frontend/src/main.ts) | สลับหน้า ส่งคำถาม อ่านผลทีละบรรทัด และเชื่อมรหัสอ้างอิงไปยังตัวบท |
| [frontend/src/style.css](../frontend/src/style.css) | รูปแบบหน้าจอและการจัดวางบนจอเล็ก |
| [data.py](../src/thai_legal_rag/data.py) | โหลดข้อมูล ตรวจคอลัมน์ จับคู่มาตรา และเลือกคำถาม |
| [retrieval.py](../src/thai_legal_rag/retrieval.py) | จัดอันดับและวัดผลการค้นในการทดลอง |
| [inference_index.py](../src/thai_legal_rag/inference_index.py) | สร้างและเปิดดัชนีสำหรับคำถามใหม่ |
| [generation.py](../src/thai_legal_rag/generation.py) | ประกอบคำสั่งกับตัวบท สร้างคำตอบและไฟล์ผลการทดลอง |
| [inference_generation.py](../src/thai_legal_rag/inference_generation.py) | โหลด Typhoon ไว้ในเซิร์ฟเวอร์และทยอยส่งคำตอบ |
| [review.py](../src/thai_legal_rag/review.py) | ตรวจไฟล์คะแนนและรวมผลจากคนตรวจ |
| [tests/](../tests/) | ทดสอบการจับคู่ อันดับ การใส่หลักฐาน การรับคะแนน และ API |

## เทียบกับ Notebook ต้นฉบับ

เลขในตารางนับทุกเซลล์ตั้งแต่ 1 ใน [Thai_Legal_RAG_Clean.ipynb](../notebooks/Thai_Legal_RAG_Clean.ipynb)

| เซลล์ | งาน | ไฟล์ Python หรือไฟล์ตั้งค่า |
| --- | --- | --- |
| 3–4 | ติดตั้งและกำหนดค่าเริ่มต้น | [pyproject.toml](../pyproject.toml), [config.py](../src/thai_legal_rag/config.py), [cli.py](../src/thai_legal_rag/cli.py) |
| 6, 8–9, 11 | โหลดข้อมูล จับคู่มาตรา และเลือกคำถาม | [data.py](../src/thai_legal_rag/data.py) |
| 13, 15–16, 18–19 | วัดความยาวมาตรา จัดอันดับ และคำนวณคะแนน | [retrieval.py](../src/thai_legal_rag/retrieval.py) |
| 21, 23 | สร้างคำตอบ ตรวจรหัสอ้างอิง และเตรียมแบบประเมิน | [generation.py](../src/thai_legal_rag/generation.py) |
| 25 | นำเข้าคะแนนจากคนตรวจ | [review.py](../src/thai_legal_rag/review.py) |
| 27 | รวมผลเป็น ZIP | [pipeline.py](../src/thai_legal_rag/pipeline.py) |
