# โครงสร้างโปรเจกต์ Thai Legal RAG

คู่มือนี้อธิบายหน้าที่ของไฟล์และโฟลเดอร์ที่เก็บใน Git พร้อมจุดเริ่มอ่านโค้ดตามงานที่ต้องทำ สำหรับคำสั่งติดตั้งและรัน ดู [เริ่มใช้งาน](getting-started.md)

## โฟลเดอร์หลัก

| โฟลเดอร์ | ใช้ทำอะไร |
| --- | --- |
| [`src/`](../src/) | เก็บแพ็กเกจ Python ที่ติดตั้งได้จากรากโปรเจกต์ โดยโค้ดอยู่ใน [`src/thai_legal_rag/`](../src/thai_legal_rag/) ทั้งคำสั่งทดลองและเซิร์ฟเวอร์ inference |
| [`frontend/`](../frontend/) | เก็บหน้าเว็บ TypeScript และการตั้งค่า Vite โค้ดที่ผู้ใช้โต้ตอบอยู่ใน [`frontend/src/`](../frontend/src/) |
| [`notebooks/`](../notebooks/) | เก็บ Notebook ต้นฉบับสำหรับทำการทดลองทีละขั้น |
| [`examples/`](../examples/) | เก็บสคริปต์สร้างตัวอย่างอธิบายเส้นทางจากข้อมูลไปถึงคำตอบ |
| [`tests/`](../tests/) | เก็บชุดทดสอบ Python สำหรับส่วนทดลอง ดัชนี และ API |
| [`docs/`](./) | เก็บคู่มือ ตัวอย่างผลรัน และภาพหน้าจอสำหรับอ่านประกอบ |

## เริ่มอ่านโค้ดตามงาน

| งาน | ลำดับที่แนะนำ |
| --- | --- |
| รันการทดลอง `pilot` หรือ `test` | [`pyproject.toml`](../pyproject.toml) ผูกคำสั่ง `thai-legal-rag run` กับ [`cli.py`](../src/thai_legal_rag/cli.py) จากนั้น [`pipeline.py`](../src/thai_legal_rag/pipeline.py) เรียกส่วนเตรียมข้อมูล ค้นมาตรา และสร้างคำตอบตามการตั้งค่า |
| รวมคะแนนจากคนตรวจ | `thai-legal-rag review` รับตำแหน่งโฟลเดอร์ผลรันผ่าน [`cli.py`](../src/thai_legal_rag/cli.py) แล้วให้ [`review.py`](../src/thai_legal_rag/review.py) ตรวจว่าคำถามใดให้คะแนนครบทุกวิธีก่อนสรุปผล |
| ไล่คำถามจากหน้าเว็บถึงคำตอบ | [`main.ts`](../frontend/src/main.ts) ส่งคำถามไปที่ `POST /api/infer` ใน [`web.py`](../src/thai_legal_rag/web.py) เซิร์ฟเวอร์ค้นตัวบทผ่าน [`inference_index.py`](../src/thai_legal_rag/inference_index.py) แล้วส่งตัวบทไปที่ [`inference_generation.py`](../src/thai_legal_rag/inference_generation.py) เมื่อเปิดการสร้างคำตอบ |
| ตามว่าคะแนนในหน้า Evaluate มาจากไหน | [`retrieval.py`](../src/thai_legal_rag/retrieval.py) เขียน `retrieval_summary.csv` ตอนรันการทดลอง และ [`web.py`](../src/thai_legal_rag/web.py) อ่านไฟล์จากตำแหน่ง `--evaluation-run` ให้หน้าเว็บทาง `GET /api/evaluate` |
| ตามผลรันจริงทีละขั้น | เปิด [Notebook แสดงผลตัวอย่าง](examples/output_preview.ipynb) หรือ [รายงานสองคำถาม](examples/run/cases.md) แล้วเปิดสคริปต์ [`export_walkthrough.py`](../examples/export_walkthrough.py) เมื่อต้องการดูวิธีสร้างไฟล์เบื้องหลัง |

## ไฟล์ที่รากโปรเจกต์

| ไฟล์ | หน้าที่ |
| --- | --- |
| [`README.md`](../README.md) | แนะนำโจทย์ของโปรเจกต์ วิธีเปิดเว็บ ตัวอย่างข้อมูลและการคำนวณ cosine จากผลรันจริง พร้อมลิงก์ไปยังคู่มือย่อย |
| [`pyproject.toml`](../pyproject.toml) | ตั้งค่าแพ็กเกจ Python และ setuptools ให้หาโค้ดจาก `src/` ระบุ dependencies ส่วนเสริม `[web]` และคำสั่ง `thai-legal-rag` กับ `thai-legal-rag-web` |
| [`.gitignore`](../.gitignore) | ระบุไฟล์ที่ Git ข้าม เช่น virtual environment, cache, ผลทดลอง ดัชนี และไฟล์ที่ frontend สร้างตอน build |
| [`LICENSE`](../LICENSE) | ระบุเงื่อนไข MIT License ของโค้ด สำหรับชุดข้อมูลและโมเดลดูเงื่อนไขของผู้เผยแพร่แต่ละราย |

## `src/thai_legal_rag/` — โค้ด Python

| ไฟล์ | หน้าที่และจุดเชื่อมต่อ |
| --- | --- |
| [`__init__.py`](../src/thai_legal_rag/__init__.py) | ประกาศแพ็กเกจ `thai_legal_rag` และกำหนดเลขเวอร์ชันของแพ็กเกจ |
| [`__main__.py`](../src/thai_legal_rag/__main__.py) | จุดเริ่มเมื่อเรียก `python -m thai_legal_rag` ส่งต่อไป `cli.main()` และคืน exit code ให้ shell |
| [`cli.py`](../src/thai_legal_rag/cli.py) | รับคำสั่ง `run` และ `review` สำหรับ `thai-legal-rag` แปลงตัวเลือกของ `run` เป็น `RunConfig` แล้วส่งงานให้ `pipeline.py` |
| [`config.py`](../src/thai_legal_rag/config.py) | `RunConfig` เก็บค่าเริ่มต้นและตรวจค่าพื้นฐาน เช่น mode, device และจำนวนคำถาม เป็นแหล่งค่าร่วมของ evaluation กับ inference |
| [`data.py`](../src/thai_legal_rag/data.py) | กำหนดชุดข้อมูลและ split โหลดตาม revision ปรับข้อความและเลขไทย จับคู่มาตราอ้างอิงกับคลัง เขียนผลตรวจสอบ และเลือกชุดคำถามสำหรับสร้างคำตอบโดยใช้ seed |
| [`generation.py`](../src/thai_legal_rag/generation.py) | จัดตัวบททั้งมาตราลง prompt ตามงบ token ใช้ Typhoon สร้างคำตอบของรอบทดลอง และเขียนผลรายข้อ ตารางสรุปหลักฐาน และแบบฟอร์มให้คนตรวจ |
| [`inference_generation.py`](../src/thai_legal_rag/inference_generation.py) | `LiveGenerator` โหลด Typhoon ไว้ให้ API ใช้ซ้ำ ทยอยส่งข้อความ รองรับการยกเลิกคำขอ และตรวจรหัสอ้างอิง ใช้ฟังก์ชันประกอบ prompt ร่วมกับ `generation.py` |
| [`inference_index.py`](../src/thai_legal_rag/inference_index.py) | สร้างและโหลดดัชนีสำหรับค้นคำถามใหม่ เก็บตัวบท manifest และเมทริกซ์ของ Binary BoW, TF-IDF และ Word2Vec (R1–R3) พร้อมตัวเลือกเพิ่ม BGE-M3 (R4) |
| [`pipeline.py`](../src/thai_legal_rag/pipeline.py) | ควบคุมลำดับงานทดลองตั้งแต่ตั้ง seed และเลือกอุปกรณ์ ระบุรุ่นข้อมูลและโมเดล จนถึงเขียน manifest และสร้าง ZIP เรียก `review.py` เมื่อรวมคะแนนของรอบเดิม |
| [`retrieval.py`](../src/thai_legal_rag/retrieval.py) | สร้างเวกเตอร์ทั้งสี่วิธี จัดอันดับด้วย exact cosine และคำนวณ Recall, Hit, nDCG และ RR/MRR พร้อมผลรายข้อและตารางสรุปสำหรับชุดคำถามที่เลือกมาสร้างคำตอบ |
| [`review.py`](../src/thai_legal_rag/review.py) | อ่าน `review_scored.csv` ตรวจรหัสและช่วงคะแนน แล้วรวมเฉพาะคำถามที่ได้คะแนนครบทุกวิธีลงใน `human_summary.csv` และ `human_evidence_stages.csv` |
| [`web.py`](../src/thai_legal_rag/web.py) | รับคำสั่ง `build-index` และ `serve` สำหรับ `thai-legal-rag-web` ตั้งค่า FastAPI ให้เสิร์ฟเว็บ รายงานสถานะ อ่านคะแนนที่บันทึกไว้ และทยอยส่งผลค้นกับคำตอบเป็น NDJSON |

## `tests/` — ชุดทดสอบ Python

| ไฟล์ | ขอบเขตที่ทดสอบ |
| --- | --- |
| [`test_data.py`](../tests/test_data.py) | การปรับเลขไทยและ Unicode การจับคู่ด้วยชื่อกฎหมายและเลขมาตราหรือข้อความตรงกัน การคัดคำถามซ้ำกับชุด test การบันทึก audit และการตรวจรุ่นกับ schema ของข้อมูล |
| [`test_generation.py`](../tests/test_generation.py) | การจัด prompt ตามงบ token การข้ามมาตราที่ยาวเกิน การตรวจผลจาก chat template และไฟล์ที่ได้จากการสร้างคำตอบด้วยโมเดลจำลอง |
| [`test_inference_generation.py`](../tests/test_inference_generation.py) | เหตุการณ์สตรีมและรหัสอ้างอิง การตอบเมื่อไม่มีหลักฐาน การยกเลิกคำขอ การส่งข้อความไทยติดกันและ byte fallback รวมถึงข้อผิดพลาดจาก worker |
| [`test_inference_index.py`](../tests/test_inference_index.py) | การสร้างและเปิดดัชนี R1–R3 จากคลังจำลอง การตรวจ revision และวิธีค้นที่เปิดอยู่ รวมถึงกรณีคำถามไม่มีคำที่รู้จัก |
| [`test_pipeline.py`](../tests/test_pipeline.py) | คำสั่ง `review` ตั้งแต่นำเข้าคะแนน สร้าง ZIP ใหม่ ล้างสรุปเก่าเมื่อไม่มีข้อที่ได้คะแนนครบ ไปจนถึงปฏิเสธรอบที่ยังสร้างคำตอบไม่เสร็จ |
| [`test_retrieval.py`](../tests/test_retrieval.py) | การจัดอันดับด้วย cosine เมื่อคะแนนเสมอหรือคำถามเป็นเวกเตอร์ศูนย์ สูตรคำนวณคะแนน รูปแบบตารางสรุปว่าง และการตรวจค่า `k` |
| [`test_review.py`](../tests/test_review.py) | การรวมเฉพาะข้อที่ได้คะแนนครบทุกวิธี กรณีแถวว่างหรือไม่มีไฟล์คะแนน การเขียนทับสรุปเดิม และการปฏิเสธรหัสซ้ำ รหัสต่างรอบ หรือคะแนนนอกช่วง |
| [`test_web.py`](../tests/test_web.py) | API ด้วยดัชนีและโมเดลจำลอง ตรวจลำดับเหตุการณ์ NDJSON การตอบเมื่อวิธีค้นหรือโมเดลยังไม่พร้อม และการอ่านตารางผลทดลอง |

## `frontend/` — เว็บสำหรับทดลองถามและตรวจผล RAG

| ไฟล์ | หน้าที่และจุดเชื่อมต่อ |
| --- | --- |
| [`frontend/index.html`](../frontend/index.html) | หน้า HTML เริ่มต้นของ Vite กำหนดภาษาและ viewport เตรียม `<div id="app">` แล้วโหลด `src/main.ts` |
| [`frontend/package.json`](../frontend/package.json) | ระบุ TypeScript และ Vite เป็น dependencies สำหรับพัฒนา พร้อมคำสั่ง `dev`, `build` และ `preview` โดย `build` ตรวจ TypeScript ก่อนสร้างไฟล์สำหรับเสิร์ฟ |
| [`frontend/package-lock.json`](../frontend/package-lock.json) | ล็อกเวอร์ชัน dependencies ที่ `npm ci` ใช้ติดตั้ง รวมถึงแพ็กเกจทางอ้อม |
| [`frontend/src/main.ts`](../frontend/src/main.ts) | สร้างและสลับหน้า Playground, Compare, Evaluate และ History ส่งคำถามให้ API อ่านเหตุการณ์ NDJSON เพื่อแสดงหลักฐานและคำตอบตามลำดับ เชื่อมรหัสอ้างอิงไปยังตัวบท และเก็บ History ในหน่วยความจำของหน้าเว็บ |
| [`frontend/src/style.css`](../frontend/src/style.css) | กำหนดสี ตัวอักษร เมนู การ์ด ตาราง และการจัดวางสำหรับจอหลายขนาด โหลดจาก `main.ts` |
| [`frontend/tsconfig.json`](../frontend/tsconfig.json) | ตั้งให้ตรวจ TypeScript แบบ strict สำหรับ ES2022 และ DOM เฉพาะไฟล์ใน `src/` โดย Vite รับหน้าที่สร้างไฟล์เว็บ |
| [`frontend/vite.config.ts`](../frontend/vite.config.ts) | ตั้ง Vite dev server ให้ส่งคำขอ `/api` ต่อไปยัง backend ที่ `http://127.0.0.1:8000` |

## `examples/` — สคริปต์สร้างตัวอย่างที่ตรวจย้อนกลับได้

| ไฟล์ | หน้าที่และจุดเชื่อมต่อ |
| --- | --- |
| [`examples/dataset_trace.py`](../examples/dataset_trace.py) | ใช้ขั้นตอนเตรียมข้อมูลของแพ็กเกจเพื่อดึงแถวต้นทางของข้อที่เลือก แสดงข้อความสำหรับค้น วิธีจับคู่และมาตราอ้างอิง พร้อมย่อรายงาน audit ให้ตรงกับตัวอย่าง |
| [`examples/embedding_trace.py`](../examples/embedding_trace.py) | ถอดรายละเอียดเวกเตอร์คำถามและมาตราจากดัชนี เช่น คำที่มีค่าใน BoW, น้ำหนัก TF-IDF, คำที่ Word2Vec ใช้เฉลี่ย และพิกัดเวกเตอร์ BGE-M3 |
| [`examples/export_walkthrough.py`](../examples/export_walkthrough.py) | คำสั่งหลักสำหรับสร้างชุดตัวอย่างจากดัชนีที่สร้างไว้ ตรวจรุ่นและลำดับคลัง คำนวณ cosine และอันดับสำหรับคำถามที่เลือก เรียกสคริปต์เก็บรายละเอียด และเขียน JSON กับ `cases.md` รองรับทั้งคลังเต็มและชุดมาตราที่คัดมาสาธิต |
| [`examples/generation_trace.py`](../examples/generation_trace.py) | ตรวจว่าผลค้นครบทุกคู่คำถามกับวิธี และมาตราที่อ้างถึงมีในคลัง จากนั้นเรียก `run_generation()` แล้วตรวจลำดับตัวบทในผลสร้างคำตอบ ใช้เมื่อระบุ `--generator-revision` |
| [`examples/render_cases.py`](../examples/render_cases.py) | ประกอบ `cases.md` จากข้อมูลที่สคริปต์ส่งออกจัดเตรียมไว้ โดยแสดงข้อมูลต้นทาง เวกเตอร์ อันดับ cosine หลักฐาน คำตอบอ้างอิง และคำตอบที่สร้างเมื่อมีผลรัน |

## `notebooks/` — Notebook สำหรับรันการทดลองใน Colab

| ไฟล์ | หน้าที่และจุดเชื่อมต่อ |
| --- | --- |
| [`notebooks/Thai_Legal_RAG_Clean.ipynb`](../notebooks/Thai_Legal_RAG_Clean.ipynb) | Notebook ต้นฉบับสำหรับรันใน Colab ตั้งแต่ติดตั้ง dependency โหลดและจับคู่ข้อมูล สร้างเวกเตอร์ทั้งสี่วิธี วัดผลการค้น สร้างคำตอบและแบบประเมิน ไปจนถึงรวมผลเป็น ZIP ดู [ตารางเทียบเซลล์กับไฟล์ Python](README.md#เทียบกับ-notebook-ต้นฉบับ) |

## `docs/` — คู่มือและผลรันที่บันทึกไว้

ตัวบทและชุดคำถามที่ใช้รันมาจากแหล่งที่ระบุใน [ข้อมูลและการจับคู่มาตรา](data.md) โฟลเดอร์ด้านล่างเก็บคู่มือและผลจากรอบที่บันทึกไว้ให้เปิดตรวจย้อนหลัง

| ตำแหน่ง | ใช้เก็บอะไร |
| --- | --- |
| [`docs/examples/`](examples/) | วิธีทำซ้ำและ Notebook สำหรับเปิดอ่านผลรันตัวอย่างที่บันทึกไว้ |
| [`docs/examples/run/`](examples/run/) | ไฟล์ที่ส่งออกจากรอบสาธิตเดียวกัน ตั้งแต่ข้อมูลต้นทางที่คัดมาจนถึงคำตอบ |
| [`docs/examples/run/audit/`](examples/run/audit/) | ตารางตรวจการจับคู่มาตราและเหตุผลที่คัดคำถามเข้าหรือออกจากรอบสาธิต |
| [`docs/examples/run/generation/`](examples/run/generation/) | ผลสร้างคำตอบ ตารางสรุปหลักฐาน และแบบฟอร์มให้คนตรวจ |
| [`docs/screenshots/`](screenshots/) | ภาพประกอบคู่มือและเอกสารอธิบายที่มาของภาพ |
| [`docs/screenshots/evaluation/`](screenshots/evaluation/) | ชุดคะแนนการค้น R1–R3 ที่ใช้จับภาพหน้า Evaluate |

### คู่มือหลักใน `docs/`

| ไฟล์ | หน้าที่ |
| --- | --- |
| [README.md](README.md) | สารบัญคู่มือและตารางเทียบเซลล์ใน Notebook ต้นฉบับกับไฟล์ Python |
| [getting-started.md](getting-started.md) | ขั้นตอนเตรียมเครื่อง ติดตั้ง เปิดเว็บ สร้างดัชนี และสั่งรันจาก Terminal พร้อมแนวตรวจอาการที่พบบ่อย |
| [data.md](data.md) | อธิบายชุดข้อมูลที่โหลด หน่วยค้นและรหัสมาตรา วิธีจับคู่ และไฟล์ audit ที่ใช้ตรวจตัวอย่างจริง |
| [retrieval.md](retrieval.md) | อธิบาย R1–R4 การแปลงข้อความเป็นเวกเตอร์ การจัดอันดับด้วย cosine และไฟล์ที่อยู่ในดัชนีสำหรับเว็บ |
| [generation.md](generation.md) | อธิบายการเลือกตัวบทใส่ prompt การอ่านรหัสอ้างอิงที่ถูกต้อง/ไม่ถูกต้อง และตัวอย่างคำตอบจากการทดลอง |
| [evaluation.md](evaluation.md) | คู่มือเลือกรอบ pilot/test อ่านเมตริก retrieval และผล generation รวมถึงแบบฟอร์มและคะแนนจากคนตรวจ |
| [inference-web.md](inference-web.md) | คู่มือหน้า Playground, Compare, Evaluate และ History ตลอดจนปลายทาง API และรูปแบบผลที่ทยอยส่งกลับ |
| [repository-structure.md](repository-structure.md) | คู่มือฉบับนี้สำหรับค้นหน้าที่ของไฟล์ใน repo และเลือกจุดเริ่มอ่านโค้ด |

### ผลรันตัวอย่างใน `docs/examples/`

ตัวอย่างนี้ใช้คำถาม pilot และ test อย่างละหนึ่งข้อ เปรียบเทียบวิธีค้นทั้งสี่บนชุดมาตรา 72 รายการจากคลัง 5,127 มาตรา รุ่นข้อมูลและวิธีเลือกมาตราอยู่ใน manifest ของรอบ ไฟล์ `.csv` เป็นตาราง ส่วน `.jsonl` เก็บผลหนึ่งรายการต่อบรรทัด

| ไฟล์ | หน้าที่และรูปแบบข้อมูล |
| --- | --- |
| [examples/README.md](examples/README.md) | วิธีสร้างดัชนีและส่งออกชุดตัวอย่างซ้ำ อธิบายเงื่อนไข revision/seed และแนะนำไฟล์ผลลัพธ์ |
| [examples/output_preview.ipynb](examples/output_preview.ipynb) | Jupyter Notebook ที่อ่านไฟล์ snapshot ใน `run/` ด้วย Python/pandas แล้วแสดงตารางแหล่งข้อมูล พิกัดเวกเตอร์ อันดับ และคำตอบที่สร้างไว้ เซลล์มีผลลัพธ์ที่บันทึกอยู่แล้ว จึงเปิดดูได้โดยไม่ต้องสร้างรอบใหม่ |
| [examples/run/manifest.json](examples/run/manifest.json) | บันทึก commit ของโค้ด รุ่นชุดข้อมูลและโมเดล วิธีค้น ชุดมาตราที่นำมาจัดอันดับ อุปกรณ์ และค่าที่ใช้รัน |
| [examples/run/dataset.json](examples/run/dataset.json) | เก็บตัวอย่างแถวต้นทาง คำถามที่เลือกและที่คัดออก ข้อความมาตราที่เข้า prompt และตำแหน่งไฟล์ผลตรวจที่เกี่ยวข้อง |
| [examples/run/embeddings.json](examples/run/embeddings.json) | เก็บรายละเอียดการแปลงคำถามและมาตราเป็นเวกเตอร์แยกตามวิธี เช่น การตัดคำ ขนาดเวกเตอร์ และพิกัดตัวอย่างก่อนและหลัง normalize |
| [examples/run/rankings.json](examples/run/rankings.json) | เก็บรหัสมาตราอ้างอิง ผลค้นเรียงตามคะแนน อันดับของมาตราอ้างอิง คะแนนรายข้อ และค่าตัวอย่างที่ใช้ตรวจการคำนวณ cosine |
| [examples/run/cases.md](examples/run/cases.md) | รายงาน Markdown ที่รวมข้อมูลต้นทาง เวกเตอร์ อันดับ คำตอบอ้างอิง และคำตอบที่โมเดลสร้าง เปิดอ่านแยกตามคำถามได้ |

#### ไฟล์ตรวจการจับคู่ข้อมูล

| ไฟล์ | หน้าที่และรูปแบบข้อมูล |
| --- | --- |
| [audit/audit_scope.json](examples/run/audit/audit_scope.json) | กำหนดขอบเขต audit ใน snapshot ได้แก่รหัสคำถามที่ตรวจ ข้อที่เลือก/กันออก และรหัสมาตรา gold |
| [audit/question_audit.csv](examples/run/audit/question_audit.csv) | หนึ่งแถวต่อคำถามที่ตรวจ ระบุ split, เหตุผลที่รับเข้าหรือคัดออก และจำนวนรายการอ้างอิง/รหัสที่จับคู่ได้ เช่น `included` หรือ `test_overlap` |
| [audit/mapping_audit.csv](examples/run/audit/mapping_audit.csv) | หนึ่งแถวต่อคู่คำถามกับมาตราอ้างอิง แสดงชื่อกฎหมาย เลขมาตรา สถานะการจับคู่ และ statute ID เพื่อย้อนตรวจว่าเชื่อมไปยังรายการใด |
| [audit/statute_ids.csv](examples/run/audit/statute_ids.csv) | ตารางพจนานุกรมรหัสมาตราที่ปรากฏในตัวอย่าง โยง statute ID กับชื่อกฎหมาย เลขมาตรา และ law code |

#### ไฟล์ผลสร้างคำตอบ

| ไฟล์ | หน้าที่และรูปแบบข้อมูล |
| --- | --- |
| [generation/evidence_stages.csv](examples/run/generation/evidence_stages.csv) | นับแยกตามวิธีค้นว่ามาตราอ้างอิงเข้า prompt หรือหลุดจากขั้นค้น |
| [generation/generation_rows.jsonl](examples/run/generation/generation_rows.jsonl) | ผลหนึ่งบรรทัดต่อคู่คำถามกับวิธีค้น เก็บคำตอบอ้างอิง คำตอบที่สร้าง รหัสตัวบทที่เข้า prompt และที่ถูกตัด รหัสที่อ้าง จำนวน token และเวลา |
| [generation/generation_summary.csv](examples/run/generation/generation_summary.csv) | สรุปผลการสร้างคำตอบตามวิธีค้น เช่น จำนวนคำตอบ สัดส่วนรหัสอ้างอิงที่อยู่ในหลักฐาน สัดส่วนมาตราอ้างอิงที่อยู่ใน prompt อัตราการงดตอบ จำนวน token และเวลา |
| [generation/review_template.csv](examples/run/generation/review_template.csv) | แบบฟอร์มตรวจคำตอบรายรายการ มีคำถาม คำตอบอ้างอิง คำตอบที่สร้าง รหัส citation และข้อความ context พร้อมช่อง `coverage`, `contradiction`, `citation_support`, `note` สำหรับผู้ตรวจกรอก ช่องคะแนนในไฟล์ต้นแบบยังเว้นไว้ |

### ภาพหน้าจอและผลสำหรับหน้า Evaluate

ภาพหน้าเว็บมาจากแอปที่รันจริงบน CPU และตาราง Evaluate อ่านผลจากรอบ pilot 40 ข้อที่รันเฉพาะ R1–R3

| ไฟล์ | หน้าที่/สิ่งที่แสดง |
| --- | --- |
| [screenshots/README.md](screenshots/README.md) | บันทึกที่มาของภาพ สภาพแวดล้อม รุ่นข้อมูลและข้อจำกัดของรอบคะแนนที่ใช้ทำภาพ พร้อมวิธีเปิดตาราง Evaluate ชุดเดียวกัน |
| [screenshots/overview.png](screenshots/overview.png) | หน้า Playground หลัง API และดัชนีพร้อม ก่อนส่งคำถาม ใช้เป็นภาพรวมสถานะเริ่มต้น |
| [screenshots/playground.png](screenshots/playground.png) | ตัวอย่างส่งคำถามเรื่องอายุบรรลุนิติภาวะผ่าน R2 TF-IDF พร้อมรายการหลักฐานที่ค้นพบ |
| [screenshots/compare.png](screenshots/compare.png) | หน้าตารางเปรียบเทียบผลค้นจาก R1–R3 สำหรับคำถามเรื่องผู้เยาว์ทำนิติกรรม |
| [screenshots/evaluate.png](screenshots/evaluate.png) | หน้า Evaluate ที่อ่านตารางสรุปจากไฟล์ CSV ใน `screenshots/evaluation/` |
| [screenshots/history.png](screenshots/history.png) | ประวัติคำถามจากการค้นและการเปรียบเทียบใน browser session เดียวกับภาพ Playground/Compare |
| [screenshots/trace.png](screenshots/trace.png) | แผง Trace ที่เปิดดูรายละเอียดผลค้น R2 จากคำถามใน Playground |
| [screenshots/getting-started.png](screenshots/getting-started.png) | ภาพข้อความ stdout จริงที่จัดให้อ่านง่าย แสดง frontend build และบรรทัดสำคัญจากการสร้างดัชนี |
| [screenshots/configuration.png](screenshots/configuration.png) | ภาพ stdout ของคำสั่ง `thai-legal-rag-web serve --help` สำหรับดูตัวเลือกตั้งค่า |
| [screenshots/contributing.png](screenshots/contributing.png) | ภาพรายชื่อบางส่วนจาก unittest พร้อมสรุปผลและ stdout ของ frontend build ที่ใช้ประกอบหัวข้อพัฒนา |

#### ไฟล์คะแนนที่ใช้จับภาพหน้า Evaluate

| ไฟล์ | หน้าที่และรูปแบบข้อมูล |
| --- | --- |
| [screenshots/evaluation/manifest.json](screenshots/evaluation/manifest.json) | บันทึกรอบ pilot 40 ข้อสำหรับ R1–R3 บน CPU พร้อมรุ่นชุดข้อมูลและรหัสคำถามที่ใช้จับภาพ Evaluate |
| [screenshots/evaluation/retrieval_rows.jsonl](screenshots/evaluation/retrieval_rows.jsonl) | ผลค้นหนึ่งบรรทัดต่อคู่คำถามกับวิธีค้น รวม 120 บรรทัด มีรหัสมาตราอ้างอิง รหัสที่ค้นได้ และคะแนนรายข้อ โดยคอลัมน์ `mrr_at_10` เป็น RR@10 ของข้อนั้น |
| [screenshots/evaluation/retrieval_summary.csv](screenshots/evaluation/retrieval_summary.csv) | ตารางค่าเฉลี่ยคะแนนและเวลาค้นของแต่ละวิธีจาก 40 ข้อ เป็นไฟล์ที่หน้า Evaluate ในภาพอ่านผ่าน API |

## โฟลเดอร์ที่เกิดขึ้นเมื่อรันบนเครื่อง

ค่าเริ่มต้นด้านล่างอ้างจากคำสั่งทดลอง คำสั่งเว็บ และ [`.gitignore`](../.gitignore) ของโปรเจกต์ รุ่นของชุดข้อมูลและโมเดลตรวจได้จาก `manifest.json` ของรอบหรือดัชนีนั้น

| ตำแหน่ง | เกิดขึ้นเมื่อใดและใช้ทำอะไร |
| --- | --- |
| `rag_results/` | `thai-legal-rag run` สร้างโฟลเดอร์ตามชื่อโหมดและเวลารัน มีไฟล์ตรวจสอบข้อมูล ผลรายข้อ ตารางสรุป และผลสร้างคำตอบตามโหมดที่เลือก เมื่อรันสำเร็จจะมี ZIP คู่กับโฟลเดอร์รอบนั้น |
| `inference_index/` | `thai-legal-rag-web build-index` สร้าง `corpus.json`, `manifest.json`, ไฟล์ตัวแปลงคำถาม และเมทริกซ์เวกเตอร์เพื่อให้ `serve` ใช้ซ้ำ เมทริกซ์ของ BGE-M3 จะมีเมื่อสร้างดัชนีพร้อมวิธี R4 |
| `frontend/node_modules/` | `npm ci` ติดตั้งแพ็กเกจ JavaScript ตาม lockfile เพื่อใช้พัฒนาและ build หน้าเว็บ |
| `frontend/dist/` | `npm run build` สร้าง `index.html` และ assets ที่ FastAPI เสิร์ฟเมื่อเปิดเว็บ |
| `.venv/` หรือ `venv/` | สภาพแวดล้อม Python ที่สร้างในเครื่องเพื่อแยก dependencies ของโปรเจกต์ |
