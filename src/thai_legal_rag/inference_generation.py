"""Warm, token-streaming inference for the Thai legal RAG demo.

Transformers and torch are imported only when a model is loaded, so the
inference API remains importable in lightweight environments.
"""

from __future__ import annotations

import queue
import re
import threading
from time import perf_counter
from typing import Any, Mapping, Sequence

from .config import RunConfig
from .generation import ABSTAIN_TEXT, _CITATION_RE, _make_prompt


DEFAULT_MODEL_ID = RunConfig.generator_id
DEFAULT_MAX_INPUT_TOKENS = RunConfig.max_input_tokens
LIVE_SYSTEM_PROMPT = (
    "ตอบเป็นภาษาไทยและตอบเฉพาะคำถามโดยใช้ข้อเท็จจริงจากตัวบทที่ให้มาเท่านั้น "
    "อย่าทวนคำถาม อย่าคัดลอกหัวข้อหรือตัวบท และอย่าเพิ่มเหตุผลหรือข้อความที่ตัวบทไม่รองรับ "
    "ใส่รหัสอ้างอิงรูปแบบ [Sxxxxx] ที่ปรากฏในตัวบทไว้หลังข้อสรุปที่รหัสนั้นรองรับ "
    "ใช้ได้เฉพาะรหัสที่เห็นในตัวบท ห้ามเดาหรือสร้างรหัส "
    "หากตัวบทไม่รองรับคำตอบ ให้ตอบเพียง 'หลักฐานไม่เพียงพอ' โดยไม่ใส่รหัสอ้างอิง"
)

_ANSWER_HEADINGS = {"คำตอบ", "answer"}
_SCAFFOLD_HEADINGS = {
    "คำถาม", "question", "ตัวบท", "หลักฐาน", "evidence", "เหตุผล", "reason",
    "อ้างอิง", "ตัวบทอ้างอิง", "เอกสารอ้างอิง", "references", "reference",
    "หมายเหตุ", "คำชี้แจง", "สรุป", "context", "sources",
}
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?P<name>คำตอบ|คำถาม|ตัวบทอ้างอิง|เอกสารอ้างอิง|ตัวบท|หลักฐาน|เหตุผล|อ้างอิง|หมายเหตุ|คำชี้แจง|สรุป|"
    r"answer|question|evidence|reason|references?|context|sources)"
    r"(?:\*\*)?\s*(?::|：\s*)(?P<inline>.*)$|"
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?P<bare>คำตอบ|คำถาม|ตัวบทอ้างอิง|เอกสารอ้างอิง|ตัวบท|หลักฐาน|เหตุผล|อ้างอิง|หมายเหตุ|คำชี้แจง|สรุป|"
    r"answer|question|evidence|reason|references?|context|sources)(?:\*\*)?\s*$",
    flags=re.IGNORECASE,
)


def _heading(line: str) -> tuple[str, str | None] | None:
    match = _HEADING_RE.match(line)
    if not match:
        return None
    name = (match.group("name") or match.group("bare") or "").casefold()
    return name, match.group("inline")


def _normalize_echo(value: str) -> str:
    return " ".join(value.strip().strip("\"'“”‘’「」『』").split()).casefold()


def _is_abstention(body: str) -> bool:
    normalized = body.lstrip()
    return normalized.startswith((ABSTAIN_TEXT, "ข้อมูลไม่เพียงพอ", "ไม่ทราบ"))


def _remove_echoed_question(body: str, question: str) -> str:
    lines = body.splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is not None and _normalize_echo(lines[first]) == _normalize_echo(question):
        del lines[first]
    return "\n".join(lines).strip()


def extract_answer_body(raw_answer: str, question: str) -> str:
    """Remove echoed prompt scaffolding while keeping the generated answer prose."""
    raw = raw_answer.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.splitlines()
    headings = [(index, _heading(line)) for index, line in enumerate(lines)]
    answer_indexes = [
        index for index, header in headings
        if header is not None and header[0] in _ANSWER_HEADINGS
    ]

    if answer_indexes:
        for answer_index in answer_indexes:
            header = _heading(lines[answer_index])
            assert header is not None
            body_lines = [header[1]] if header[1] else []
            for line in lines[answer_index + 1:]:
                next_header = _heading(line)
                if next_header is not None and next_header[0] in _SCAFFOLD_HEADINGS | _ANSWER_HEADINGS:
                    break
                body_lines.append(line)
            body = _remove_echoed_question("\n".join(body_lines), question)
            if not body:
                continue
            if _is_abstention(body):
                return ABSTAIN_TEXT
            return body
        return ""

    first_scaffold = next(
        ((index, header) for index, header in headings
         if header is not None and header[0] in _SCAFFOLD_HEADINGS),
        None,
    )
    if first_scaffold is not None:
        scaffold_index, _ = first_scaffold
        prefix = "\n".join(lines[:scaffold_index]).strip()
        if not prefix:
            return ""
        body = prefix
    else:
        body = raw.strip()
    body = _remove_echoed_question(body, question)
    if _normalize_echo(body) == _normalize_echo(question):
        return ""
    return ABSTAIN_TEXT if _is_abstention(body) else body


def _is_copied_statute(body: str, cited_ids: Sequence[str], corpus: Mapping[str, Mapping[str, Any]]) -> bool:
    citation_only = re.fullmatch(r"\s*(?:\[S\d{5}\]\s*)+", body, flags=re.ASCII)
    if citation_only:
        return True
    if len(cited_ids) != 1:
        return False
    text_without_citations = _CITATION_RE.sub("", body).strip()
    section = corpus.get(cited_ids[0], {}).get("section_content")
    return bool(section) and _normalize_echo(text_without_citations) == _normalize_echo(str(section))


def citation_audit(
    raw_answer: str | None,
    prompt_ids: Sequence[str],
    corpus: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return raw citation IDs and those absent from prompt or corpus."""
    cited_ids = list(dict.fromkeys(_CITATION_RE.findall(raw_answer or "")))
    allowed = set(prompt_ids).intersection(corpus)
    invalid_ids = [statute_id for statute_id in cited_ids if statute_id not in allowed]
    return cited_ids, invalid_ids


def canonicalize_answer(
    raw_answer: str | None,
    question: str,
    prompt_ids: Sequence[str],
    corpus: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Return a citation-validated answer and its cited corpus rows."""
    if raw_answer is None:
        return ABSTAIN_TEXT, []
    body = extract_answer_body(raw_answer, question)
    if not body or _is_abstention(body):
        return ABSTAIN_TEXT, []

    cited_ids = list(dict.fromkeys(_CITATION_RE.findall(body)))
    allowed = set(prompt_ids).intersection(corpus)
    if (
        not cited_ids
        or any(statute_id not in allowed for statute_id in cited_ids)
        or _is_copied_statute(body, cited_ids, corpus)
    ):
        return ABSTAIN_TEXT, []

    legal_citations = [
        {
            "id": statute_id,
            "law_name": str(corpus[statute_id]["law_name"]),
            "section_num": str(corpus[statute_id]["section_num"]),
            "text": str(corpus[statute_id]["section_content"]),
        }
        for statute_id in cited_ids
    ]
    return body, legal_citations


class LiveGenerator:
    """A loaded causal language model that streams decoded answer chunks."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str = DEFAULT_MODEL_ID,
        device: Any = "cpu",
        max_new_tokens: int = RunConfig.max_new_tokens,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        revision: str | None = None,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be at least 1")
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.revision = revision
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

    @classmethod
    def load(
        cls,
        device: Any = "auto",
        revision: str | None = None,
        max_new_tokens: int = RunConfig.max_new_tokens,
    ) -> "LiveGenerator":
        """Load the default Typhoon instruct checkpoint onto the selected device."""
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .pipeline import _resolve_model_revision

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = DEFAULT_MODEL_ID
        resolved_revision = _resolve_model_revision(HfApi(), model_id, revision)
        kwargs = {"revision": resolved_revision}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            **kwargs,
        ).to(device).eval()
        return cls(model, tokenizer, model_id, device, max_new_tokens, revision=resolved_revision)

    def events(
        self,
        question: str,
        retrieved_ids: Sequence[str],
        corpus: Mapping[str, Mapping[str, Any]],
        cancelled: threading.Event,
    ):
        """Yield text deltas and a final result event for one question."""
        if cancelled.is_set():
            return
        input_ids, included, omitted = _make_prompt(
            question,
            retrieved_ids,
            corpus,
            self.tokenizer,
            self.max_input_tokens,
            self.device,
            system_prompt=LIVE_SYSTEM_PROMPT,
        )
        input_tokens = int(input_ids.shape[-1])
        if not included:
            yield {
                "type": "result",
                "raw_answer": None,
                "prompt_ids": included,
                "omitted_ids": omitted,
                "cited_ids": [],
                "invalid_ids": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "generation_ms": 0.0,
                "ttft_ms": 0.0,
            }
            return

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class CancelOnEvent(StoppingCriteria):
            def __call__(self, _input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
                return cancelled.is_set()

        class ThaiTextIteratorStreamer(TextIteratorStreamer):
            """Decode the whole token prefix but publish stable new text eagerly.

            Unlike TextStreamer’s space-based word buffering, this works for
            Thai text without spaces. Decoding the accumulated token IDs keeps
            SentencePiece byte fallback intact; replacement characters are
            held until the byte sequence is complete.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._decoded = ""
                self._emitted = ""

            def put(self, value: Any) -> None:
                if len(value.shape) > 1:
                    if value.shape[0] > 1:
                        raise ValueError("TextStreamer only supports batch size 1")
                    value = value[0]
                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return
                self.token_cache.extend(value.tolist())
                decoded = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
                self._decoded = decoded
                if decoded.startswith(self._emitted):
                    stable = decoded[:-1] if decoded.endswith("\ufffd") else decoded
                    piece = stable[len(self._emitted):]
                    if piece:
                        self._emitted = stable
                        self.on_finalized_text(piece)

            def end(self) -> None:
                # Flush any held final character at stream completion.
                piece = self._decoded[len(self._emitted):] if self._decoded.startswith(self._emitted) else ""
                self.token_cache = []
                self.next_tokens_are_prompt = True
                self.on_finalized_text(piece, stream_end=True)

        streamer = ThaiTextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.1,
        )
        worker_errors: list[Exception] = []
        output_tokens = 0
        generated_answer: str | None = None
        answer_parts: list[str] = []
        first_token_at: float | None = None
        generation_started = 0.0

        def generate() -> None:
            nonlocal generation_started, output_tokens, generated_answer
            try:
                generation_started = perf_counter()
                with torch.inference_mode():
                    output = self.model.generate(
                        input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        do_sample=False,
                        use_cache=True,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([CancelOnEvent()]),
                    )
                if output is not None:
                    output_tokens = max(0, int(output.shape[-1]) - input_tokens)
                    generated_answer = self.tokenizer.decode(
                        output[0, input_tokens:], **streamer.decode_kwargs,
                    )
            except Exception as exc:
                worker_errors.append(exc)
                # Ensure the consumer wakes even when generate fails before the
                # streamer receives its normal end-of-stream marker.
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=generate, name="thai-legal-generation")
        thread_started = False
        try:
            thread.start()
            thread_started = True
            while True:
                try:
                    piece = next(streamer)
                except StopIteration:
                    break
                except (queue.Empty, TimeoutError):
                    if not thread.is_alive():
                        break
                    continue
                if piece:
                    if first_token_at is None:
                        first_token_at = perf_counter()
                    answer_parts.append(piece)
                    yield {"type": "delta", "text": piece}
            thread.join()
            if worker_errors:
                raise worker_errors[0]
        finally:
            # Closing a consumer early also stops generation at the next model
            # step and ensures the background thread is not left behind.
            if thread_started and thread.is_alive():
                cancelled.set()
                thread.join()

        # Stream deltas are best-effort incremental text. Decode the returned
        # token suffix once for the authoritative answer and citation audit.
        raw_answer = generated_answer if generated_answer is not None else "".join(answer_parts)
        cited_ids, invalid_ids = citation_audit(raw_answer, included, corpus)
        finished = perf_counter()
        yield {
            "type": "result",
            "raw_answer": raw_answer,
            "prompt_ids": included,
            "omitted_ids": omitted,
            "cited_ids": cited_ids,
            "invalid_ids": invalid_ids,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_ms": max(0.0, (finished - generation_started) * 1000),
            "ttft_ms": (
                max(0.0, (first_token_at - generation_started) * 1000)
                if first_token_at is not None else None
            ),
        }
