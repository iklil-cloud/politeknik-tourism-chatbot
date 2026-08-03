from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.config import LLM_MODEL
from app.ollama_client import OllamaClient


JUDGE_PROMPT_TEMPLATE = """Anda ialah penilai bebas untuk chatbot Sejarah SPM Tingkatan 5.

Nilai jawapan chatbot berdasarkan soalan pengguna dan jawapan rujukan.

Skala markah:
- 1 = sangat lemah
- 2 = lemah
- 3 = sederhana
- 4 = baik
- 5 = sangat baik

Kriteria:
1. correctness: ketepatan fakta berbanding jawapan rujukan.
2. relevance: sejauh mana jawapan menjawab soalan.
3. completeness: kecukupan isi penting.
4. clarity: kejelasan Bahasa Melayu dan susunan jawapan.
5. groundedness: tiada fakta rekaan atau bercanggah dengan rujukan.

Pulangkan JSON sahaja tanpa markdown:
{{
  "correctness": 1-5,
  "relevance": 1-5,
  "completeness": 1-5,
  "clarity": 1-5,
  "groundedness": 1-5,
  "overall": 1-5,
  "reason": "ringkasan sebab dalam satu ayat"
}}

Soalan:
{question}

Jawapan rujukan:
{reference_answer}

Jawapan chatbot:
{prediction}
"""


JUDGE_SCORE_FIELDS = [
    "correctness",
    "relevance",
    "completeness",
    "clarity",
    "groundedness",
    "overall",
]


@dataclass
class JudgeEvaluation:
    correctness: float
    relevance: float
    completeness: float
    clarity: float
    groundedness: float
    overall: float
    reason: str
    raw_response: str

    def to_row(self) -> dict[str, object]:
        return {
            "judge_correctness": self.correctness,
            "judge_relevance": self.relevance,
            "judge_completeness": self.completeness,
            "judge_clarity": self.clarity,
            "judge_groundedness": self.groundedness,
            "judge_overall": self.overall,
            "judge_reason": self.reason,
            "judge_raw_response": self.raw_response,
        }


class LLMJudge:
    def __init__(
        self,
        model: str = LLM_MODEL,
        client: OllamaClient | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.client = client or OllamaClient()
        self.temperature = temperature

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        prediction: str,
    ) -> JudgeEvaluation:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question.strip(),
            reference_answer=reference_answer.strip(),
            prediction=prediction.strip(),
        )
        raw_response = self.client.generate(
            self.model,
            prompt,
            temperature=self.temperature,
            num_predict=500,
        )
        payload = parse_judge_json(raw_response)
        return JudgeEvaluation(
            correctness=coerce_score(payload.get("correctness")),
            relevance=coerce_score(payload.get("relevance")),
            completeness=coerce_score(payload.get("completeness")),
            clarity=coerce_score(payload.get("clarity")),
            groundedness=coerce_score(payload.get("groundedness")),
            overall=coerce_score(payload.get("overall")),
            reason=str(payload.get("reason", "")).strip(),
            raw_response=raw_response,
        )


def parse_judge_json(raw_response: str) -> dict[str, Any]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return JSON: {raw_response}")
        return json.loads(match.group(0))


def coerce_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(5.0, max(0.0, score))
