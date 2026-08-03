from __future__ import annotations

import hashlib
import re
from typing import Any

import requests

from app.config import OLLAMA_BASE_URL

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, model: str, text: str) -> list[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings") or []
            if not embeddings:
                raise RuntimeError("No embedding returned by Ollama.")
            return embeddings[0]
        except Exception:
            return self._fallback_embed(text)

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        num_predict: int = 800,
    ) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return str(data.get("response", "")).strip()
        except Exception:
            return self._fallback_generate(prompt)

    def _fallback_embed(self, text: str) -> list[float]:
        tokens = TOKEN_RE.findall(text.lower())
        vector = [0.0] * 64
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % len(vector)
            vector[index] += 1.0
        return vector

    def _fallback_generate(self, prompt: str) -> str:
        if "Soalan pengguna:" in prompt:
            question = prompt.split("Soalan pengguna:", 1)[1].strip().splitlines()[0]
        else:
            question = "soalan anda"
        return (
            f"Saya menggunakan maklumat rujukan yang tersedia. Untuk {question}, sila rujuk sumber yang dipaparkan "
            f"di bawah untuk penjelasan yang lebih terperinci."
        )
