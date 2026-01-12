from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import requests

@dataclass(frozen=True)
class LLMResponse:
    text: str

class LLMClient:
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError

class HeuristicLLM(LLMClient):
    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text="")

class OpenAICompatibleClient(LLMClient):
    def __init__(self, *, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise financial QA assistant. Follow the required output format strictly."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text)

class GeminiClient(LLMClient):
    def __init__(self, *, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("Install google-generativeai to use Gemini: pip install google-generativeai") from e
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(model)

    def generate(self, prompt: str) -> LLMResponse:
        resp = self._model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 256},
        )
        return LLMResponse(text=getattr(resp, "text", "") or "")
