"""Ollama LLM client wrapper."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import ollama

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model: str = "qwen3:8b", name: str = ""):
        self.model = model
        self.name = name or model
        self._check_connection()

    def _check_connection(self) -> None:
        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama. Make sure Ollama is running (`ollama serve`).\n"
                f"Error: {e}"
            ) from e

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        log.info("[%s] Ollama request: prompt=%d chars", self.name, len(prompt))
        t = time.perf_counter()
        response = ollama.chat(model=self.model, messages=messages, think=False)
        elapsed = time.perf_counter() - t
        prompt_eval_ms = response.get("prompt_eval_duration", 0) / 1e6
        eval_ms = response.get("eval_duration", 0) / 1e6
        total_ms = response.get("total_duration", 0) / 1e6
        eval_count = response.get("eval_count", 0)
        log.info(
            "[%s] Ollama response: %.1fs | prompt_eval=%.0fms eval=%.0fms total=%.0fms eval_tokens=%d",
            self.name, elapsed, prompt_eval_ms, eval_ms, total_ms, eval_count,
        )
        return response["message"]["content"]

    def complete_json(self, prompt: str, system: str = "") -> Any:
        """Call Ollama and parse JSON from the response."""
        full_system = (system + "\n\n" if system else "") + (
            "Always respond with valid JSON only. Do not include any explanation outside the JSON."
        )
        raw = self.complete(prompt, system=full_system)

        # Try to extract JSON from markdown code blocks first
        code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", raw)
        if code_block:
            raw = code_block.group(1)

        # Strip leading/trailing whitespace and try direct parse
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object or array in the response
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")

    def complete_with_truncation(
        self, prompt: str, system: str = "", max_chars: int = 12000
    ) -> str:
        """Truncate prompt if it exceeds max_chars to avoid context overflow."""
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n\n[... truncated ...]"
        return self.complete(prompt, system=system)

    def complete_json_with_truncation(
        self, prompt: str, system: str = "", max_chars: int = 12000
    ) -> Any:
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n\n[... truncated ...]"
        return self.complete_json(prompt, system=system)
