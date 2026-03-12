"""Shared utilities for vision-based stage modules."""

from __future__ import annotations

import json
import re

import fitz  # pymupdf
import ollama


_DEFAULT_DPI = 100


def pdf_to_images(pdf_path: str, dpi: int = _DEFAULT_DPI) -> list[bytes]:
    """Render each PDF page to a PNG byte string."""
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def vision_chat(model: str, prompt: str, images: list[bytes]) -> dict:
    """Send a vision prompt to Ollama and return the parsed JSON response."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": images}],
        options={"temperature": 0, "num_keep": 0},
        think=False,
    )
    return extract_json(response["message"]["content"])


def _sanitize_json_strings(text: str) -> str:
    """Escape literal control characters inside JSON string values.

    LLMs sometimes emit literal newlines / tabs inside string values,
    which are invalid JSON.  Walk the text character-by-character and
    escape them only when inside a string literal.
    """
    out: list[str] = []
    in_str = False
    i = 0
    while i < len(text):
        c = text[i]
        # Handle backslash escape sequences – skip the next char
        if c == "\\" and in_str and i + 1 < len(text):
            out.append(c)
            i += 1
            out.append(text[i])
        elif c == '"':
            in_str = not in_str
            out.append(c)
        elif in_str and ord(c) < 0x20:
            # Literal control character inside a string – escape it
            if c == "\n":
                out.append("\\n")
            elif c == "\r":
                out.append("\\r")
            elif c == "\t":
                out.append("\\t")
            # else: drop other control chars silently
        else:
            out.append(c)
        i += 1
    return "".join(out)


def extract_json(text: str) -> dict:
    """Extract and parse JSON from an LLM response.

    Strips markdown code fences if present, sanitizes control characters
    inside string values, then calls json.loads().  Falls back to
    json_repair when the output is malformed (e.g. truncated strings).
    """
    from json_repair import repair_json  # lazy import

    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1).strip()
    text = _sanitize_json_strings(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        raise
