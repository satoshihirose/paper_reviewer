"""Stage: Limitations evaluation (vision-based, all pages at once).

NeurIPS Checklist — Limitations criterion:
  The authors are encouraged to create a separate "Limitations" section in their paper.
  The paper should point out any strong assumptions and how robust the results are to
  violations of these assumptions. Scope of claims, failure cases, and potential negative
  societal impacts should be honestly acknowledged.
"""

from __future__ import annotations

import logging

from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import ChecklistFinding, LimitationsResult, PaperDocument
from paper_reviewer.stages._json_utils import pdf_to_images, vision_chat

log = logging.getLogger(__name__)

_PROMPT = """\
You are an expert machine learning researcher conducting a professional, neutral, and fair peer review. \
Evaluate the following NeurIPS Paper Checklist criterion with balanced judgment — neither overly lenient nor overly harsh. \
This checklist is designed to encourage best practices for responsible machine learning research, \
addressing issues of reproducibility, transparency, research ethics, and societal impact.

**Limitations**: The authors are encouraged to create a separate "Limitations" section in their paper. \
The paper should point out any strong assumptions and how robust the results are to violations of these assumptions \
(e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). \
The authors should reflect on the scope of the claims (e.g., if the approach was only tested on a few datasets or with a few runs). \
Authors should honestly acknowledge limitations; reviewers will not penalize for transparency. \
NA is appropriate only if the paper has no limitations whatsoever (extremely rare).

The images above are ALL pages of the paper, in order.

Evaluate the following specific aspects:
1. Does the paper have a dedicated "Limitations" section (or equivalent discussion)?
2. Does it discuss the scope of claims (e.g., datasets, domains, scale tested)?
3. Does it address strong assumptions and robustness to their violation?
4. Does it reflect on potential failure cases or edge cases?
5. Does it discuss potential negative societal impacts or misuse risks?

Give an overall verdict:
- Yes: limitations are thoroughly and honestly discussed
- Partial: some limitations mentioned but coverage is incomplete or superficial
- No: no meaningful limitations discussion despite clear limitations existing
- NA: the paper genuinely has no notable limitations (extremely rare; justify carefully)

Reply with ONLY valid JSON in the following format (no markdown fences, no extra text):

{
  "overall_verdict": "Yes" | "Partial" | "No" | "NA",
  "overall_summary": "<2-3 sentence human-readable summary>",
  "reasoning": "<step-by-step reasoning for the verdict>",
  "findings": [
    {
      "text": "<description of what was found for this aspect>",
      "status": "ok" | "warn" | "ng" | "none"
    }
  ],
  "issues": ["<specific gaps or missing limitations that should be discussed>"]
}
"""


def run(doc: PaperDocument, llm: OllamaClient) -> PaperDocument:
    if not doc.pdf_path:
        log.warning("Limitations: no pdf_path, skipping")
        doc.limitations_result = LimitationsResult(
            overall_verdict="NA",
            overall_summary="PDF not available for vision-based limitations check.",
            reasoning="",
        )
        return doc

    log.info("Limitations: converting PDF to images — %s", doc.pdf_path)
    images = pdf_to_images(doc.pdf_path)
    log.info("Limitations: %d pages, sending to LLM (%s)", len(images), llm.model)

    try:
        data = vision_chat(llm.model, _PROMPT, images)

        findings = [
            ChecklistFinding(
                text=f.get("text", ""),
                status=f.get("status", "none"),
            )
            for f in data.get("findings", [])
        ]

        doc.limitations_result = LimitationsResult(
            overall_verdict=data.get("overall_verdict", "NA"),
            overall_summary=data.get("overall_summary", ""),
            reasoning=data.get("reasoning", ""),
            findings=findings,
            issues=data.get("issues", []),
        )
        log.info(
            "Limitations: verdict=%s findings=%d issues=%d",
            doc.limitations_result.overall_verdict,
            len(findings),
            len(doc.limitations_result.issues),
        )

    except Exception as exc:
        log.error("Limitations: failed — %s", exc)
        doc.limitations_result = LimitationsResult(
            overall_verdict="NA",
            overall_summary=f"評価中にエラーが発生しました: {exc}",
            reasoning="",
        )

    return doc
