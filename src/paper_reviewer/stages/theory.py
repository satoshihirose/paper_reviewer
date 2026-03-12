"""Stage: Theory, Assumptions and Proofs evaluation (vision-based, all pages at once).

NeurIPS Checklist — Theory, Assumptions and Proofs criterion:
  If you are including theoretical results, did you state the full set of assumptions
  of all theoretical results, and did you include complete proofs of all theoretical results?
  All assumptions should be clearly stated or referenced in the statement of any theorems.
  The proofs can appear in the main paper or supplemental material; if supplemental,
  authors are encouraged to provide a short proof sketch.
"""

from __future__ import annotations

import logging

from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import ChecklistFinding, PaperDocument, TheoryResult
from paper_reviewer.stages._json_utils import pdf_to_images, vision_chat

log = logging.getLogger(__name__)

_PROMPT = """\
You are an expert machine learning researcher conducting a professional, neutral, and fair peer review. \
Evaluate the following NeurIPS Paper Checklist criterion with balanced judgment — neither overly lenient nor overly harsh. \
This checklist is designed to encourage best practices for responsible machine learning research, \
addressing issues of reproducibility, transparency, research ethics, and societal impact.

**Theory, Assumptions and Proofs**: If you are including theoretical results, did you state the full set \
of assumptions of all theoretical results, and did you include complete proofs of all theoretical results? \
All assumptions should be clearly stated or referenced in the statement of any theorems. \
The proofs can either appear in the main paper or the supplemental material, but if they appear in the \
supplemental material, authors are encouraged to provide a short proof sketch to provide intuition. \
Informal proofs in the paper should be complemented by formal proofs in the appendix. \
Theorems and Lemmas should be properly referenced.

The images above are ALL pages of the paper, in order.

IMPORTANT: If the paper contains NO theoretical results (theorems, lemmas, propositions, proofs), \
answer NA for the overall verdict and explain that this criterion is not applicable.

For papers WITH theoretical results, evaluate the following aspects:
1. Are all assumptions of each theorem/lemma clearly stated or referenced?
2. Are complete proofs provided (in main paper or appendix/supplemental)?
3. If proofs are in supplemental material, is a proof sketch provided in the main paper?
4. Are theorems and lemmas properly numbered and referenced?
5. Is the relationship between theoretical results and related literature discussed?

Give an overall verdict:
- Yes: all theoretical results have clearly stated assumptions and complete proofs
- Partial: most results are properly supported but some assumptions or proofs are missing/incomplete
- No: theoretical results are present but assumptions or proofs are substantially missing
- NA: the paper contains no theoretical results (most empirical ML papers)

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
  "issues": ["<specific missing assumptions, proofs, or other gaps>"]
}
"""


def run(doc: PaperDocument, llm: OllamaClient) -> PaperDocument:
    if not doc.pdf_path:
        log.warning("Theory: no pdf_path, skipping")
        doc.theory_result = TheoryResult(
            overall_verdict="NA",
            overall_summary="PDF not available for vision-based theory check.",
            reasoning="",
        )
        return doc

    log.info("Theory: converting PDF to images — %s", doc.pdf_path)
    images = pdf_to_images(doc.pdf_path)
    log.info("Theory: %d pages, sending to LLM (%s)", len(images), llm.model)

    try:
        data = vision_chat(llm.model, _PROMPT, images)

        findings = [
            ChecklistFinding(
                text=f.get("text", ""),
                status=f.get("status", "none"),
            )
            for f in data.get("findings", [])
        ]

        doc.theory_result = TheoryResult(
            overall_verdict=data.get("overall_verdict", "NA"),
            overall_summary=data.get("overall_summary", ""),
            reasoning=data.get("reasoning", ""),
            findings=findings,
            issues=data.get("issues", []),
        )
        log.info(
            "Theory: verdict=%s findings=%d issues=%d",
            doc.theory_result.overall_verdict,
            len(findings),
            len(doc.theory_result.issues),
        )

    except Exception as exc:
        log.error("Theory: failed — %s", exc)
        doc.theory_result = TheoryResult(
            overall_verdict="NA",
            overall_summary=f"評価中にエラーが発生しました: {exc}",
            reasoning="",
        )

    return doc
