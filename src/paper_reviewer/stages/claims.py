"""Stage: Claims evaluation (vision-based, all pages at once).

NeurIPS Checklist — Claims criterion:
  Do the main claims made in the abstract and introduction accurately reflect
  the paper's contributions and scope? Claims should match theoretical and
  experimental results in terms of generalizability. Contributions should be
  clearly stated with assumptions and limitations. Aspirational goals should be
  clearly distinguished from actual achievements.
"""

from __future__ import annotations

import logging

from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import ClaimVerdict, ClaimsResult, PaperDocument
from paper_reviewer.stages._json_utils import pdf_to_images, vision_chat

log = logging.getLogger(__name__)

_PROMPT = """\
You are an expert machine learning researcher conducting a professional, neutral, and fair peer review. \
Evaluate the following NeurIPS Paper Checklist criterion with balanced judgment — neither overly lenient nor overly harsh. \
This checklist is designed to encourage best practices for responsible machine learning research, \
addressing issues of reproducibility, transparency, research ethics, and societal impact.

**Claims**: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
- Claims in the paper should match theoretical and experimental results in terms of how much the results can be expected to generalize.
- The contributions should be clearly stated with appropriate assumptions and limitations.
- Aspirational goals are acceptable if they are clearly distinguished from actual achievements.

The images above are ALL pages of the paper, in order.

Your task:
1. Identify the main claims from the abstract and introduction (aim for 3–6 claims).
2. For each claim, assess whether it is:
   - SUPPORTED: clearly backed by experimental/theoretical results in the paper
   - PARTIALLY_SUPPORTED: some evidence exists but the claim overgeneralizes or lacks key caveats
   - UNSUPPORTED: no evidence found, or the claim contradicts the results
   - UNCLEAR: not enough information to judge
3. Cite specific evidence for each claim (e.g. "Table 2", "Figure 3", "Section 4.2", "page N").
4. Give an overall verdict: Yes / Partial / No
   - Yes: all or nearly all claims are well-supported
   - Partial: some claims are overstated or lack caveats
   - No: multiple major claims are unsupported or misleading

Reply with ONLY valid JSON in the following format (no markdown fences, no extra text):

{
  "overall_verdict": "Yes" | "Partial" | "No",
  "overall_summary": "<2-3 sentence human-readable summary>",
  "reasoning": "<step-by-step reasoning for the verdict>",
  "claims": [
    {
      "text": "<verbatim or close paraphrase of the claim>",
      "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "UNCLEAR",
      "evidence": "<specific evidence: table/figure/section/page numbers>"
    }
  ],
  "issues": ["<list of specific overclaiming or scope issues, if any>"]
}
"""


def run(doc: PaperDocument, llm: OllamaClient) -> PaperDocument:
    if not doc.pdf_path:
        log.warning("Claims: no pdf_path, skipping")
        doc.claims_result = ClaimsResult(
            overall_verdict="NA",
            overall_summary="PDF not available for vision-based claims check.",
            reasoning="",
        )
        return doc

    log.info("Claims: converting PDF to images — %s", doc.pdf_path)
    images = pdf_to_images(doc.pdf_path)
    log.info("Claims: %d pages, sending to LLM (%s)", len(images), llm.model)

    try:
        data = vision_chat(llm.model, _PROMPT, images)

        claims = [
            ClaimVerdict(
                text=c.get("text", ""),
                verdict=c.get("verdict", "UNCLEAR"),
                evidence=c.get("evidence", ""),
            )
            for c in data.get("claims", [])
        ]

        doc.claims_result = ClaimsResult(
            overall_verdict=data.get("overall_verdict", "NA"),
            overall_summary=data.get("overall_summary", ""),
            reasoning=data.get("reasoning", ""),
            claims=claims,
            issues=data.get("issues", []),
        )
        log.info(
            "Claims: verdict=%s claims=%d issues=%d",
            doc.claims_result.overall_verdict,
            len(claims),
            len(doc.claims_result.issues),
        )

    except Exception as exc:
        log.error("Claims: failed — %s", exc)
        doc.claims_result = ClaimsResult(
            overall_verdict="NA",
            overall_summary=f"評価中にエラーが発生しました: {exc}",
            reasoning="",
        )

    return doc
