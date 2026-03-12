"""Stage: NeurIPS Checklist items 4–16 (vision-based, all pages at once).

All items share the same JSON output schema:
  { overall_verdict, overall_summary, reasoning, findings[], issues[] }
"""

from __future__ import annotations

import logging

from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import ChecklistFinding, ChecklistResult, PaperDocument
from paper_reviewer.stages._json_utils import pdf_to_images, vision_chat

log = logging.getLogger(__name__)

_PREAMBLE = (
    "You are an expert machine learning researcher conducting a professional, neutral, and fair peer review. "
    "Evaluate the following NeurIPS Paper Checklist criterion with balanced judgment — neither overly lenient nor overly harsh. "
    "This checklist is designed to encourage best practices for responsible machine learning research, "
    "addressing issues of reproducibility, transparency, research ethics, and societal impact.\n\n"
    "The images above are ALL pages of the paper, in order.\n\n"
)

_JSON_SCHEMA = """
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
  "issues": ["<specific gaps or problems found>"]
}
"""

# (title, criterion_text_for_prompt)
_ITEMS: dict[str, tuple[str, str]] = {
    "4": (
        "4. Experimental Result Reproducibility",
        """**Experimental Result Reproducibility**: While NeurIPS does not require releasing code, \
all submissions must provide some reasonable avenue for reproducibility.
- For new algorithms: the paper should make it clear how to reproduce that algorithm.
- For new architectures: the paper should describe the architecture fully.
- For new models: there should either be a way to access the model for reproducing results or a way to reproduce the model.
- For closed-source models: other researchers should have some path to access or reproduce the work.

NA means the paper makes no experimental contributions (purely theoretical).

Evaluate:
1. Is the proposed algorithm/method described in enough detail to reimplement?
2. Is the architecture or model fully described?
3. Is there a reproducible path (code URL, model access, detailed description)?
4. For closed-source resources, is there at least a partial access path?

Verdict: Yes / Partial / No / NA""",
    ),
    "5": (
        "5. Open Access to Data and Code",
        """**Open Access to Data and Code**: Did the authors include the code, data, and instructions \
needed to reproduce the main experimental results (either in supplemental material or as a URL)?
- Instructions should specify the exact commands and environment needed to reproduce results.
- Main experimental results include the new method AND baselines.
- "No because code/data is proprietary" is an acceptable reason.
- At submission time, anonymized versions should be released.

NA means the paper does not include experiments.

Evaluate:
1. Is code available (URL or supplemental)?
2. Is data available or is a clear access path described?
3. Are reproduction instructions (commands, environment) provided?
4. Are baseline experiment scripts/details included?

Verdict: Yes / Partial / No / NA""",
    ),
    "6": (
        "6. Experimental Setting/Details",
        """**Experimental Setting/Details**: Did the authors specify all training details \
(e.g., data splits, hyperparameters, how they were chosen)?
- The full details can be in code, appendix, or supplemental, but important details must be in the main paper.
- The experimental setting should be presented at a level of detail necessary to appreciate and make sense of the results.

NA means the paper does not include experiments.

Evaluate:
1. Are dataset splits (train/validation/test) specified?
2. Are all hyperparameters reported (learning rate, batch size, epochs, optimizer, etc.)?
3. Is it described how hyperparameters were selected (grid search, prior work, etc.)?
4. Is sufficient detail provided in the main paper (not just appendix)?

Verdict: Yes / Partial / No / NA""",
    ),
    "7": (
        "7. Experiment Statistical Significance",
        """**Experiment Statistical Significance**: Does the paper report error bars suitably and \
correctly defined, or other appropriate statistical significance information?
- Results supporting main claims should be accompanied by error bars, confidence intervals, or significance tests.
- The factors of variability captured by error bars should be clearly stated.
- The method for calculating error bars should be explained.
- It should be clear whether the error bar is standard deviation or standard error of the mean.
- For asymmetric distributions, symmetric error bars in tables/plots should be avoided.

NA means the paper does not include experiments.

Evaluate:
1. Are error bars or confidence intervals reported for main results?
2. Is it clear what variability the error bars capture (e.g., random seed, data split)?
3. Is the calculation method explained (std, SEM, CI level)?
4. Are asymmetric distributions handled appropriately?

Verdict: Yes / Partial / No / NA""",
    ),
    "8": (
        "8. Experiments Compute Resources",
        """**Experiments Compute Resources**: For each experiment, does the paper provide sufficient \
information on compute resources (type of compute, memory, execution time) needed to reproduce?
- Should indicate the type of compute workers: CPU or GPU, internal cluster, or cloud provider, including memory and storage.
- Should provide the amount of compute required for each individual experimental run AND an estimate of total compute.
- Should disclose whether the full research project required more compute than the reported experiments.

NA means the paper does not include experiments.

Evaluate:
1. Is the type of hardware specified (GPU model, CPU, cloud provider)?
2. Is memory/storage information provided?
3. Is per-experiment runtime or compute cost reported?
4. Is total project compute disclosed?

Verdict: Yes / Partial / No / NA""",
    ),
    "9": (
        "9. Code of Ethics",
        """**Code of Ethics**: Have the authors ensured that their research conforms to the NeurIPS Code of Ethics?
- The NeurIPS Code of Ethics covers: data privacy, fairness, societal harm, dual-use risks, research integrity.
- Special circumstances requiring deviation should be explained.
- This item is always applicable — NA is not valid.

Evaluate whether the research shows signs of ethical concerns, including:
1. Privacy violations or unauthorized data collection
2. Potential for direct harm, discrimination, or unfair impact on groups
3. Dual-use risks (research that could be weaponized or misused)
4. Deceptive or manipulative content generation
5. Any indication the authors have considered and addressed ethical implications

Verdict: Yes (no ethical concerns or concerns are addressed) / Partial (some concerns present but partially addressed) / No (clear ethical issues not addressed)""",
    ),
    "10": (
        "10. Broader Impacts",
        """**Broader Impacts**: If appropriate for the scope and focus of the paper, did the authors \
discuss potential negative societal impacts of their work?
- Examples: disinformation, fake profiles, surveillance, fairness issues, privacy concerns, security concerns.
- Authors should point out if there is a direct path to negative applications.
- Consider: harms when technology works as intended, harms when it gives incorrect results, harms from misuse.
- Mitigation strategies (e.g., gated release, defenses, monitoring) should be discussed if applicable.
- Foundational research with no direct negative application path may answer NA.

Evaluate:
1. Does the paper discuss potential negative societal impacts?
2. Are fairness, privacy, or security implications considered?
3. Is there discussion of misuse potential?
4. Are mitigation strategies proposed?

Verdict: Yes / Partial / No / NA""",
    ),
    "11": (
        "11. Safeguards",
        """**Safeguards**: Do the authors have safeguards in place for responsible release of models \
with high risk for misuse (e.g., pretrained language models, generative models)?
- Released models with high misuse/dual-use risk should include safeguards (usage guidelines, access restrictions).
- Datasets scraped from the Internet should describe how unsafe content was avoided.
- NA if the paper does not release models/datasets with high misuse risk.

Evaluate:
1. Does the paper release a model or dataset with potential for misuse?
2. If yes, are safeguards described (access controls, usage policies, content filters)?
3. For scraped datasets, is safety filtering described?

Verdict: Yes / Partial / No / NA""",
    ),
    "12": (
        "12. Licenses for Existing Assets",
        """**Licenses for Existing Assets**: If the paper uses existing assets (code, data, models), \
did the authors cite the creators and respect the license and terms of use?
- Original paper/source for each asset should be cited.
- The version of the asset should be stated.
- A URL should be included if possible.
- The name of the license (e.g., CC-BY 4.0, MIT, Apache 2.0) should be stated for each asset.
- For scraped data, copyright and terms of service should be stated.
- NA if the paper does not use existing external assets.

Evaluate:
1. Are all used datasets, models, and code packages cited with sources?
2. Are licenses named for each external asset?
3. Are versions and URLs provided?
4. For scraped data, are terms of service considerations addressed?

Verdict: Yes / Partial / No / NA""",
    ),
    "13": (
        "13. New Assets",
        """**New Assets**: If the paper releases new assets (datasets, code, models), are they \
documented with necessary details?
- Details about training, license, limitations, etc. should be communicated.
- The paper should discuss whether and how consent was obtained from people whose data is included.
- At submission time, assets should be anonymized if applicable.
- NA if the paper does not release new assets.

Evaluate:
1. Are new assets (code/data/models) released or planned for release?
2. Is documentation provided (license, intended use, limitations)?
3. Is consent/data collection methodology described for datasets?
4. Are training details for new models documented?

Verdict: Yes / Partial / No / NA""",
    ),
    "14": (
        "14. Crowdsourcing and Research with Human Subjects",
        """**Crowdsourcing and Research with Human Subjects**: If the authors used crowdsourcing or \
conducted research with human subjects, did they include full instructions, screenshots, and compensation details?
- Full text of instructions given to participants should be included (supplemental is acceptable).
- If human subjects are the main contribution, details should be in the main paper.
- Workers involved in data collection must be paid at least the minimum wage in their country.
- NA if the paper does not involve crowdsourcing or human subjects research.

Evaluate:
1. Were crowdsourcing platforms or human participants used?
2. Are full participant instructions provided?
3. Is compensation information disclosed (amount, whether it meets minimum wage)?
4. Are screenshots or interface details included?

Verdict: Yes / Partial / No / NA""",
    ),
    "15": (
        "15. Institutional Review Board (IRB) Approvals",
        """**Institutional Review Board (IRB) Approvals**: Did the authors describe potential \
participant risks and obtain IRB approval (or equivalent) if applicable?
- IRB approval (or equivalent) may be required for human subjects research depending on country/institution.
- If IRB approval was obtained, it should be clearly stated in the paper.
- Participant risks should be described.
- NA if the paper does not involve human subjects research.

Evaluate:
1. Does the paper involve human subjects research?
2. Is IRB approval (or equivalent) mentioned and described?
3. Are potential participant risks discussed?

Verdict: Yes / Partial / No / NA""",
    ),
    "16": (
        "16. Declaration of LLM Usage",
        """**Declaration of LLM Usage**: Does the paper describe the usage of LLMs if it is an \
important, original, or non-standard component of the core methods in this research?
- Declaration is NOT required if the LLM is used only for writing, editing, or formatting.
- Declaration IS required if LLMs are part of the core methodology, experimental pipeline, or evaluation.
- NA if the core method development does not involve LLMs as any important, original, or non-standard component.

Evaluate:
1. Are LLMs used as part of the core research methodology (not just writing assistance)?
2. If yes, is this usage clearly declared and described?
3. Are the specific LLM models, versions, and prompts documented where relevant?

Verdict: Yes / Partial / No / NA""",
    ),
}


def run_item(item_key: str, doc: PaperDocument, llm: OllamaClient) -> PaperDocument:
    """Run a single checklist item evaluation and store result in doc.checklist_results[item_key]."""
    title, criterion = _ITEMS[item_key]

    if not doc.pdf_path:
        log.warning("Checklist %s: no pdf_path, skipping", item_key)
        doc.checklist_results[item_key] = ChecklistResult(
            overall_verdict="NA",
            overall_summary="PDF not available for vision-based check.",
            reasoning="",
        )
        return doc

    log.info("Checklist %s (%s): sending %s to LLM", item_key, title, llm.model)

    # Reuse cached images if available (stored as _cached_images on doc to avoid repeated fitz.open)
    images: list[bytes] = getattr(doc, "_cached_images", None) or pdf_to_images(doc.pdf_path)

    prompt = _PREAMBLE + criterion + _JSON_SCHEMA

    try:
        data = vision_chat(llm.model, prompt, images)
        findings = [
            ChecklistFinding(text=f.get("text", ""), status=f.get("status", "none"))
            for f in data.get("findings", [])
        ]
        doc.checklist_results[item_key] = ChecklistResult(
            overall_verdict=data.get("overall_verdict", "NA"),
            overall_summary=data.get("overall_summary", ""),
            reasoning=data.get("reasoning", ""),
            findings=findings,
            issues=data.get("issues", []),
        )
        log.info(
            "Checklist %s: verdict=%s findings=%d issues=%d",
            item_key,
            doc.checklist_results[item_key].overall_verdict,
            len(findings),
            len(doc.checklist_results[item_key].issues),
        )

    except Exception as exc:
        log.error("Checklist %s: failed — %s", item_key, exc)
        doc.checklist_results[item_key] = ChecklistResult(
            overall_verdict="NA",
            overall_summary=f"評価中にエラーが発生しました: {exc}",
            reasoning="",
        )

    return doc


def item_keys() -> list[str]:
    return list(_ITEMS.keys())


def item_title(key: str) -> str:
    return _ITEMS[key][0]
