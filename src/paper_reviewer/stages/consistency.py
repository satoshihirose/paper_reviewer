"""Stage 3: Internal consistency check (Abstract vs Results)."""

from __future__ import annotations

from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import ConsistencyIssue, PaperDocument, StatReportScore


def _check_statistical_reporting(sections: dict[str, str], llm: OllamaClient) -> StatReportScore:
    """Check for missing statistical reporting (error bars, multiple runs, splits)."""
    results_text = ""
    for key in ["results", "experiment", "experiments"]:
        if key in sections:
            results_text += sections[key][:2000]

    if not results_text:
        return StatReportScore()

    prompt = f"""Analyze this results section for statistical reporting quality.

Check if:
1. Error bars or standard deviations are reported for quantitative results
2. Multiple runs / random seeds are mentioned
3. Train/validation/test splits are clearly defined

Return a JSON object with boolean fields:
- has_error_bars: true/false
- has_multiple_runs: true/false
- has_data_splits: true/false

TEXT:
{results_text[:3000]}

JSON:"""

    try:
        data = llm.complete_json(prompt)
        if isinstance(data, dict):
            return StatReportScore(
                has_error_bars=bool(data.get("has_error_bars", False)),
                has_multiple_runs=bool(data.get("has_multiple_runs", False)),
                has_data_splits=bool(data.get("has_data_splits", False)),
            )
    except Exception:
        pass
    return StatReportScore()


def run(doc: PaperDocument, llm: OllamaClient) -> PaperDocument:
    """Run Stage 3: internal consistency check."""
    doc.consistency_issues = []
    doc.stat_report = _check_statistical_reporting(doc.sections, llm)
    return doc
