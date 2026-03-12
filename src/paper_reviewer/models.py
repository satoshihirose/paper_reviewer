"""Data models for the paper review pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Claim:
    id: str                    # e.g. "C001"
    text: str
    type: str                  # QUANTITATIVE / COMPARATIVE / METHODOLOGICAL / CONCEPTUAL
    section: str
    cited_refs: list[str] = field(default_factory=list)


@dataclass
class Reference:
    id: str                    # e.g. "[23]"
    raw_text: str
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    existence_status: Optional[str] = None   # VERIFIED / HALLUCINATED / UNRESOLVABLE
    semantic_scholar_id: Optional[str] = None
    page_no: Optional[int] = None            # PDF page number (1-indexed)


@dataclass
class CitationResult:
    ref_id: str
    status: str                # VERIFIED / HALLUCINATED / UNRESOLVABLE
    matched_title: Optional[str] = None
    match_score: Optional[float] = None
    note: Optional[str] = None


@dataclass
class ConsistencyIssue:
    id: str                    # e.g. "S001"
    type: str                  # INTERNAL_INCONSISTENCY / UNSUPPORTED_CONTRIBUTION / STATISTICAL_REPORTING_WEAK
    description: str
    evidence_abstract: Optional[str] = None
    evidence_results: Optional[str] = None


@dataclass
class ReproScore:
    total: int                 # 0-10
    code_url: Optional[str] = None
    code_url_page: Optional[int] = None
    code_accessible: Optional[bool] = None
    code_has_env_file: Optional[bool] = None
    data_url_present: bool = False
    data_url: Optional[str] = None
    data_url_page: Optional[int] = None
    data_only_on_request: bool = False
    compute_mentioned: bool = False
    compute_excerpt: Optional[str] = None
    compute_page: Optional[int] = None
    method_overview: Optional[str] = None
    method_overview_page: Optional[int] = None
    method_hyperparams_mentioned: bool = False
    method_hyperparams_excerpt: Optional[str] = None
    method_hyperparams_page: Optional[int] = None
    method_seed_mentioned: bool = False
    method_seed_excerpt: Optional[str] = None
    method_seed_page: Optional[int] = None
    details: list[str] = field(default_factory=list)


@dataclass
class StatReportScore:
    has_error_bars: bool = False      # エラーバー・標準偏差の報告
    has_multiple_runs: bool = False   # 複数回試行の記述
    has_data_splits: bool = False     # Train/Val/Test 分割の明示


@dataclass
class StructureFlags:
    has_limitations: bool = False
    limitations_nontrivial: bool = False
    has_broader_impacts: bool = False
    has_llm_disclosure: bool = False
    llm_used_without_disclosure: bool = False
    has_dataset_license: bool = False
    has_irb_mention: bool = False
    missing: list[str] = field(default_factory=list)


@dataclass
class ChecklistFinding:
    text: str                  # 確認項目の説明
    status: str                # ok / warn / ng / none


@dataclass
class ChecklistResult:
    """汎用チェックリスト評価結果（items 4–16 で共用）。"""
    overall_verdict: str       # Yes / Partial / No / NA
    overall_summary: str
    reasoning: str
    findings: list[ChecklistFinding] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class LimitationsResult:
    overall_verdict: str       # Yes / Partial / No / NA
    overall_summary: str
    reasoning: str
    findings: list[ChecklistFinding] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class TheoryResult:
    overall_verdict: str       # Yes / Partial / No / NA
    overall_summary: str
    reasoning: str
    findings: list[ChecklistFinding] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class ClaimVerdict:
    text: str                  # 抽出されたクレームのテキスト
    verdict: str               # SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / UNCLEAR
    evidence: str              # 根拠（Figure/Table/Section番号など）


@dataclass
class ClaimsResult:
    overall_verdict: str       # Yes / Partial / No / NA
    overall_summary: str       # 2-3文の人間向けサマリー
    reasoning: str             # ステップごとの判断根拠
    claims: list[ClaimVerdict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class PaperDocument:
    title: str
    arxiv_id: Optional[str] = None
    pdf_path: Optional[str] = None
    markdown_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    citation_results: Optional[list[CitationResult]] = None
    consistency_issues: Optional[list[ConsistencyIssue]] = None
    stat_report: Optional[StatReportScore] = None
    repro_score: Optional[ReproScore] = None
    claims_result: Optional[ClaimsResult] = None
    limitations_result: Optional[LimitationsResult] = None
    theory_result: Optional[TheoryResult] = None
    checklist_results: dict = field(default_factory=dict)  # key: "4"–"16" → ChecklistResult
    repro_current_page: Optional[int] = None   # set during vision processing, None when done
    repro_total_pages: Optional[int] = None    # set during vision processing, None when done
    structure_flags: Optional[StructureFlags] = None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
