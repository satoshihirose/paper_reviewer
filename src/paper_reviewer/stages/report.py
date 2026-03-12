"""Stage 5: Report generation (terminal via Rich + JSON output)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from paper_reviewer.models import PaperDocument

console = Console()


def _score_bar(value: int, max_val: int, width: int = 20) -> str:
    filled = int(width * value / max_val) if max_val else 0
    return "█" * filled + "░" * (width - filled)


def _citation_score(doc: PaperDocument) -> tuple[int, int]:
    """Return (verified_count, total_count)."""
    total = len(doc.citation_results)
    verified = sum(1 for r in doc.citation_results if r.status == "VERIFIED")
    return verified, total


def print_report(doc: PaperDocument) -> None:
    """Print a rich-formatted report to the terminal."""

    # --- Header ---
    console.print()
    console.print(
        Panel(
            f"[bold cyan]{doc.title}[/bold cyan]\n"
            + (f"[dim]arXiv: {doc.arxiv_id}[/dim]" if doc.arxiv_id else ""),
            title="[bold]Paper Reviewer Report[/bold]",
            border_style="cyan",
        )
    )

    # --- Scorecard table ---
    table = Table(title="Scorecard", box=box.ROUNDED, show_header=True)
    table.add_column("Category", style="bold")
    table.add_column("Score")
    table.add_column("Bar")
    table.add_column("Notes")

    # Citation score
    verified, total_cit = _citation_score(doc)
    cit_score = int(100 * verified / total_cit) if total_cit else 0
    hallucinated = sum(1 for r in doc.citation_results if r.status == "HALLUCINATED")
    table.add_row(
        "Citation Existence",
        f"{cit_score}/100",
        _score_bar(cit_score, 100),
        f"{verified}/{total_cit} verified" + (f", {hallucinated} hallucinated" if hallucinated else ""),
    )

    # Consistency score
    issues = doc.consistency_issues or []
    internal = [i for i in issues if i.type == "INTERNAL_INCONSISTENCY"]
    stat_weak = [i for i in issues if i.type == "STATISTICAL_REPORTING_WEAK"]
    consistency_score = max(0, 100 - len(internal) * 20 - len(stat_weak) * 10)
    table.add_row(
        "Internal Consistency",
        f"{consistency_score}/100",
        _score_bar(consistency_score, 100),
        f"{len(internal)} inconsistencies, {len(stat_weak)} stat warnings",
    )

    # Structure compliance
    sf = doc.structure_flags
    if sf:
        flags_ok = sum(
            [sf.has_limitations, sf.limitations_nontrivial, sf.has_broader_impacts, sf.has_llm_disclosure or not sf.llm_used_without_disclosure]
        )
        struct_score = int(100 * flags_ok / 4)
        table.add_row(
            "Structure Compliance",
            f"{struct_score}/100",
            _score_bar(struct_score, 100),
            f"{len(sf.missing)} issues" if sf.missing else "OK",
        )
    else:
        table.add_row("Structure Compliance", "N/A", "", "")

    # Reproducibility
    rs = doc.repro_score
    if rs:
        table.add_row(
            "Reproducibility Readiness",
            f"{rs.total}/10",
            _score_bar(rs.total, 10),
            "",
        )
    else:
        table.add_row("Reproducibility Readiness", "N/A", "", "")

    console.print(table)
    console.print()

    # --- Structure flags ---
    if sf and sf.missing:
        console.print("[bold yellow]Structure Issues[/bold yellow]")
        for item in sf.missing:
            console.print(f"  [yellow]⚠[/yellow]  {item}")
        console.print()

    # --- High risk citations ---
    bad_cits = [r for r in doc.citation_results if r.status == "HALLUCINATED"]
    if bad_cits:
        console.print("[bold red]Potentially Hallucinated Citations[/bold red]")
        for r in bad_cits[:10]:
            console.print(f"  [red]✗[/red]  {r.ref_id}: not found in Semantic Scholar")
        console.print()

    # --- Consistency issues ---
    if issues:
        console.print("[bold red]Consistency / Statistical Issues[/bold red]")
        for issue in issues[:10]:
            color = "red" if issue.type == "INTERNAL_INCONSISTENCY" else "yellow"
            console.print(f"  [{color}]•[/{color}]  [{issue.id}] {issue.description}")
            if issue.evidence_abstract:
                console.print(f"       Abstract: [dim]{issue.evidence_abstract[:120]}[/dim]")
            if issue.evidence_results:
                console.print(f"       Results:  [dim]{issue.evidence_results[:120]}[/dim]")
        console.print()

    # --- Reproducibility details ---
    if rs:
        console.print("[bold]Reproducibility Details[/bold]")
        for line in rs.details:
            console.print(f"  {line}")
        console.print()

    # --- Claims summary ---
    if doc.claims:
        console.print(f"[bold]Claims Extracted[/bold]: {len(doc.claims)}")
        for c in doc.claims[:5]:
            console.print(f"  [{c.id}] ({c.type}) {c.text[:100]}")
        if len(doc.claims) > 5:
            console.print(f"  ... and {len(doc.claims) - 5} more")
        console.print()


def save_json(doc: PaperDocument, output_path: Path) -> None:
    """Save the full report as JSON."""
    data = doc.to_dict()
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    console.print(f"[green]Report saved to:[/green] {output_path}")
