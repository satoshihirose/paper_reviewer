"""Stage 1: PDF parsing, section splitting, claim/reference extraction."""

from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

import pymupdf4llm

from paper_reviewer.apis import download_arxiv_pdf, parse_arxiv_id_from_input
from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import Claim, PaperDocument, Reference, StructureFlags

# Headers that typically begin each section.
# Matches both markdown headings (## References) and bold headings (**References**).
_SEC_NAMES = [
    r"(abstract)",
    r"(introduction)",
    r"(related\s+work)",
    r"(background)",
    r"(method(?:ology)?s?)",
    r"(experiment(?:al\s+setup)?s?)",
    r"(results?)",
    r"(discussion)",
    r"(conclusion)",
    r"(limitations?)",
    r"(broader\s+impacts?|ethics\s+statement)",
    r"(references?|bibliography)",
    r"(appendix|supplementary)",
]
SECTION_PATTERNS = (
    [rf"^#{{1,3}}\s*{n}" for n in _SEC_NAMES]
    + [rf"^\*\*{n}\*\*\s*$" for n in _SEC_NAMES]
    + [rf"^{n}\s*$" for n in _SEC_NAMES]
)


def split_sections(markdown: str) -> dict[str, str]:
    """Split markdown text into named sections."""
    sections: dict[str, str] = {}
    current_name = "preamble"
    current_lines: list[str] = []

    for line in markdown.splitlines():
        matched = False
        for pat in SECTION_PATTERNS:
            m = re.match(pat, line, re.IGNORECASE)
            if m:
                if current_lines:
                    sections[current_name] = "\n".join(current_lines).strip()
                current_name = m.group(1).lower().replace(" ", "_")
                current_lines = [line]
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections[current_name] = "\n".join(current_lines).strip()

    return sections


def extract_title(markdown: str) -> str:
    """Heuristically extract title from the top of the markdown."""
    for line in markdown.splitlines()[:30]:
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
        if len(line) > 20 and not line.startswith("-") and not re.match(r"^\d", line):
            return line
    return "Unknown Title"


def extract_claims_with_llm(sections: dict[str, str], llm: OllamaClient) -> list[Claim]:
    """Use Ollama to extract claims from abstract + introduction."""
    text = ""
    for sec in ["abstract", "introduction"]:
        if sec in sections:
            text += f"\n\n## {sec.upper()}\n{sections[sec]}"
    if not text:
        text = list(sections.values())[0] if sections else ""

    prompt = f"""Extract the main claims made in this research paper section.
For each claim, provide:
- id: sequential identifier like "C001", "C002", ...
- text: the claim as stated or paraphrased
- type: one of QUANTITATIVE / COMPARATIVE / METHODOLOGICAL / CONCEPTUAL
- section: which section this appears in (abstract / introduction)
- cited_refs: list of citation markers like "[1]", "[Smith, 2023]" if any

Return a JSON array of claim objects.

TEXT:
{text[:6000]}

JSON:"""

    try:
        data = llm.complete_json(prompt)
        claims = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                claims.append(
                    Claim(
                        id=item.get("id", f"C{i+1:03d}"),
                        text=item.get("text", ""),
                        type=item.get("type", "CONCEPTUAL"),
                        section=item.get("section", "unknown"),
                        cited_refs=item.get("cited_refs", []),
                    )
                )
        return claims
    except Exception:
        return []



_FORMAT_DETECT_PROMPT = """Analyze this reference list sample and output Python regex patterns to parse it.

SAMPLE:
{sample}

Identify the reference style:
- Style A (numbered): "[1] Author, A. Title. Journal, Year." or "1 Author, A. Title. Journal, Year."
- Style B (author-year): "Author, A., & Author, B. (Year). Title. Journal."

Output ONLY a plain text block in exactly this format (no JSON, no quotes, no escaping):

split: <regex using positive lookahead (?=...) to split entries, used with re.MULTILINE>
id: <regex with one capture group for citation marker, or NONE if Style B>
authors: <regex with one capture group for author string>
year: <regex with one capture group for 4-digit year>
title: <regex with one capture group for paper title>
doi: <regex with one capture group for DOI, or NONE if not present in sample>
url: <regex with one capture group for URL, or NONE if not present in sample>

Rules:
- Write raw Python regex characters directly. Do NOT escape backslashes.
- split MUST use positive lookahead (?=...), never lookbehind (?<=...).
- Style A split example:  split: (?=^\[\d+\])
- Style B split example:  split: (?=^[A-Z][a-z]+,\s)
- Style B year example:   year: \((\d{{4}})\)
- Style B title example:  title: \)\.\s+(.+?)(?=\.\s+[A-Z]|\n|$)
- doi example:            doi: (10\.\d{{4,}}/\S+)
- url example:            url: (https?://\S+)
- Each pattern (except split) must have exactly one capture group ( ).

OUTPUT:"""


def _parse_pattern_block(text: str) -> dict[str, str]:
    """Parse key: value lines from the LLM plain-text output."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key in ("split", "id", "authors", "year", "title", "doi", "url") and value:
            result[key] = value
    return result


def _detect_format_and_extract(
    ref_text: str, llm: OllamaClient
) -> list[Reference] | None:
    """Use LLM to detect reference format, then apply regex to extract all entries.

    Returns None if format detection fails.
    """
    sample = ref_text[:800]
    try:
        raw = llm.complete(
            _FORMAT_DETECT_PROMPT.format(sample=sample),
            system="Output only the plain text block as instructed. No JSON. No extra explanation.",
        )
    except Exception as e:
        log.warning("Parser/format_detect: LLM failed — %s", e)
        return None

    patterns = _parse_pattern_block(raw)
    log.info("Parser/format_detect: raw patterns=%s", patterns)

    # Validate split pattern
    split_pat_str = patterns.get("split", "")
    try:
        split_pat = re.compile(split_pat_str, re.MULTILINE)
    except re.error as e:
        log.warning("Parser/format_detect: invalid split_pattern %r — %s", split_pat_str, e)
        return None

    # Validate field patterns (skip invalid ones, don't abort)
    field_pats: dict[str, re.Pattern] = {}
    for field in ("id", "authors", "year", "title", "doi", "url"):
        pat_str = patterns.get(field, "")
        if not pat_str or pat_str.upper() == "NONE":
            continue
        try:
            field_pats[field] = re.compile(pat_str, re.DOTALL | re.IGNORECASE)
        except re.error as e:
            log.warning("Parser/format_detect: invalid %s pattern %r — skipping", field, pat_str)

    # Split into entries
    entries = [e.strip() for e in re.split(split_pat, ref_text.strip()) if e.strip() and len(e.strip()) > 10]
    if len(entries) < 2:
        log.warning("Parser/format_detect: split yielded only %d entries with pattern %r, falling back", len(entries), split_pat_str)
        return None
    if len(entries) > 300:
        log.warning("Parser/format_detect: split yielded %d entries (too many) with pattern %r, falling back", len(entries), split_pat_str)
        return None

    log.info("Parser/format_detect: %d entries, split=%r, fields=%s", len(entries), split_pat_str, {k: p.pattern for k, p in field_pats.items()})

    # Extract fields from each entry via regex
    refs: list[Reference] = []
    for i, entry in enumerate(entries):
        fields: dict[str, str | None] = {}
        for field, pat in field_pats.items():
            m = pat.search(entry)
            fields[field] = m.group(1).strip() if m else None

        refs.append(Reference(
            id=fields.get("id") or f"[{i + 1}]",
            raw_text=entry,
            title=fields.get("title"),
            authors=fields.get("authors"),
            year=fields.get("year"),
            doi=fields.get("doi"),
            url=fields.get("url"),
        ))

    return refs


def extract_references_with_llm(
    sections: dict[str, str], llm: OllamaClient
) -> list[Reference]:
    """Extract references: try format-detection (1 LLM call) first, fall back to chunk-based LLM."""
    ref_text = sections.get("references", sections.get("bibliography", ""))
    if not ref_text:
        return []

    ref_text = re.sub(r"\n{3,}", "\n\n", ref_text)

    t = time.perf_counter()
    refs = _detect_format_and_extract(ref_text, llm)
    if refs is not None:
        log.info("Parser/refs: %d refs via format-detection in %.1fs", len(refs), time.perf_counter() - t)
        return refs

    log.warning("Parser/refs: format detection failed, returning empty list")
    return []


def check_structure_flags(markdown: str, sections: dict[str, str], llm: OllamaClient) -> StructureFlags:
    """Check for required structural sections and compliance flags."""
    flags = StructureFlags()
    lower = markdown.lower()

    # Limitations
    flags.has_limitations = bool(
        re.search(r"\blimitation", lower) or "limitations" in sections
    )
    if flags.has_limitations:
        lim_text = sections.get("limitations", "")
        flags.limitations_nontrivial = bool(
            lim_text and len(lim_text.strip()) > 50
            and "n/a" not in lim_text.lower()
        )

    # Broader impacts / ethics
    flags.has_broader_impacts = bool(
        re.search(r"\bbroader\s+impact|\bethics\s+statement|\bsocietal\s+impact", lower)
        or "broader_impacts" in sections
    )

    # LLM usage disclosure
    llm_used = bool(
        re.search(
            r"\b(gpt-[34]|gpt-4o|claude|gemini|llama|qwen|mistral|chatgpt)\b.*\b(used|employ|utiliz|leverage)",
            lower,
        )
        or re.search(
            r"\b(used|employ|utiliz|leverage)\b.{0,60}\b(gpt-[34]|gpt-4o|claude|gemini|llama|qwen|mistral|chatgpt)\b",
            lower,
        )
    )
    disclosure_present = bool(
        re.search(r"\bllm\s+(?:usage|disclosure|use statement)\b", lower)
        or re.search(r"\bgenerated\s+(?:by|with|using)\b.{0,40}\b(gpt|claude|gemini)\b", lower)
    )
    flags.has_llm_disclosure = disclosure_present
    flags.llm_used_without_disclosure = llm_used and not disclosure_present

    # Dataset license
    flags.has_dataset_license = bool(
        re.search(r"\b(license|cc-by|mit license|apache|creative\s+commons)\b", lower)
    )

    # IRB / ethics approval
    flags.has_irb_mention = bool(
        re.search(r"\b(irb|institutional\s+review\s+board|ethics\s+approv|informed\s+consent)\b", lower)
    )

    # Build missing list
    missing = []
    if not flags.has_limitations:
        missing.append("Limitations section missing")
    elif not flags.limitations_nontrivial:
        missing.append("Limitations section appears trivial/empty")
    if not flags.has_broader_impacts:
        missing.append("Broader Impacts / Ethics Statement missing")
    if flags.llm_used_without_disclosure:
        missing.append("LLM usage detected but no disclosure statement found")
    flags.missing = missing

    return flags


def load_pdf_as_markdown(pdf_path: Path) -> str:
    """Convert PDF to Markdown using pymupdf4llm."""
    return pymupdf4llm.to_markdown(str(pdf_path))


def run(
    input_str: str,
    llm: OllamaClient,
    pdf_cache_dir: Optional[Path] = None,
) -> PaperDocument:
    """Run Stage 1: parse input, extract text, split sections, extract claims/refs."""
    arxiv_id = parse_arxiv_id_from_input(input_str)
    pdf_path: Optional[Path] = None

    if arxiv_id:
        cache = pdf_cache_dir or Path(tempfile.gettempdir())
        pdf_path = download_arxiv_pdf(arxiv_id, dest_dir=cache)
    elif input_str.startswith("http"):
        # Generic URL — try to download
        import httpx

        cache = pdf_cache_dir or Path(tempfile.gettempdir())
        out = cache / "downloaded_paper.pdf"
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(input_str)
            resp.raise_for_status()
            out.write_bytes(resp.content)
        pdf_path = out
    else:
        pdf_path = Path(input_str)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    t = time.perf_counter()
    markdown = load_pdf_as_markdown(pdf_path)
    log.info("Parser/pdf2md: %.1fs (%d chars)", time.perf_counter() - t, len(markdown))

    sections = split_sections(markdown)
    log.info("Parser/sections: %s", list(sections.keys()))

    title = extract_title(markdown)

    # claims = extract_claims_with_llm(sections, llm)
    claims = []
    log.info("Parser/claims: skipped")

    t = time.perf_counter()
    structure_flags = check_structure_flags(markdown, sections, llm)
    log.info("Parser/structure_flags: %.1fs", time.perf_counter() - t)

    return PaperDocument(
        title=title,
        arxiv_id=arxiv_id,
        pdf_path=str(pdf_path),
        markdown_text=markdown,
        sections=sections,
        claims=claims,
        structure_flags=structure_flags,
    )
