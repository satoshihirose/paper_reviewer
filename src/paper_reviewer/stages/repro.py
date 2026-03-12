"""Stage 4a: Reproducibility readiness check (vision-based)."""

from __future__ import annotations

import logging
import re

import fitz  # pymupdf
import ollama

from paper_reviewer.apis import check_github_repo_files, check_url_accessible
from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import PaperDocument, ReproScore

log = logging.getLogger(__name__)

_KEYS = ["code_url", "data_url", "compute", "hyperparams", "seed", "method_overview"]

_PROMPT = (
    "This is a page from a research paper. "
    "Extract the following 6 fields about THIS paper's OWN experiments (not related work or baselines). "
    "These details are often found in footnotes, appendices, or implementation details sections — check carefully. "
    "Quote sentences verbatim for non-URL fields. Write null if not present on this page.\n\n"
    "code_url: <actual URL to this paper's code repository, or null."
    " e.g. 'https://github.com/QwenLM/Qwen3'>\n"
    "data_url: <actual URL to this paper's dataset, or null."
    " e.g. 'https://huggingface.co/datasets/math-ai/aime24'>\n"
    "compute: <verbatim sentence(s) naming a specific GPU/TPU model, training duration, or LLM model name/version/parameters used in THIS paper's experiments, or null."
    " e.g. 'All experiments are conducted using a single NVIDIA H200 GPU."
    " Training for 4 epochs on the 42.4B-token dataset corresponds to 95k optimization steps."
    " We evaluate using GPT-4o-2024-08-06 and Claude-3.5-Sonnet with temperature=0.'>\n"
    "hyperparams: <verbatim sentence(s) listing concrete numeric hyperparameter values, or null."
    " e.g. 'We linearly warm up the learning rate for 5,000 steps and then keep it at 1e-3."
    " We apply weight decay of 0.01 and dropout of 0.1. The batch size is 4,096.'>\n"
    "seed: <verbatim sentence stating a specific seed number or a concrete run count, or null."
    " e.g. 'For all tasks, we run training with three random seeds and report"
    " the mean and standard deviation of test-set performance.'>\n"
    "method_overview: <1–2 verbatim sentence(s) summarizing the core experimental approach"
    " (task type, model, training paradigm), or null."
    " e.g. 'We fine-tune LLaMA-3-8B on the SuperGLUE benchmark using supervised learning"
    " with cross-entropy loss, comparing full fine-tuning against LoRA adapters.'>\n\n"
    "Reply with only the 6 lines above, no extra text."
)


def _page_to_png(pdf_path: str, page_no: int, dpi: int = 150) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def _val(v: object) -> str | None:
    """null・空文字・テンプレートエコーは None として扱う。"""
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("null", "none", "n/a", ""):
        return None
    # LLM がプロンプトのテンプレートをそのままエコーした場合（例: "<value or null>"）
    if s.startswith("<") and s.endswith(">"):
        return None
    return s


_URL_KEYS = {"code_url", "data_url"}


def _is_url(s: str) -> bool:
    """簡易 URL 判定: ドットを含む文字列か http(s):// で始まるもの。"""
    return "." in s or s.startswith("http")


def _extract_from_page(image: bytes, model: str) -> dict[str, str | None]:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": _PROMPT, "images": [image]}],
        think=False,
    )
    raw = response["message"]["content"]
    result: dict[str, str | None] = {}
    for line in raw.splitlines():
        for key in _KEYS:
            if line.lower().startswith(key + ":"):
                value = _val(line[len(key) + 1:].strip())
                # URL フィールドは URL らしい値のみ採用
                if value and key in _URL_KEYS and not _is_url(value):
                    value = None
                result[key] = value
                break
    return result


# AllValues: key -> [(page_no, text), ...]  複数ページの検出を蓄積
_AllValues = dict[str, list[tuple[int, str]]]


def _build_excerpt(entries: list[tuple[int, str]]) -> str | None:
    """複数ページの検出結果を改行結合した excerpt 文字列を生成する。"""
    if not entries:
        return None
    return "\n".join(f"（p.{page}） {text}" for page, text in entries)


def _first_page(entries: list[tuple[int, str]]) -> int | None:
    return entries[0][0] if entries else None


def _build_partial_repro(all_values: _AllValues) -> ReproScore:
    """Vision 抽出の途中結果から部分的な ReproScore を生成する（ネットワーク確認なし）。"""
    code_entries = all_values.get("code_url", [])
    data_entries = all_values.get("data_url", [])
    return ReproScore(
        total=0,  # 最終スコアは run() 完了後に確定
        code_url=code_entries[0][1] if code_entries else None,
        code_url_page=_first_page(code_entries),
        data_url=data_entries[0][1] if data_entries else None,
        data_url_page=_first_page(data_entries),
        data_url_present=bool(data_entries),
        compute_mentioned=bool(all_values.get("compute")),
        compute_excerpt=_build_excerpt(all_values.get("compute", [])),
        compute_page=_first_page(all_values.get("compute", [])),
        method_overview=_build_excerpt(all_values.get("method_overview", [])),
        method_overview_page=_first_page(all_values.get("method_overview", [])),
        method_hyperparams_mentioned=bool(all_values.get("hyperparams")),
        method_hyperparams_excerpt=_build_excerpt(all_values.get("hyperparams", [])),
        method_hyperparams_page=_first_page(all_values.get("hyperparams", [])),
        method_seed_mentioned=bool(all_values.get("seed")),
        method_seed_excerpt=_build_excerpt(all_values.get("seed", [])),
        method_seed_page=_first_page(all_values.get("seed", [])),
    )


def _extract_vision(
    pdf_path: str,
    model: str,
    on_page: callable = None,
    cancel_event=None,
) -> _AllValues:
    """全ページをVision LLMで処理し、6項目の全検出結果（複数ページ）を返す。

    各キーに対して検出された (page_no, text) のリストを蓄積する。
    on_page(page_no: int, total_pages: int, partial_score: ReproScore) が指定された場合、
    各ページ処理後にコールバックを呼ぶ。
    全項目に少なくとも1件の検出がそろった時点で早期終了。
    cancel_event が set された場合、次ページ開始前に中断する。
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    all_values: _AllValues = {k: [] for k in _KEYS}

    for page_no in range(total_pages):
        if cancel_event and cancel_event.is_set():
            log.info("[repro/vision] Cancelled before page %d", page_no + 1)
            break
        image = _page_to_png(pdf_path, page_no)
        result = _extract_from_page(image, model)
        log.info("[repro/vision] Page %d/%d: found=%s", page_no + 1, total_pages,
                 [k for k, v in result.items() if v])

        for key in _KEYS:
            if result.get(key):
                all_values[key].append((page_no + 1, result[key]))

        if on_page:
            on_page(page_no + 1, total_pages, _build_partial_repro(all_values))

        if all(len(v) > 0 for v in all_values.values()):
            log.info("[repro/vision] All items found at page %d, stopping", page_no + 1)
            break

    return all_values


def _find_github_urls(text: str) -> list[str]:
    return re.findall(r"https?://github\.com/[^\s\)\]\>\"']+", text)


def run(doc: PaperDocument, llm: OllamaClient, on_page: callable = None, cancel_event=None) -> PaperDocument:
    """Run Stage 4a: check reproducibility readiness (vision-based).

    on_page(page_no: int, total_pages: int, partial_score: ReproScore) が指定された場合、
    各ページ処理後にコールバックを呼ぶ。
    cancel_event が set された場合、次ページ開始前に中断する。
    """
    full_text = doc.markdown_text
    lower = full_text.lower()
    score = 0
    details: list[str] = []

    # ── Vision-based extraction ────────────────────────────────────────────────
    all_values: _AllValues = {k: [] for k in _KEYS}

    if doc.pdf_path:
        log.info("[repro/vision] Starting vision extraction: %s", doc.pdf_path)
        all_values = _extract_vision(doc.pdf_path, llm.model, on_page=on_page, cancel_event=cancel_event)
        log.info("[repro/vision] Done: %s", {k: len(v) for k, v in all_values.items()})
    else:
        log.warning("[repro/vision] No pdf_path, falling back to text-based detection")

    # ── Code ──────────────────────────────────────────────────────────────────
    # Vision で取得できなければ正規表現でフォールバック
    code_entries = all_values.get("code_url", [])
    code_url: str | None = code_entries[0][1] if code_entries else None
    code_url_page: int | None = _first_page(code_entries)
    if not code_url:
        github_urls = _find_github_urls(full_text)
        code_url = github_urls[0] if github_urls else None

    code_accessible: bool | None = None
    code_has_env_file: bool | None = None

    if code_url:
        score += 2
        details.append(f"✅ [Code] GitHub URL found: {code_url}")
        code_accessible = check_url_accessible(code_url)
        if code_accessible:
            score += 1
            details.append("✅ [Code] Repository is publicly accessible")
            env_files = check_github_repo_files(code_url)
            found_env = [f for f, present in env_files.items() if present]
            if found_env:
                code_has_env_file = True
                score += 1
                details.append(f"✅ [Code] Environment file(s) found: {', '.join(found_env)}")
            else:
                code_has_env_file = False
                details.append("⚠️  [Code] No requirements.txt / pyproject.toml / environment.yml found")
        else:
            details.append("❌ [Code] Repository is not publicly accessible")
    else:
        details.append("❌ [Code] No GitHub URL found in paper")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_entries = all_values.get("data_url", [])
    data_url: str | None = data_entries[0][1] if data_entries else None
    data_url_page: int | None = _first_page(data_entries)
    if not data_url:
        # 正規表現フォールバック
        _data_url_m = re.search(
            r"https?://(?:huggingface\.co|zenodo\.org|figshare\.com|github\.com|drive\.google|dropbox|osf\.io)[^\s\)\]\"']+",
            full_text,
        )
        data_url = _data_url_m.group(0) if _data_url_m else None
        if not data_url and re.search(r"\bhuggingface\.co/datasets\b", lower):
            data_url = "huggingface.co/datasets (本文参照)"

    data_url_present = bool(data_url)
    data_only_on_request = bool(
        re.search(r"available\s+(?:upon|on)\s+request", lower)
        or re.search(r"upon\s+reasonable\s+request", lower)
    )

    if data_url_present:
        score += 1
        details.append("✅ [Data] Dataset URL found")
    elif data_only_on_request:
        details.append("⚠️  [Data] Data only 'available upon request' — weak reproducibility")
    else:
        details.append("❌ [Data] No dataset URL found")

    if data_only_on_request:
        details.append("⚠️  [Data] 'Available upon request' detected")

    # ── Compute ───────────────────────────────────────────────────────────────
    compute_entries = all_values.get("compute", [])
    compute_excerpt = _build_excerpt(compute_entries)
    compute_page = _first_page(compute_entries)
    compute_mentioned = bool(compute_entries)
    if compute_mentioned:
        score += 1
        details.append("✅ [Compute] Compute environment/time described")
    else:
        details.append("❌ [Compute] No compute environment/time description found")

    # ── Method ────────────────────────────────────────────────────────────────
    overview_entries = all_values.get("method_overview", [])
    method_overview = _build_excerpt(overview_entries)
    method_overview_page = _first_page(overview_entries)

    hyperparams_entries = all_values.get("hyperparams", [])
    method_hyperparams_excerpt = _build_excerpt(hyperparams_entries)
    method_hyperparams_page = _first_page(hyperparams_entries)
    method_hyperparams_mentioned = bool(hyperparams_entries)
    if method_hyperparams_mentioned:
        score += 1
        details.append("✅ [Method] Hyperparameters described")
    else:
        details.append("❌ [Method] No hyperparameter description found")

    seed_entries = all_values.get("seed", [])
    method_seed_excerpt = _build_excerpt(seed_entries)
    method_seed_page = _first_page(seed_entries)
    method_seed_mentioned = bool(seed_entries)
    if method_seed_mentioned:
        score += 1
        details.append("✅ [Method] Random seed/runs described")
    else:
        details.append("❌ [Method] No random seed description found")

    repro = ReproScore(
        total=min(score, 10),
        code_url=code_url,
        code_url_page=code_url_page,
        code_accessible=code_accessible,
        code_has_env_file=code_has_env_file,
        data_url_present=data_url_present,
        data_url=data_url,
        data_url_page=data_url_page,
        data_only_on_request=data_only_on_request,
        compute_mentioned=compute_mentioned,
        compute_excerpt=compute_excerpt,
        compute_page=compute_page,
        method_overview=method_overview,
        method_overview_page=method_overview_page,
        method_hyperparams_mentioned=method_hyperparams_mentioned,
        method_hyperparams_excerpt=method_hyperparams_excerpt,
        method_hyperparams_page=method_hyperparams_page,
        method_seed_mentioned=method_seed_mentioned,
        method_seed_excerpt=method_seed_excerpt,
        method_seed_page=method_seed_page,
        details=details,
    )

    doc.repro_score = repro
    return doc
