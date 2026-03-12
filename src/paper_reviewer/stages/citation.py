"""Stage 2: Reference extraction (vision) + Citation verification via Semantic Scholar."""

from __future__ import annotations

import json
import logging
import re

import fitz  # pymupdf
import ollama

from paper_reviewer.apis import search_semantic_scholar, title_similarity
from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import CitationResult, PaperDocument, Reference

log = logging.getLogger(__name__)

VERIFIED_THRESHOLD = 0.50

_VISION_PROMPT = (
    "This image shows a page from the References/Bibliography section of an academic paper. "
    "Extract ALL references visible on this page as a JSON array. "
    "Each object must have these fields (use null if not present): "
    "title (paper title), authors (author string), year (4-digit string), "
    "doi (DOI string without https://doi.org/), url (URL string). "
    "If a reference spans across a page break, include whatever is visible. "
    "Return ONLY a valid JSON array, no explanation."
)


# ── Vision helpers ────────────────────────────────────────────────────────────

def _find_ref_pages(pdf_path: str) -> list[int]:
    """参照セクションのページ番号（0-indexed）を返す。"""
    doc = fitz.open(pdf_path)
    ref_pages = []
    in_refs = False
    for i in range(len(doc)):
        text = doc[i].get_text().lower()
        if not in_refs:
            if re.search(r"^\s*references?\s*$", text, re.MULTILINE):
                in_refs = True
                ref_pages.append(i)
        else:
            if re.search(r"^\s*(appendix|supplementary)\s*$", text, re.MULTILINE) and i > ref_pages[0] + 1:
                break
            ref_pages.append(i)
    doc.close()
    return ref_pages


def _page_to_png(pdf_path: str, page_no: int, dpi: int = 150) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def _extract_refs_from_page(image: bytes, model: str) -> list[dict]:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": _VISION_PROMPT, "images": [image]}],
        think=False,
    )
    raw = response["message"]["content"]
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        log.warning("[citation/vision] No JSON array in response: %s", raw[:100])
        return []
    json_str = m.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # フォールバック: 不正なバックスラッシュエスケープを修正してリトライ
        # 有効なJSONエスケープ文字 (" \ / b f n r t u) 以外の \ を \\ に置換
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            log.warning("[citation/vision] JSON parse error (unfixable): %s", e)
            return []


_CONCAT_FIELDS = {"title", "authors"}


def _merge_item(a: dict, b: dict) -> dict:
    merged = dict(a)
    for k, v in b.items():
        if not v:
            continue
        if not merged.get(k):
            merged[k] = v
        elif k in _CONCAT_FIELDS:
            merged[k] = merged[k] + " " + v
    return merged


def _merge_page_boundaries(pages_items: list[list[dict]]) -> list[dict]:
    """ページ境界をまたぐ不完全エントリを後処理でマージする。"""
    if not pages_items:
        return []

    result_pages: list[list[dict]] = [list(pages_items[0])]

    for next_page in pages_items[1:]:
        if not next_page:
            continue
        prev_page = result_pages[-1]
        if not prev_page:
            result_pages.append(list(next_page))
            continue

        last = prev_page[-1]
        first = next_page[0]

        last_incomplete = not last.get("year") or not last.get("title") or not last.get("authors")
        first_incomplete = not first.get("year") or not first.get("title") or not first.get("authors")

        if last_incomplete or first_incomplete:
            merged = _merge_item(last, first)
            log.info(
                "[citation/vision] Merging page-boundary entries: '%s' + fragment",
                (last.get("title") or "")[:50],
            )
            prev_page[-1] = merged
            result_pages.append(list(next_page[1:]))
        else:
            result_pages.append(list(next_page))

    return [item for page in result_pages for item in page]


def _dedup_refs(all_items: list[dict]) -> list[dict]:
    seen_titles: set[str] = set()
    result = []
    for item in all_items:
        # title・authors・year がすべて空のエントリは除外
        if not item.get("title") and not item.get("authors") and not item.get("year"):
            log.debug("[citation/vision] Skipping empty entry: %s", item)
            continue
        title = str(item.get("title") or "").lower().strip()
        if title and title in seen_titles:
            continue
        if title:
            seen_titles.add(title)
        result.append(item)
    return result


def extract_references_vision(
    pdf_path: str,
    model: str,
    on_page: callable = None,
    cancel_event=None,
) -> list[Reference]:
    """Vision LLM を使って PDF の参照リストを抽出する。

    on_page(page_no: int, refs_so_far: list[Reference]) が指定された場合、
    各ページ処理後にコールバックを呼ぶ。
    cancel_event が set された場合、次ページ開始前に中断する。
    """
    pages = _find_ref_pages(pdf_path)
    if not pages:
        log.warning("[citation/vision] References section not found in PDF")
        return []

    pages = pages[:20]
    log.info("[citation/vision] Reference pages: %s", [p + 1 for p in pages])

    pages_items: list[list[dict]] = []
    page_nos: list[int] = []  # 各ページのPDFページ番号（1-indexed）

    for page_no in pages:
        if cancel_event and cancel_event.is_set():
            log.info("[citation/vision] Cancelled before page %d", page_no + 1)
            break
        image = _page_to_png(pdf_path, page_no)
        items = _extract_refs_from_page(image, model)
        log.info("[citation/vision] Page %d: %d refs", page_no + 1, len(items))
        if not items:
            log.info("[citation/vision] 0 refs on page %d, stopping", page_no + 1)
            break
        pages_items.append(items)
        page_nos.append(page_no + 1)

        if on_page:
            # ここまでの結果を途中経過として渡す（マージ・重複除去済み）
            partial_items = _dedup_refs(_merge_page_boundaries(pages_items))
            partial_refs = _build_refs(partial_items, pages_items, page_nos)
            on_page(page_no + 1, partial_refs)

    all_items = _dedup_refs(_merge_page_boundaries(pages_items))
    refs = _build_refs(all_items, pages_items, page_nos)
    log.info("[citation/vision] Extracted %d references", len(refs))
    return refs


def _build_refs(
    all_items: list[dict],
    pages_items: list[list[dict]],
    page_nos: list[int],
) -> list[Reference]:
    """抽出済みアイテムリストから Reference オブジェクトを生成する。"""
    # タイトルからページ番号を逆引きするマップを構築
    title_to_page: dict[str, int] = {}
    for items, pno in zip(pages_items, page_nos):
        for item in items:
            t = str(item.get("title") or "").lower().strip()
            if t and t not in title_to_page:
                title_to_page[t] = pno

    refs = []
    for i, item in enumerate(all_items):
        t = str(item.get("title") or "").lower().strip()
        page_no = title_to_page.get(t)
        refs.append(Reference(
            id=f"[{i+1}]",
            raw_text=json.dumps(item),
            title=item.get("title"),
            authors=item.get("authors"),
            year=str(item["year"]) if item.get("year") else None,
            doi=item.get("doi"),
            url=item.get("url"),
            page_no=page_no,
        ))
    return refs


# ── Stage run ─────────────────────────────────────────────────────────────────

def run(
    doc: PaperDocument,
    llm: OllamaClient,
    skip_semantic_scholar: bool = False,
    on_page: callable = None,
    cancel_event=None,
) -> PaperDocument:
    """Stage 2: 参照抽出（vision）+ Semantic Scholar による実在確認。"""
    if not doc.pdf_path:
        log.warning("[citation] No pdf_path in doc, skipping reference extraction")
        doc.references = []
        doc.citation_results = []
        return doc

    doc.references = extract_references_vision(
        doc.pdf_path, llm.model, on_page=on_page, cancel_event=cancel_event,
    )

    results: list[CitationResult] = []

    for ref in doc.references:
        if skip_semantic_scholar or not ref.title:
            result = CitationResult(
                ref_id=ref.id,
                status="UNRESOLVABLE",
                note="Semantic Scholar スキップ" if skip_semantic_scholar else "no title extracted",
            )
            ref.existence_status = "UNRESOLVABLE"
            results.append(result)
            continue

        ss_paper = search_semantic_scholar(ref.title)

        if ss_paper is None:
            result = CitationResult(
                ref_id=ref.id,
                status="HALLUCINATED",
                note="No result from Semantic Scholar",
            )
            ref.existence_status = "HALLUCINATED"
        else:
            matched_title = ss_paper.get("title", "")
            score = title_similarity(ref.title, matched_title)

            status = "VERIFIED" if score >= VERIFIED_THRESHOLD else "UNRESOLVABLE"
            result = CitationResult(
                ref_id=ref.id,
                status=status,
                matched_title=matched_title,
                match_score=score,
            )
            ref.existence_status = status

        results.append(result)

    doc.citation_results = results
    return doc
