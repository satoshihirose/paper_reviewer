"""Pipeline orchestration: parser → (citation ∥ consistency ∥ repro) → report."""

from __future__ import annotations

import copy
import logging
import queue
import threading
import time
import concurrent.futures
from pathlib import Path
from typing import Generator, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from paper_reviewer.llm import OllamaClient
from paper_reviewer.models import PaperDocument
from paper_reviewer.stages import parser, citation, consistency, repro, report, claims, limitations, theory, checklist as checklist_stage

console = Console()


def run_pipeline(
    input_str: str,
    model: str = "qwen3:8b",
    output_path: Optional[Path] = None,
    skip_semantic_scholar: bool = False,
    pdf_cache_dir: Optional[Path] = None,
) -> PaperDocument:
    """Run the full review pipeline and return the populated PaperDocument."""

    llm = OllamaClient(model=model, name="parser")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:

        task = progress.add_task("[cyan]Parsing PDF and extracting content...", total=None)
        doc = parser.run(input_str, llm, pdf_cache_dir=pdf_cache_dir)
        progress.update(task, description=f"[green]Parsed — {len(doc.claims)} claims, {len(doc.references)} references")

        progress.update(task, description="[cyan]Verifying citations...")
        doc = citation.run(doc, llm, skip_semantic_scholar=skip_semantic_scholar)
        verified = sum(1 for r in doc.citation_results if r.status == "VERIFIED")
        hallucinated = sum(1 for r in doc.citation_results if r.status == "HALLUCINATED")
        progress.update(task, description=f"[green]Citations — {verified} verified, {hallucinated} hallucinated")

        progress.update(task, description="[cyan]Checking internal consistency...")
        doc = consistency.run(doc, llm)
        progress.update(task, description=f"[green]Consistency — {len(doc.consistency_issues)} issues found")

        progress.update(task, description="[cyan]Checking reproducibility readiness...")
        doc = repro.run(doc, llm)
        repro_total = doc.repro_score.total if doc.repro_score else 0
        progress.update(task, description=f"[green]Reproducibility — score: {repro_total}/10")

    report.print_report(doc)

    if output_path:
        report.save_json(doc, output_path)

    return doc


def run_pipeline_generator(
    input_str: str,
    model: str = "qwen3:8b",
    skip_semantic_scholar: bool = False,
    pdf_cache_dir: Optional[Path] = None,
) -> Generator[tuple[str, Optional[PaperDocument]], None, None]:
    """Generator version: runs checkers in parallel after parsing.

    Parser runs first (sequential), then citation / consistency / repro run
    concurrently. Results are yielded as each checker completes.
    """
    llm = OllamaClient(model=model, name="parser")

    # ── Step 1: Parser ────────────────────────────────────────────────────────
    yield ("PDF 解析・クレーム抽出中...", None)
    log.info("=== Request: input=%r model=%s ===", input_str, model)
    log.info("Parser: start")
    t0 = time.perf_counter()
    doc = parser.run(input_str, llm, pdf_cache_dir=pdf_cache_dir)
    log.info("Parser: done in %.1fs — claims=%d references=%d", time.perf_counter() - t0, len(doc.claims), len(doc.references))

    # Immediately surface parser results; checker cards show "チェック中..."
    yield ("", doc)

    # ── Step 2: Checkers (claims → limitations → theory の順次実行 ∥ citation) ──
    result_queue: queue.Queue = queue.Queue()
    base = copy.deepcopy(doc)  # snapshot for each checker thread
    cancel_event = threading.Event()

    # Sequential chain: claims → limitations → theory → item4 → … → 16
    # Citation runs in parallel with the chain.
    _CHAIN_KEYS = ["claims", "limitations", "theory"] + checklist_stage.item_keys()
    _prev_done: dict[str, threading.Event] = {}
    _self_done: dict[str, threading.Event] = {}
    start_event = threading.Event()
    start_event.set()
    prev = start_event
    for key in _CHAIN_KEYS:
        _prev_done[key] = prev
        done = threading.Event()
        _self_done[key] = done
        prev = done

    def _make_runner(key: str):
        def _run():
            _prev_done[key].wait()
            if cancel_event.is_set():
                return
            log.info("%s: start", key)
            t = time.perf_counter()
            d = copy.deepcopy(base)
            if key == "claims":
                d = claims.run(d, OllamaClient(model=model, name=key))
                result_queue.put(("claims", d.claims_result, None))
            elif key == "limitations":
                d = limitations.run(d, OllamaClient(model=model, name=key))
                result_queue.put(("limitations", d.limitations_result, None))
            elif key == "theory":
                d = theory.run(d, OllamaClient(model=model, name=key))
                result_queue.put(("theory", d.theory_result, None))
            else:
                d = checklist_stage.run_item(key, d, OllamaClient(model=model, name=f"item{key}"))
                result_queue.put(("checklist", key, d.checklist_results.get(key)))
            log.info("%s: done in %.1fs", key, time.perf_counter() - t)
            _self_done[key].set()
        return _run

    def _run_citation() -> None:
        log.info("Citation: start")
        t = time.perf_counter()

        def _on_page(page_no: int, refs_so_far: list) -> None:
            result_queue.put(("citation_partial", refs_so_far, None))

        d = citation.run(
            copy.deepcopy(base),
            OllamaClient(model=model, name="citation"),
            skip_semantic_scholar=skip_semantic_scholar,
            on_page=_on_page,
            cancel_event=cancel_event,
        )
        log.info("Citation: done in %.1fs", time.perf_counter() - t)
        result_queue.put(("citation", d.citation_results, d.references))

    # チェーン全体 + citation を並列起動
    total_checkers = len(_CHAIN_KEYS) + 1  # +1 for citation
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_checkers)
    try:
        for key in _CHAIN_KEYS:
            executor.submit(_make_runner(key))
        executor.submit(_run_citation)

        received = 0
        while received < total_checkers:
            try:
                item = result_queue.get(timeout=10)
            except queue.Empty:
                yield ("", None)  # keepalive: prevents SSE connection timeout
                continue

            checker = item[0]
            if checker == "citation_partial":
                doc.references = item[1]
                yield ("", doc)
                continue  # received をインクリメントしない

            if checker == "claims":
                doc.claims_result = item[1]
            elif checker == "limitations":
                doc.limitations_result = item[1]
            elif checker == "theory":
                doc.theory_result = item[1]
            elif checker == "checklist":
                doc.checklist_results[item[1]] = item[2]
            elif checker == "citation":
                doc.citation_results = item[1]
                doc.references = item[2]
                log.info("Pipeline: doc.references set to %d refs", len(doc.references) if doc.references else 0)

            received += 1
            yield ("", doc)
    except GeneratorExit:
        log.info("Pipeline: client disconnected, cancelling background tasks")
        cancel_event.set()
        raise
    finally:
        cancel_event.set()  # 正常終了・例外いずれでも必ずセット
        executor.shutdown(wait=False)  # 実行中スレッドを待たずに即時返却
