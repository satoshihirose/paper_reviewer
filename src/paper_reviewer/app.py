"""Gradio web UI for Paper Reviewer."""

from __future__ import annotations

import html as _html
import tempfile
from pathlib import Path
from typing import Generator

import uuid

import gradio as gr
from fastapi.responses import FileResponse, Response

from paper_reviewer.pipeline import run_pipeline_generator
from paper_reviewer import report_html

# セッションごとの PDF パス管理（token → file path）
_pdf_sessions: dict[str, str] = {}


def _pdf_embed_html(token: str) -> str:
    return (
        f'<iframe name="pdf-frame" src="/pdf/{token}#navpanes=0" width="100%" '
        'style="height:calc(100vh - 120px);min-height:400px;'
        'border:1px solid #ddd;border-radius:6px;display:block;"></iframe>'
    )


def _error_html(message: str) -> str:
    escaped = _html.escape(message)
    return f'<p style="color:#c62828;padding:8px 0;white-space:pre-wrap;">{escaped}</p>'


def review_handler(
    pdf_file: str | None,
    model: str,
    skip_semantic_scholar: bool,
    _prev_token: str | None,
) -> Generator:
    if not pdf_file:
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            _error_html("❌ PDF ファイルを選択してください。"),
            _prev_token,
        )
        return

    # セッションごとに一意トークンを発行
    token = uuid.uuid4().hex[:12]
    _pdf_sessions[token] = pdf_file
    pdf_url = f"/pdf/{token}"

    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        _pdf_embed_html(token),
        report_html.initial_html(pdf_url=pdf_url),
        token,
    )

    pdf_cache_dir = Path(tempfile.mkdtemp(prefix="paper_reviewer_"))

    try:
        for status, doc in run_pipeline_generator(
            pdf_file,
            model=model,
            skip_semantic_scholar=skip_semantic_scholar,
            pdf_cache_dir=pdf_cache_dir,
        ):
            if doc is None:
                yield (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            else:
                yield (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    report_html.to_html(doc, pdf_url=pdf_url),
                    gr.update(),
                )
    except Exception as exc:
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(),
            _error_html(f"❌ エラーが発生しました:\n{exc}"),
            gr.update(),
        )


_RESTORE_JS = """
(function() {
    var openKeys = new Set();

    function setup() {
        var host = document.getElementById('report-html');
        if (!host) { setTimeout(setup, 300); return; }

        // Capture toggle events (toggle does not bubble, so useCapture=true)
        host.addEventListener('toggle', function(e) {
            if (e.target.tagName !== 'DETAILS') return;
            var key = e.target.getAttribute('data-prv-key');
            if (!key) return;
            if (e.target.open) openKeys.add(key);
            else openKeys.delete(key);
        }, true);

        // After each DOM update, restore open state
        new MutationObserver(function() {
            host.querySelectorAll('details[data-prv-key]').forEach(function(d) {
                if (openKeys.has(d.getAttribute('data-prv-key'))) d.open = true;
            });
        }).observe(host, { childList: true, subtree: true });
    }

    setup();
})();
"""

with gr.Blocks(title="Paper Reviewer", js=_RESTORE_JS) as demo:

    pdf_token = gr.State(value=None)

    gr.Markdown("# Paper Reviewer\nローカル LLM による論文品質チェック", elem_id="app-header")
    home_btn = gr.Button("home", elem_id="home-btn")

    # ── Input screen ──────────────────────────────────────────────────────────
    with gr.Column(visible=True) as input_screen:
        pdf_input = gr.File(file_types=[".pdf"], label="PDF ファイルを選択")
        model_dropdown = gr.Dropdown(
            choices=["qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b"],
            value="qwen3.5:9b",
            label="モデル",
        )
        skip_ss_check = gr.Checkbox(
            label="Semantic Scholar スキップ",
            value=True,
        )
        run_btn = gr.Button("レビュー開始", variant="primary")

    # ── Results screen ────────────────────────────────────────────────────────
    with gr.Column(visible=False) as results_screen:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                pdf_html_out = gr.HTML()
            with gr.Column(scale=1):
                report_out = gr.HTML(elem_id="report-html")

    # ── Event bindings ────────────────────────────────────────────────────────
    home_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[input_screen, results_screen],
    )

    run_btn.click(
        fn=review_handler,
        inputs=[pdf_input, model_dropdown, skip_ss_check, pdf_token],
        outputs=[input_screen, results_screen, pdf_html_out, report_out, pdf_token],
        show_progress="hidden",
        concurrency_limit=None,
    )
