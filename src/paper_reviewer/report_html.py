"""Convert PaperDocument to a styled HTML string for Gradio gr.HTML()."""

from __future__ import annotations

import html
import re
from paper_reviewer.models import PaperDocument, LimitationsResult, TheoryResult
from paper_reviewer.stages import checklist as checklist_stage

_CARD_STYLE = (
    "border:1px solid #bbb;border-radius:8px;padding:16px;"
    "margin-bottom:16px;background:#fff;"
)
_COLOR_OK   = "#2e7d32"
_COLOR_WARN = "#d97706"
_COLOR_BAD  = "#c62828"
_COLOR_SKIP = "#424242"

_GLOBAL_STYLE = (
    "<style>"
    "@keyframes _prv_spin{to{transform:rotate(360deg)}}"
    "._prv_tip{position:relative;display:inline-block;cursor:help;}"
    "._prv_tip ._prv_tip_t{"
    "visibility:hidden;opacity:0;background:#2a2a2a;color:#e0e0e0;font-size:0.8em;"
    "line-height:1.6;border-radius:6px;padding:10px 12px;position:absolute;"
    "z-index:999;width:340px;bottom:130%;left:50%;transform:translateX(-50%);"
    "transition:opacity 0.15s;pointer-events:auto;white-space:normal;}"
    "._prv_tip:hover ._prv_tip_t{visibility:visible;opacity:1;}"
    "._prv_tip_t::after{content:'';position:absolute;bottom:-10px;left:0;right:0;height:10px;}"
    "._prv_tip_t a{color:#90caf9;text-decoration:underline;}"
    "._prv_tip_t b{color:#fff;}"
    "._prv_verdict summary{list-style:none;}"
    "._prv_verdict summary::-webkit-details-marker{display:none;}"
    "._prv_verdict summary ._prv_chevron{"
    "display:inline-block;transition:transform 0.2s;font-size:0.8em;color:#777;}"
    "._prv_verdict[open] summary ._prv_chevron{transform:rotate(90deg);}"
    "._prv_verdict summary:hover{opacity:0.8;}"
    "</style>"
)

_CHECKING_HTML = (
    '<div style="display:flex;align-items:center;gap:10px;padding:8px 0;">'
    '<div style="width:20px;height:20px;flex-shrink:0;border:2.5px solid #ddd;'
    "border-top-color:#1565c0;border-radius:50%;"
    'animation:_prv_spin 0.85s linear infinite;"></div>'
    '<span style="color:#333;font-size:0.9em;">チェック中...</span>'
    "</div>"
)

# (title, description) for each checker card, in display order.
# Index 0=Claims, 1=Limitations, 2=Theory, 3+=checklist items 4-16, last=Citations.
_CARD_DEFS = [
    (
        "1. Claims（主張の妥当性）",
        "Abstract・Introduction で述べられた主要な主張が、論文の実験・理論結果と整合しているかを確認します。"
        "主張の一般化スコープが実験結果に対して過大でないか、貢献の前提・限界が明示されているかを評価します。",
    ),
    (
        "2. Limitations（限界の開示）",
        "論文の主張・手法・実験に対する限界が誠実に議論されているかを確認します。"
        "強い仮定とその違反への頑健性、スコープの明示、潜在的な失敗ケースや社会的影響の考察を評価します。",
    ),
    (
        "3. Theory, Assumptions and Proofs（理論・仮定・証明）",
        "理論的結果を含む場合に、すべての定理・補題の仮定が明示されているか、完全な証明が提供されているかを確認します。"
        "証明が補足資料にある場合の proof sketch の有無、定理の適切な参照も評価します。"
        "理論的結果を含まない論文は NA（対象外）となります。",
    ),
    (
        "4. Experimental Result Reproducibility",
        "提案アルゴリズム・アーキテクチャ・モデルが再現できるだけの詳細が記述されているかを確認します。"
        "コード非公開の場合も、再現への何らかの経路が示されているかを評価します。",
    ),
    (
        "5. Open Access to Data and Code",
        "実験結果を再現するために必要なコード・データ・手順が補足資料または URL で提供されているかを確認します。",
    ),
    (
        "6. Experimental Setting/Details",
        "データ分割・ハイパーパラメータ・選定方法など、実験設定の詳細が記述されているかを確認します。",
    ),
    (
        "7. Experiment Statistical Significance",
        "主要な実験結果にエラーバー・信頼区間・統計的有意性検定が適切に報告されているかを確認します。",
    ),
    (
        "8. Experiments Compute Resources",
        "各実験に必要な計算資源（ハードウェア種別・メモリ・実行時間）が再現に十分な粒度で記述されているかを確認します。",
    ),
    (
        "9. Code of Ethics",
        "研究が NeurIPS Code of Ethics に準拠しているかを確認します。プライバシー・公平性・デュアルユースリスク・研究誠実性などを評価します。",
    ),
    (
        "10. Broader Impacts",
        "研究の潜在的な負の社会的影響（誤情報・監視・差別・プライバシー侵害など）が議論されているかを確認します。",
    ),
    (
        "11. Safeguards",
        "誤用リスクの高いモデルやデータセットを公開する場合に、適切な安全策（利用制限・コンテンツフィルタなど）が講じられているかを確認します。",
    ),
    (
        "12. Licenses for Existing Assets",
        "使用した既存コード・データ・モデルについて、作成者の引用とライセンス・利用規約の遵守が示されているかを確認します。",
    ),
    (
        "13. New Assets",
        "新たに公開するデータセット・コード・モデルについて、ライセンス・制限事項・同意取得などのドキュメントが提供されているかを確認します。",
    ),
    (
        "14. Crowdsourcing and Research with Human Subjects",
        "クラウドソーシングや人間参加者を用いた研究の場合、指示文・補償・スクリーンショットなどの詳細が開示されているかを確認します。",
    ),
    (
        "15. Institutional Review Board (IRB) Approvals",
        "人間を対象とした研究において、IRB 承認（または同等の倫理審査）の取得と参加者リスクの記述があるかを確認します。",
    ),
    (
        "16. Declaration of LLM Usage",
        "LLM がコア手法として重要・独創的・非標準的な役割を果たしている場合に、その使用が適切に宣言されているかを確認します。",
    ),
    (
        "引用文献の実在確認",
        "参考文献リストを Semantic Scholar と照合し、存在しない文献（幻覚引用）を検出します。"
        "引用が実在しない場合、その主張の根拠が成り立ちません。",
    ),
]



def _esc(text: str) -> str:
    return html.escape(str(text))


def _linkify_pages(text: str, pdf_url: str = "/current-pdf") -> str:
    """テキストをHTMLエスケープしたうえで、（p.N）パターンをPDFページリンクに変換する。"""
    escaped = html.escape(str(text))
    def _replace(m: re.Match) -> str:
        n = m.group(1)
        return (
            f'<a href="{pdf_url}#page={n}&navpanes=0" target="pdf-frame" '
            f'style="color:#1565c0;text-decoration:none;font-weight:bold;">'
            f'（p.{n}）</a>'
        )
    return re.sub(r'（p\.(\d+)）', _replace, escaped).replace('\n', '<br>')


def _score_color(pct: int) -> str:
    if pct >= 70:
        return _COLOR_OK
    if pct >= 40:
        return _COLOR_WARN
    return _COLOR_BAD


def _bar(value: int, max_val: int, width: int = 160) -> str:
    pct = int(100 * value / max_val) if max_val else 0
    color = _score_color(pct)
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;">'
        f'<div style="background:#bbb;border-radius:4px;width:{width}px;height:10px;">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;"></div>'
        f'</div>'
        f'<span style="font-weight:bold;color:{color};">{value}/{max_val}</span>'
        f'</div>'
    )


def _badge(label: str, bg: str) -> str:
    return (
        f'<span style="display:inline-block;background:{bg};color:#fff;'
        f'font-size:0.72em;font-weight:bold;padding:1px 6px;border-radius:3px;'
        f'vertical-align:middle;margin-right:6px;">{label}</span>'
    )


def _card_header(title: str, description: str) -> str:
    return (
        f'<div style="margin-bottom:12px;">'
        f'<div style="font-size:1.05em;font-weight:bold;color:#111;">{_esc(title)}</div>'
        f'<p style="margin:4px 0 0;color:#222;font-size:0.9em;line-height:1.5;">{_esc(description)}</p>'
        f'</div>'
    )


def _finding(status: str, text: str, sub: str = "") -> str:
    if status == "ok":
        badge = _badge("OK", _COLOR_OK)
        color = _COLOR_OK
    elif status == "ng":
        badge = _badge("NG", _COLOR_BAD)
        color = _COLOR_BAD
    elif status == "warn":
        badge = _badge("WARN", _COLOR_WARN)
        color = _COLOR_WARN
    else:
        badge = _badge("—", _COLOR_SKIP)
        color = "#111"

    s = (
        f'<div style="margin:6px 0;font-size:0.9em;line-height:1.4;">'
        f'{badge}<span style="color:{color};font-weight:500;">{_esc(text)}</span>'
    )
    if sub:
        s += f'<div style="margin:2px 0 0 28px;color:#222;font-size:0.88em;">{_esc(sub)}</div>'
    s += '</div>'
    return s


def _finding_detail(text: str, color: str = "#111") -> str:
    return f'<div style="margin:3px 0 3px 28px;font-size:0.88em;color:{color};">{_esc(text)}</div>'


def _tip(tooltip_html: str) -> str:
    """ホバーするとツールチップが表示される ⓘ アイコン。tooltip_html は HTML として挿入される。"""
    return (
        f'<span class="_prv_tip" style="margin-left:5px;color:#1976d2;font-size:0.85em;">'
        f'ⓘ<span class="_prv_tip_t">{tooltip_html}</span>'
        f'</span>'
    )


def _ts(name: str, quote: str, url: str) -> str:
    """ツールチップ内の1参照源ブロックを生成する（source block）。"""
    return (
        f'<b>{_esc(name)}</b><br>'
        f'{_esc(quote)}<br>'
        f'<a href="{_esc(url)}" target="_blank">→ 原文を確認</a>'
    )


def _excerpt_with_page(excerpt: str | None, page: int | None) -> str:
    """抜粋テキストにページ番号サフィックスを付加する。"""
    if not excerpt:
        return ""
    if page:
        return f"（p.{page}）  {excerpt}"
    return excerpt


def _cat_header(label: str) -> str:
    """再現性カード内のカテゴリ見出し（Code / Data / Compute / Method）。"""
    return (
        f'<div style="font-weight:bold;color:#555;font-size:0.82em;letter-spacing:.04em;'
        f'text-transform:uppercase;margin:14px 0 4px;padding-bottom:3px;'
        f'border-bottom:1px solid #e0e0e0;">{_esc(label)}</div>'
    )


def _repro_finding(status: str, subtitle: str, detail: str, tooltip: str) -> str:
    """サブタイトル・詳細テキスト・ツールチップ付きの確認項目行。"""
    if status == "ok":
        badge = _badge("OK", _COLOR_OK)
        sub_color = _COLOR_OK
    elif status == "ng":
        badge = _badge("NG", _COLOR_BAD)
        sub_color = _COLOR_BAD
    elif status == "warn":
        badge = _badge("WARN", _COLOR_WARN)
        sub_color = _COLOR_WARN
    elif status == "none":
        badge = _badge("—", "#9e9e9e")
        sub_color = "#888"
    else:
        badge = _badge("—", _COLOR_SKIP)
        sub_color = _COLOR_SKIP

    return (
        f'<div style="margin:5px 0;">'
        f'<div style="font-size:0.88em;color:#666;margin-bottom:1px;">'
        f'{_esc(subtitle)}{_tip(tooltip)}</div>'
        f'<div style="font-size:0.9em;line-height:1.6;">'
        f'{badge}<span style="color:{sub_color};font-weight:500;">{_linkify_pages(detail)}</span>'
        f'</div>'
        f'</div>'
    )


def _wrap(inner: str) -> str:
    return (
        '<div style="height:calc(100vh - 120px);overflow-y:auto;overflow-x:hidden;'
        'box-sizing:border-box;">'
        '<div style="font-family:sans-serif;padding:0 16px 32px;color:#111;">'
        + _GLOBAL_STYLE
        + inner
        + '</div>'
        '</div>'
    )


def initial_html(pdf_url: str = "/current-pdf") -> str:
    """Return HTML with all checker cards showing a spinner (before any checker runs)."""
    parts: list[str] = []
    for title, desc in _CARD_DEFS:
        parts.append(f'<div style="{_CARD_STYLE}">')
        parts.append(_card_header(title, desc))
        parts.append(_CHECKING_HTML)
        parts.append('</div>')
    return _wrap("".join(parts))


_VERDICT_COLOR = {
    "Yes": _COLOR_OK,
    "Partial": _COLOR_WARN,
    "No": _COLOR_BAD,
    "NA": _COLOR_SKIP,
}

_CLAIM_VERDICT_COLOR = {
    "SUPPORTED": _COLOR_OK,
    "PARTIALLY_SUPPORTED": _COLOR_WARN,
    "UNSUPPORTED": _COLOR_BAD,
    "UNCLEAR": _COLOR_SKIP,
}

_CLAIM_VERDICT_LABEL = {
    "SUPPORTED": "支持",
    "PARTIALLY_SUPPORTED": "部分的",
    "UNSUPPORTED": "不支持",
    "UNCLEAR": "不明",
}


def _checklist_card(
    title: str,
    desc: str,
    overall_verdict: str | None,
    overall_summary: str,
    reasoning: str,
    findings: list,
    issues: list[str],
    prv_key: str = "",
) -> str:
    """汎用チェックリストカード（Limitations・Theory で共用）。"""
    parts: list[str] = []
    parts.append(f'<div style="{_CARD_STYLE}">')
    parts.append(_card_header(title, desc))

    if overall_verdict is None:
        parts.append(_CHECKING_HTML)
    else:
        verdict_color = _VERDICT_COLOR.get(overall_verdict, _COLOR_SKIP)
        verdict_label = {
            "Yes": "✓ 適切", "Partial": "△ 一部問題あり",
            "No": "✗ 問題あり", "NA": "— 対象外",
        }.get(overall_verdict, overall_verdict)

        key_attr = f' data-prv-key="{_esc(prv_key)}"' if prv_key else ""
        parts.append(
            f'<details class="_prv_verdict"{key_attr} style="margin-top:10px;">'
            f'<summary style="cursor:pointer;display:inline-flex;align-items:center;gap:8px;'
            f'padding:6px 12px;border-radius:6px;border:1.5px solid {verdict_color};'
            f'background:{verdict_color}18;user-select:none;">'
            f'<span style="font-size:1em;font-weight:bold;color:{verdict_color};">'
            f'総合判定: {_esc(verdict_label)}</span>'
            f'<span class="_prv_chevron">▶</span>'
            f'</summary>'
        )

        if overall_summary:
            parts.append(
                f'<p style="margin:10px 0;font-size:0.9em;color:#222;line-height:1.6;">'
                f'{_esc(overall_summary)}</p>'
            )

        if findings:
            parts.append('<div style="margin-top:12px;">')
            for f in findings:
                status = f.status if hasattr(f, "status") else "none"
                text = f.text if hasattr(f, "text") else str(f)
                if status == "ok":
                    badge = _badge("OK", _COLOR_OK); fc = _COLOR_OK
                elif status == "ng":
                    badge = _badge("NG", _COLOR_BAD); fc = _COLOR_BAD
                elif status == "warn":
                    badge = _badge("WARN", _COLOR_WARN); fc = _COLOR_WARN
                else:
                    badge = _badge("—", _COLOR_SKIP); fc = "#111"
                parts.append(
                    f'<div style="margin:6px 0;font-size:0.9em;color:{fc};">'
                    f'{badge}{_esc(text)}</div>'
                )
            parts.append('</div>')

        if issues:
            parts.append(
                f'<div style="margin-top:12px;padding:10px 12px;background:#fff3e0;'
                f'border-radius:6px;border-left:3px solid {_COLOR_WARN};">'
                f'<div style="font-weight:bold;font-size:0.88em;color:{_COLOR_WARN};margin-bottom:6px;">'
                f'指摘事項</div>'
            )
            for issue in issues:
                parts.append(
                    f'<div style="font-size:0.88em;color:#333;margin:3px 0;">• {_esc(issue)}</div>'
                )
            parts.append('</div>')

        if reasoning:
            parts.append(
                f'<div style="margin-top:12px;font-size:0.85em;color:#333;line-height:1.6;'
                f'white-space:pre-wrap;background:#fafafa;padding:10px;border-radius:4px;">'
                f'{_esc(reasoning)}</div>'
            )

        parts.append('</details>')

    parts.append('</div>')
    return "".join(parts)


def _claims_card(doc: PaperDocument) -> str:
    title, desc = _CARD_DEFS[0]  # Claims is now index 0
    parts: list[str] = []
    parts.append(f'<div style="{_CARD_STYLE}">')
    parts.append(_card_header(title, desc))

    cr = doc.claims_result
    if cr is None:
        parts.append(_CHECKING_HTML)
    else:
        verdict_color = _VERDICT_COLOR.get(cr.overall_verdict, _COLOR_SKIP)
        verdict_label = {"Yes": "✓ 適切", "Partial": "△ 一部問題あり", "No": "✗ 問題あり", "NA": "— 評価不能"}.get(
            cr.overall_verdict, cr.overall_verdict
        )
        parts.append(
            f'<details class="_prv_verdict" data-prv-key="claims" style="margin-top:10px;">'
            f'<summary style="cursor:pointer;display:inline-flex;align-items:center;gap:8px;'
            f'padding:6px 12px;border-radius:6px;border:1.5px solid {verdict_color};'
            f'background:{verdict_color}18;user-select:none;">'
            f'<span style="font-size:1em;font-weight:bold;color:{verdict_color};">'
            f'総合判定: {_esc(verdict_label)}</span>'
            f'<span class="_prv_chevron">▶</span>'
            f'</summary>'
        )

        if cr.overall_summary:
            parts.append(
                f'<p style="margin:10px 0;font-size:0.9em;color:#222;line-height:1.6;">'
                f'{_esc(cr.overall_summary)}</p>'
            )

        if cr.claims:
            parts.append('<div style="margin-top:12px;">')
            for cv in cr.claims:
                c_color = _CLAIM_VERDICT_COLOR.get(cv.verdict, _COLOR_SKIP)
                c_label = _CLAIM_VERDICT_LABEL.get(cv.verdict, cv.verdict)
                badge = (
                    f'<span style="background:{c_color};color:#fff;font-size:0.72em;font-weight:bold;'
                    f'padding:1px 6px;border-radius:3px;margin-right:6px;vertical-align:middle;flex-shrink:0;">'
                    f'{_esc(c_label)}</span>'
                )
                parts.append(
                    f'<div style="margin:8px 0;padding:8px 10px;background:#f5f5f5;border-radius:6px;'
                    f'border-left:3px solid {c_color};">'
                    f'<div style="font-size:0.9em;color:#111;margin-bottom:4px;">{badge}{_esc(cv.text)}</div>'
                    f'<div style="font-size:0.82em;color:#555;margin-left:2px;">根拠: {_esc(cv.evidence)}</div>'
                    f'</div>'
                )
            parts.append('</div>')

        if cr.issues:
            parts.append(
                f'<div style="margin-top:12px;padding:10px 12px;background:#fff3e0;'
                f'border-radius:6px;border-left:3px solid {_COLOR_WARN};">'
                f'<div style="font-weight:bold;font-size:0.88em;color:{_COLOR_WARN};margin-bottom:6px;">'
                f'指摘事項</div>'
            )
            for issue in cr.issues:
                parts.append(
                    f'<div style="font-size:0.88em;color:#333;margin:3px 0;">• {_esc(issue)}</div>'
                )
            parts.append('</div>')

        if cr.reasoning:
            parts.append(
                f'<div style="margin-top:12px;font-size:0.85em;color:#333;line-height:1.6;'
                f'white-space:pre-wrap;background:#fafafa;padding:10px;border-radius:4px;">'
                f'{_esc(cr.reasoning)}</div>'
            )

        parts.append('</details>')

    parts.append('</div>')
    return "".join(parts)


def to_html(doc: PaperDocument, pdf_url: str = "/current-pdf") -> str:
    parts: list[str] = []

    # ── Claims（主張の妥当性） ────────────────────────────────────────────────
    parts.append(_claims_card(doc))

    # ── Limitations（限界の開示） ─────────────────────────────────────────────
    lr = doc.limitations_result
    title, desc = _CARD_DEFS[1]
    parts.append(_checklist_card(
        title=title, desc=desc,
        overall_verdict=lr.overall_verdict if lr else None,
        overall_summary=lr.overall_summary if lr else "",
        reasoning=lr.reasoning if lr else "",
        findings=lr.findings if lr else [],
        issues=lr.issues if lr else [],
        prv_key="limitations",
    ))

    # ── Theory, Assumptions and Proofs（理論・仮定・証明） ────────────────────
    tr = doc.theory_result
    title, desc = _CARD_DEFS[2]
    parts.append(_checklist_card(
        title=title, desc=desc,
        overall_verdict=tr.overall_verdict if tr else None,
        overall_summary=tr.overall_summary if tr else "",
        reasoning=tr.reasoning if tr else "",
        findings=tr.findings if tr else [],
        issues=tr.issues if tr else [],
        prv_key="theory",
    ))

    # ── Items 4–16（汎用チェックリストカード） ────────────────────────────────
    for i, key in enumerate(checklist_stage.item_keys()):  # "4", "5", ..., "16"
        card_index = 3 + i  # _CARD_DEFS[3] = item 4, [4] = item 5, ...
        title, desc = _CARD_DEFS[card_index]
        cr = doc.checklist_results.get(key)
        parts.append(_checklist_card(
            title=title, desc=desc,
            overall_verdict=cr.overall_verdict if cr else None,
            overall_summary=cr.overall_summary if cr else "",
            reasoning=cr.reasoning if cr else "",
            findings=cr.findings if cr else [],
            issues=cr.issues if cr else [],
            prv_key=f"item-{key}",
        ))

    # ── 引用文献の実在確認 ────────────────────────────────────────────────────
    title, desc = _CARD_DEFS[-1]
    parts.append(f'<div style="{_CARD_STYLE}">')
    parts.append(_card_header(title, desc))

    if doc.citation_results is None and not doc.references:
        parts.append(_CHECKING_HTML)
    else:
        if doc.citation_results is not None:
            total_cit    = len(doc.citation_results)
            verified     = sum(1 for r in doc.citation_results if r.status == "VERIFIED")
            hallucinated = [r for r in doc.citation_results if r.status == "HALLUCINATED"]
            unresolvable = sum(1 for r in doc.citation_results if r.status == "UNRESOLVABLE")

            parts.append('<div style="margin-top:10px;">')
            if total_cit == 0:
                parts.append(_finding("ng", "引用文献の抽出に失敗しました（参照フォーマットの自動検出に失敗）"))
            else:
                parts.append(_finding("ok", f"{verified} 件 実在確認済み（全 {total_cit} 件中）"))
                if unresolvable:
                    parts.append(_finding("skip", f"{unresolvable} 件 照合不能（タイトル不明・API エラー等）"))
                if hallucinated:
                    parts.append(_finding(
                        "ng", f"{len(hallucinated)} 件 幻覚引用の疑い",
                        sub="Semantic Scholar に該当する文献が見つかりませんでした",
                    ))
                    for r in hallucinated[:5]:
                        raw = r.matched_title or r.note or "—"
                        parts.append(_finding_detail(f"{r.ref_id}  {raw}", color=_COLOR_BAD))
            parts.append('</div>')
        else:
            parts.append(
                f'<div style="margin-top:8px;font-size:0.88em;color:#555;">'
                f'参照文献を抽出中... （{len(doc.references)} 件取得済み）{_CHECKING_HTML}</div>'
            )

        if doc.references:
            is_streaming = doc.citation_results is None
            summary_label = (
                f'抽出中の参照文献（{len(doc.references)} 件）を表示'
                if is_streaming else
                f'抽出された参照文献（{len(doc.references)} 件）を表示'
            )
            parts.append(
                f'<details data-prv-key="ref-list" style="margin-top:14px;">'
                f'<summary style="cursor:pointer;color:#1565c0;font-weight:bold;font-size:0.92em;'
                f'padding:6px 0;border-top:1px solid #ddd;">'
                f'{summary_label}</summary>'
            )
            parts.append('<table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-top:10px;">')
            parts.append(
                '<thead><tr style="background:#e8eaf6;">'
                '<th style="text-align:left;padding:6px 8px;border-bottom:2px solid #9fa8da;color:#1a237e;width:7%;">ID / p.</th>'
                '<th style="text-align:left;padding:6px 8px;border-bottom:2px solid #9fa8da;color:#1a237e;width:18%;">著者</th>'
                '<th style="text-align:left;padding:6px 8px;border-bottom:2px solid #9fa8da;color:#1a237e;width:5%;">年</th>'
                '<th style="text-align:left;padding:6px 8px;border-bottom:2px solid #9fa8da;color:#1a237e;">タイトル</th>'
                '<th style="text-align:left;padding:6px 8px;border-bottom:2px solid #9fa8da;color:#1a237e;width:8%;">DOI / URL</th>'
                '</tr></thead>'
            )
            parts.append('<tbody>')
            for i, ref in enumerate(doc.references):
                status_color = {
                    "VERIFIED": _COLOR_OK,
                    "HALLUCINATED": _COLOR_BAD,
                    "UNRESOLVABLE": "#5c6bc0",
                }.get(ref.existence_status or "", "#111")
                if ref.doi:
                    doi_url_cell = f'<a href="https://doi.org/{_esc(ref.doi)}" target="_blank" style="color:#1565c0;font-weight:bold;">DOI</a>'
                elif ref.url:
                    doi_url_cell = f'<a href="{_esc(ref.url)}" target="_blank" style="color:#1565c0;font-weight:bold;">URL</a>'
                else:
                    doi_url_cell = '<span style="color:#888;">—</span>'
                row_bg = "#fafafa" if i % 2 == 0 else "#fff"
                page_link = (
                    f'<a href="{pdf_url}#page={ref.page_no}&navpanes=0" target="pdf-frame" '
                    f'style="color:#1565c0;text-decoration:none;font-size:0.9em;">p.{ref.page_no}</a>'
                    if ref.page_no else ''
                )
                id_page_cell = (
                    f'<span style="color:{status_color};font-weight:bold;">{_esc(ref.id or "—")}</span>'
                    + (f'<br>{page_link}' if page_link else '')
                )
                parts.append(
                    f'<tr style="border-bottom:1px solid #e0e0e0;background:{row_bg};">'
                    f'<td style="padding:5px 8px;">{id_page_cell}</td>'
                    f'<td style="padding:5px 8px;color:#111;">{_esc((ref.authors or "—")[:100])}</td>'
                    f'<td style="padding:5px 8px;color:#111;font-weight:bold;">{_esc(ref.year or "—")}</td>'
                    f'<td style="padding:5px 8px;color:#111;">{_esc(ref.title or "—")}</td>'
                    f'<td style="padding:5px 8px;">{doi_url_cell}</td>'
                    f'</tr>'
                )
            parts.append('</tbody></table>')
            parts.append('</details>')

    parts.append('</div>')

    return _wrap("".join(parts))
