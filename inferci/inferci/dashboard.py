"""Static, dependency-free HTML dashboard for InferCI.

Renders the accumulated run history (the ledger) into a single self-contained
HTML document: inline CSS only, no JavaScript, no external assets. It surfaces:

  * a summary row (total runs, backends, models);
  * a run table (run_id, time, backend, model, quantization, device,
    pp_tps, tg_tps, ttft_ms, cost when present);
  * a per-spec trend section (first/last tg_tps and delta%);
  * red highlighting for obvious regressions, judged by reusing
    :func:`inferci.regression.compare_runs` on adjacent runs of the same spec.

Usage:
    python -m inferci.dashboard --db inferci.db --out dashboard.html

Constraint: standard library only.
"""
from __future__ import annotations

import argparse
import html
import sys
from collections import defaultdict

from .regression import Verdict, compare_runs
from .schema import RunResult
from .store import Store


def _esc(value) -> str:
    """Escape a value for safe inclusion in HTML text/attributes."""
    return html.escape("" if value is None else str(value), quote=True)


def _num(value, digits: int = 2, fallback: str = "—") -> str:
    """Format a numeric metric, showing ``fallback`` for unset values."""
    if value is None:
        return fallback
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return fallback


def _created_sort_key(run: RunResult):
    return (run.created_at or "", run.run_id or "")


def _device_of(run: RunResult) -> str:
    """Best-effort device label: the explicit ``extra.device`` first, then the
    accelerator kind captured in the environment."""
    extra = run.spec.extra or {}
    device = extra.get("device", "") or run.environment.accelerator.kind or ""
    return device or "—"


def _cost_cell(run: RunResult) -> str:
    """Render the cost column: ``—`` when absent, ``local`` for zero-price
    runs, otherwise instance type + $/1M output tokens."""
    cost = run.cost
    if cost is None:
        return "—"
    has_price = (
        cost.instance_hourly > 0
        or cost.price_per_input_1m > 0
        or cost.price_per_output_1m > 0
    )
    if not has_price:
        return "local"
    parts = []
    if cost.instance_type:
        parts.append(_esc(cost.instance_type))
    if cost.price_per_output_1m > 0:
        parts.append(f"${cost.price_per_output_1m:.4f}/1M out")
    return " · ".join(parts) if parts else "—"


def _group_by_spec(runs: list[RunResult]) -> dict[str, list[RunResult]]:
    """Group runs by canonical spec id, chronologically ordered within each."""
    by_spec: dict[str, list[RunResult]] = defaultdict(list)
    for run in runs:
        by_spec[run.spec.canonical_id()].append(run)
    for spec_id in by_spec:
        by_spec[spec_id].sort(key=_created_sort_key)
    return dict(by_spec)


def _detect_regressions(
    by_spec: dict[str, list[RunResult]],
) -> dict[str, list]:
    """Reuse the regression judge on adjacent (chronological) runs per spec.

    Returns a mapping of run_id -> list of REGRESSION findings. The *candidate*
    (later) run is the one flagged when it regresses against its predecessor.
    """
    flagged: dict[str, list] = {}
    for spec_id, spec_runs in by_spec.items():
        if len(spec_runs) < 2:
            continue
        for prev, cand in zip(spec_runs, spec_runs[1:]):
            findings = compare_runs(prev, cand)
            regressions = [f for f in findings if f.verdict == Verdict.REGRESSION]
            if regressions:
                flagged[cand.run_id] = regressions
    return flagged


def _summary_html(runs: list[RunResult]) -> str:
    backends = sorted({r.spec.backend for r in runs if r.spec.backend})
    models = sorted({r.spec.model_id for r in runs if r.spec.model_id})
    return (
        '<div class="summary">'
        f'<div class="stat"><span class="stat-value">{len(runs)}</span>'
        '<span class="stat-label">total runs</span></div>'
        f'<div class="stat"><span class="stat-value">{len(backends)}</span>'
        '<span class="stat-label">backend(s)</span>'
        f'<span class="stat-detail">{_esc(", ".join(backends))}</span></div>'
        f'<div class="stat"><span class="stat-value">{len(models)}</span>'
        '<span class="stat-label">model(s)</span>'
        f'<span class="stat-detail">{_esc(", ".join(models))}</span></div>'
        "</div>"
    )


def _table_html(
    runs: list[RunResult], regressions: dict[str, list]
) -> str:
    rows = []
    for run in sorted(runs, key=_created_sort_key):
        flagged = run.run_id in regressions
        cls = ' class="regression"' if flagged else ""
        badge = ""
        title = ""
        if flagged:
            notes = " &#183; ".join(
                _esc(f.metric + " " + f.note) for f in regressions[run.run_id]
            )
            badge = '<span class="badge" title="' + _esc(
                " ; ".join(f.fmt() for f in regressions[run.run_id])
            ) + '">REGRESSION</span>'
            title = ' title="' + _esc(
                " ; ".join(f.fmt() for f in regressions[run.run_id])
            ) + '"'
        rows.append(
            "<tr" + cls + ">"
            f"<td><code>{_esc(run.run_id)}</code> {badge}</td>"
            f"<td>{_esc(run.created_at[:19] if run.created_at else '—')}</td>"
            f"<td>{_esc(run.spec.backend)}</td>"
            f"<td>{_esc(run.spec.model_id)}</td>"
            f"<td>{_esc(run.spec.quantization)}</td>"
            f"<td>{_esc(_device_of(run))}</td>"
            f'<td class="num">{_num(run.metrics.pp_tps)}</td>'
            f'<td class="num"{title}>{_num(run.metrics.tg_tps)}</td>'
            f'<td class="num">{_num(run.metrics.ttft_ms)}</td>'
            f"<td>{_cost_cell(run)}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>run_id</th><th>time</th><th>backend</th><th>model</th>"
        "<th>quantization</th><th>device</th>"
        '<th class="num">pp_tps</th><th class="num">tg_tps</th>'
        '<th class="num">ttft_ms</th><th>cost</th>'
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _trend_html(by_spec: dict[str, list[RunResult]]) -> str:
    rows = []
    for spec_id in sorted(by_spec):
        spec_runs = by_spec[spec_id]
        first, last = spec_runs[0], spec_runs[-1]
        f_tg = first.metrics.tg_tps
        l_tg = last.metrics.tg_tps
        if f_tg:
            delta = (l_tg - f_tg) / f_tg * 100.0
            delta_txt = f"{delta:+.1f}%"
            delta_cls = "trend-up" if delta >= 0 else "trend-down"
        else:
            delta_txt = "—"
            delta_cls = ""
        rows.append(
            "<tr>"
            f"<td><code>{_esc(spec_id)}</code></td>"
            f'<td class="num">{len(spec_runs)}</td>'
            f'<td class="num">{_num(f_tg, 3)}</td>'
            f'<td class="num">{_num(l_tg, 3)}</td>'
            f'<td class="num {delta_cls}">{delta_txt}</td>'
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>spec</th><th class=\"num\">runs</th>"
        '<th class="num">first tg_tps</th><th class="num">last tg_tps</th>'
        '<th class="num">delta%</th>'
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


_STYLE = """
:root { color-scheme: light; }
body {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 2rem auto; max-width: 1080px; padding: 0 1rem;
    color: #1f2937; background: #ffffff; line-height: 1.5;
}
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
.subtitle { color: #6b7280; margin-bottom: 1.5rem; }
.summary { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }
.stat {
    border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.75rem 1rem;
    min-width: 160px; background: #f9fafb;
}
.stat-value { display: block; font-size: 1.5rem; font-weight: 700; }
.stat-label { display: block; color: #6b7280; font-size: 0.85rem; }
.stat-detail { display: block; color: #374151; font-size: 0.8rem; overflow-wrap: anywhere; }
.section-title { margin: 2rem 0 0.5rem; font-size: 1.1rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; }
th { background: #f3f4f6; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85em; }
tr.regression { background: #fee2e2; }
.badge {
    background: #dc2626; color: #ffffff; border-radius: 4px;
    padding: 1px 6px; font-size: 0.72rem; font-weight: 600; margin-left: 6px;
}
.trend-up { color: #16a34a; font-weight: 600; }
.trend-down { color: #dc2626; font-weight: 600; }
.footer { margin-top: 2rem; color: #9ca3af; font-size: 0.8rem; }
"""


def render_html(runs: list[RunResult]) -> str:
    """Render ``runs`` into a complete, self-contained HTML document."""
    if not runs:
        return (
            "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>InferCI Dashboard</title><style>" + _STYLE + "</style></head>"
            "<body><h1>InferCI Dashboard</h1><p>no runs yet</p></body></html>"
        )

    by_spec = _group_by_spec(runs)
    regressions = _detect_regressions(by_spec)
    n_regressions = sum(1 for r in runs if r.run_id in regressions)

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>InferCI Dashboard</title>",
        "<style>" + _STYLE + "</style>",
        "</head>",
        "<body>",
        "<h1>InferCI Dashboard</h1>",
        '<p class="subtitle">Neutral, reproducible inference performance &amp; cost '
        "regression CI — rendered from the local ledger.</p>",
        _summary_html(runs),
    ]

    if n_regressions:
        parts.append(
            '<p><span class="badge">' + str(n_regressions)
            + " regression(s)</span></p>"
        )

    parts.append('<h2 class="section-title">Runs</h2>')
    parts.append(_table_html(runs, regressions))
    parts.append('<h2 class="section-title">Per-spec trend</h2>')
    parts.append(_trend_html(by_spec))
    parts.append('<p class="footer">Generated by inferci.dashboard — stdlib only.</p>')
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m inferci.dashboard",
        description="Render the InferCI run ledger into a static HTML dashboard.",
    )
    parser.add_argument("--db", default="inferci.db", help="SQLite ledger path")
    parser.add_argument("--out", default="dashboard.html", help="output HTML path")
    parser.add_argument("--limit", type=int, default=1000, help="max runs to render")
    parser.add_argument("--backend", default=None, help="filter by backend")
    parser.add_argument("--model", default=None, help="filter by model_id")
    args = parser.parse_args(argv)

    store = Store(args.db)
    runs = store.list(limit=args.limit, backend=args.backend, model_id=args.model)
    html_text = render_html(runs)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html_text)

    print(
        f"[inferci.dashboard] wrote {len(runs)} run(s) to {args.out} "
        f"({len(html_text)} bytes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
