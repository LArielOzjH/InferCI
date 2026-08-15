"""InferCI command-line interface (stdlib argparse, zero deps).

Examples:
  inferci run --model-file models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
              --model-id Qwen2.5-0.5B-Instruct --quantization Q4_K_M --device cpu
  inferci list
  inferci diff <base_run_id> <candidate_run_id>
  inferci report
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .cost import compute_cost
from .regression import any_regression, comparability_issues, compare_runs
from .runners import get_runner, available_runners
from .schema import BenchmarkSpec, Sampling, capture_local_environment
from .store import Store


def _fmt_ts(iso: str) -> str:
    return iso[:19] if iso else ""


def _build_spec(args) -> BenchmarkSpec:
    extra = {"device": args.device, "threads": args.threads}
    for kv in getattr(args, "set", None) or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        extra[k] = v
    return BenchmarkSpec(
        model_id=args.model_id,
        model_file=args.model_file,
        backend=args.backend,
        quantization=args.quantization,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        repeats=args.repeats,
        warmup_repeats=args.warmup,
        batch=args.batch,
        sampling=Sampling(temperature=args.temperature, top_p=args.top_p, seed=args.seed),
        extra=extra,
    )


def cmd_run(args) -> int:
    spec = _build_spec(args)
    spec.id = spec.canonical_id()
    if args.backend != "mock" and not spec.model_file:
        print("error: --model-file is required for this backend", file=sys.stderr)
        return 2
    runner = get_runner(args.backend)
    env = runner.capture_environment()
    print(f"[inferci] running {spec.canonical_id()} on {args.device} ...", file=sys.stderr)
    result = runner.run(spec, env)
    if args.instance:
        result.cost = compute_cost(
            result.metrics.pp_tps, result.metrics.tg_tps, instance_type=args.instance
        )
    store = Store(args.db)
    store.insert(result)
    if args.json:
        print(result.to_json())
    else:
        m = result.metrics
        print(json.dumps({
            "run_id": result.run_id,
            "spec": spec.canonical_id(),
            "pp_tps": round(m.pp_tps, 3),
            "tg_tps": round(m.tg_tps, 3),
            "ttft_ms_derived": round(m.ttft_ms, 2),
            "itl_ms_derived": round(m.itl.mean_ms, 2),
            "backend_version": result.environment.backend_version,
        }, ensure_ascii=False, indent=2))
    return 0


def cmd_list(args) -> int:
    store = Store(args.db)
    runs = store.list(limit=args.limit, backend=args.backend, model_id=args.model)
    print(f"{'run_id':<37}{'created':<20}{'backend':<14}{'model':<22}{'q':<10}{'tg_tps':>9}{'pp_tps':>10}")
    for r in runs:
        print(f"{r.run_id:<37}{_fmt_ts(r.created_at):<20}{r.spec.backend:<14}"
              f"{r.spec.model_id:<22}{r.spec.quantization:<10}"
              f"{r.metrics.tg_tps:>9.2f}{r.metrics.pp_tps:>10.2f}")
    return 0


def cmd_diff(args) -> int:
    store = Store(args.db)
    base = store.resolve(args.base)
    cand = store.resolve(args.candidate)
    if not base or not cand:
        print("run_id not found (or ambiguous prefix)", file=sys.stderr)
        return 1
    if base.spec.canonical_id() != cand.spec.canonical_id():
        print("WARNING: spec ids differ; comparison may be apples-to-oranges:", file=sys.stderr)
        print(f"  base: {base.spec.canonical_id()}", file=sys.stderr)
        print(f"  cand: {cand.spec.canonical_id()}", file=sys.stderr)
    issues = comparability_issues(base, cand)
    if issues:
        print("ERROR: runs are not fairly comparable:", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        if not args.force:
            print("refusing to compare (pass --force to override)", file=sys.stderr)
            return 2
    findings = compare_runs(base, cand)
    print(f"base={args.base} ({base.environment.backend_version})  ->  cand={args.candidate} ({cand.environment.backend_version})")
    for f in findings:
        print("  " + f.fmt())
    return 1 if any_regression(findings) else 0


def cmd_dashboard(args) -> int:
    from .dashboard import render_html
    store = Store(args.db)
    runs = store.list(limit=None)
    html = render_html(runs)
    out = args.out or "dashboard.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({len(runs)} runs)")
    return 0


def cmd_report(args) -> int:
    store = Store(args.db)
    runs = store.list(limit=None)
    if not runs:
        print("no runs yet")
        return 0
    print(f"total runs: {len(runs)}")
    by_spec: dict[str, list] = {}
    for r in runs:
        by_spec.setdefault(r.spec.canonical_id(), []).append(r)
    print(f"{'spec':<60}{'n':>4}{'first_tg_tps':>13}{'last_tg_tps':>13}{'delta%':>9}")
    for spec_id, rs in sorted(by_spec.items()):
        rs_sorted = sorted(rs, key=lambda x: x.created_at)
        first, last = rs_sorted[0], rs_sorted[-1]
        d = ((last.metrics.tg_tps - first.metrics.tg_tps) / first.metrics.tg_tps * 100
             if first.metrics.tg_tps else 0.0)
        print(f"{spec_id:<60}{len(rs):>4}{first.metrics.tg_tps:>13.2f}{last.metrics.tg_tps:>13.2f}{d:>8.1f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inferci", description="Inference performance & cost regression CI")
    p.add_argument("--version", action="version", version=f"inferci {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark")
    r.add_argument("--backend", default="llama_cpp", choices=available_runners())
    r.add_argument("--model-file", default="", help="model path (not needed for --backend mock)")
    r.add_argument("--model-id", required=True, help="canonical model name (used in the spec id)")
    r.add_argument("--quantization", default="")
    r.add_argument("--prompt-tokens", type=int, default=512)
    r.add_argument("--gen-tokens", type=int, default=128)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--batch", type=int, default=1)
    r.add_argument("--device", default="cpu", choices=["cpu", "metal", "cuda", "gpu", "all"])
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="set a spec.extra key (repeatable), e.g. --set slowdown=0.9")
    r.add_argument("--instance", default=None, help="cloud instance type for cost (e.g. gpu.t4.g4dn.xlarge)")
    r.add_argument("--db", default="inferci.db")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("list", help="list runs")
    l.add_argument("--db", default="inferci.db")
    l.add_argument("--limit", type=int, default=50)
    l.add_argument("--backend", default=None)
    l.add_argument("--model", default=None)
    l.set_defaults(func=cmd_list)

    d = sub.add_parser("diff", help="compare two runs")
    d.add_argument("base")
    d.add_argument("candidate")
    d.add_argument("--db", default="inferci.db")
    d.add_argument("--force", action="store_true", help="compare even if runs are not fairly comparable")
    d.set_defaults(func=cmd_diff)

    rep = sub.add_parser("report", help="summary over full history")
    rep.add_argument("--db", default="inferci.db")
    rep.set_defaults(func=cmd_report)

    dash = sub.add_parser("dashboard", help="render a static HTML dashboard from the ledger")
    dash.add_argument("--db", default="inferci.db")
    dash.add_argument("--out", default=None)
    dash.set_defaults(func=cmd_dashboard)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
