"""RecallGate: long-context quality gating, measured as quality-per-dollar.

Motivation
----------
KV-cache compression and sparse attention save memory, but "did the quality
survive the compression, and what did each dollar buy" used to be unobservable.
RecallGate turns that into a deployable gate:

    quality_per_dollar = quality / cost_per_1m_output_usd

For every context/compression budget it (1) runs a deterministic
Needle-In-A-Haystack (NIAH) probe against any OpenAI-compatible
``/v1/completions`` endpoint, (2) derives $/1M output tokens from the *measured*
throughput of that very call via ``cost.compute_cost``, and (3) gates each budget
against the full-context baseline: a relative quality drop beyond ``threshold``
(default 10%) is FAIL.

Stdlib only. The HTTP style mirrors ``runners.serving.OpenAIServingRunner``
(``http.client`` + ``urllib.parse``), and the deterministic single-token haystack
filler reuses ``runners.serving._make_prompt``.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

from .cost import compute_cost
from .runners.serving import _completions_path, _make_prompt


def quality_per_dollar(quality: float, cost_per_1m_output_usd: float) -> float:
    """Quality per dollar of output generation.

    ``quality`` must be normalized to ``[0, 1]`` (a recall score); ``cost_per_1m_output_usd``
    is the price of 1M output tokens in USD (``CostResult.price_per_output_1m``).

    Returns ``quality / cost_per_1m_output_usd``. A non-positive cost means the
    run is effectively free (local hardware, no cloud price), where the ratio is
    undefined; we return ``math.inf`` to mean "unbounded quality-per-dollar" and
    call it out so callers can detect it with ``math.isinf``.

    Units note: this is quality-points per (USD / 1M output tokens) — kept
    against the 1M-token price the cost model emits so budgets compare
    apples-to-apples. A literal "per dollar" figure is this value scaled by 1e6.
    """
    if cost_per_1m_output_usd <= 0.0:
        return math.inf
    return float(quality) / float(cost_per_1m_output_usd)


# ---------------------------------------------------------------------------
# Needle-In-A-Haystack (reference probe)
# ---------------------------------------------------------------------------

_NEEDLE_ADJECTIVES = (
    "crimson", "turquoise", "obsidian", "scarlet", "violet", "emerald",
    "cobalt", "golden",
)
_NEEDLE_NOUNS = (
    "falcon", "lantern", "orchid", "compass", "lighthouse", "thunder",
    "harbor", "comet",
)

_QUESTION = "Question: What is the passphrase mentioned in the document?\nAnswer:"


def make_needle(salt: int = 0) -> str:
    """Return a deterministic, unique passphrase derived from ``salt``.

    Deterministic so two runs of the same ``salt`` probe identical text (and two
    different salts probe different text, defeating prompt caching).
    """
    adj = _NEEDLE_ADJECTIVES[salt % len(_NEEDLE_ADJECTIVES)]
    noun = _NEEDLE_NOUNS[(salt // len(_NEEDLE_ADJECTIVES)) % len(_NEEDLE_NOUNS)]
    num = (salt * 7919 + 104729) % 10000
    return f"{adj}-{noun}-{num:04d}".upper()


def build_needle_prompt(
    context_tokens: int,
    *,
    needle: Optional[str] = None,
    position: float = 0.5,
    salt: int = 0,
) -> str:
    """Build a deterministic NIAH prompt of ~``context_tokens`` haystack tokens.

    The haystack is a run of filler tokens (each filler word is one BPE token,
    via ``serving._make_prompt``). The unique ``needle`` sentence is buried at
    ``position`` (a fraction of the haystack in ``[0, 1]``), and a question is
    appended asking for the passphrase.
    """
    n = max(1, int(context_tokens))
    pos = min(1.0, max(0.0, float(position)))
    needle = needle or make_needle(salt)
    prefix_tokens = int(round(n * pos))
    suffix_tokens = max(0, n - prefix_tokens)
    prefix = _make_prompt(prefix_tokens, salt=salt)
    suffix = _make_prompt(suffix_tokens, salt=salt + 1)
    document = f"{prefix}\n\nThe passphrase is {needle}.\n\n{suffix}"
    return (
        "You are given a document and a question. Answer the question using "
        "only the information in the document. If the answer is not in the "
        'document, respond with "not found".\n\n'
        f"<document>\n{document}\n</document>\n\n"
        f"{_QUESTION}"
    )


def _normalize(s: str) -> str:
    """Lowercase and strip every non-alphanumeric character (for recall match)."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def needle_in_answer(needle: str, answer: str) -> bool:
    """Recall: does the passphrase appear in the answer?

    Case- and punctuation-insensitive: the model may lower-case the passphrase
    or insert spaces where the hyphens were, and it still counts as a hit.
    """
    return _normalize(needle) in _normalize(answer)


@dataclass
class NiahResult:
    """One NIAH probe: recall outcome plus the measured throughput inputs."""
    quality: float = 0.0        # 0.0 (miss) or 1.0 (hit); normalized 0..1
    recall: bool = False
    answer: str = ""
    needle: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    pp_tps: float = 0.0
    tg_tps: float = 0.0
    prompt: str = ""


def _open_connection(base_url: str, timeout: float):
    """Return ``(http.client.HTTPConnection, completions_path)`` for ``base_url``."""
    parts = urllib.parse.urlsplit(base_url)
    scheme = (parts.scheme or "http").lower()
    host = parts.hostname
    port = parts.port
    if not host:
        raise ValueError(f"invalid base_url: {base_url!r}")
    if port is None:
        port = 443 if scheme == "https" else 80
    if scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            host, port, timeout=timeout
        )
    elif scheme == "http":
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    else:
        raise ValueError(f"unsupported URL scheme: {scheme!r}")
    return conn, _completions_path(base_url)


def _stream_completion(
    base_url: str, payload: dict, api_key: Optional[str], timeout: float
):
    """POST ``payload`` to ``/v1/completions`` and read the SSE stream.

    Returns ``(text, prompt_tokens, generated_tokens, ttft_ms, decode_ms, total_ms)``.
    """
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    conn, path = _open_connection(base_url, timeout)
    text_parts: list[str] = []
    token_times: list[float] = []
    usage: dict = {}
    first_token_s = 0.0
    last_token_s = 0.0
    t_start = time.perf_counter()
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            raw = resp.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"completion returned HTTP {resp.status}: {raw[:500]}"
            )
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(b"\n\n"):
                block = buf[:-2]
                buf = b""
                for line in block.split(b"\n"):
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:"):].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(data.decode("utf-8", "replace"))
                    except json.JSONDecodeError:
                        continue
                    arrival = time.perf_counter() - t_start
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choice = (obj.get("choices") or [{}])[0]
                    text = choice.get("text") or ""
                    if not text:
                        continue
                    text_parts.append(text)
                    if not token_times:
                        first_token_s = arrival
                    last_token_s = arrival
                    token_times.append(arrival)
    finally:
        conn.close()

    text = "".join(text_parts)
    if not text:
        raise RuntimeError("completion stream ended with no tokens (empty completion?)")

    generated_tokens = int(usage.get("completion_tokens") or len(token_times))
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    ttft_ms = first_token_s * 1000.0
    decode_ms = max(0.0, last_token_s - first_token_s) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0
    return text, prompt_tokens, generated_tokens, ttft_ms, decode_ms, total_ms


class NeedleInHaystack:
    """Reference NIAH probe against any OpenAI-compatible ``/v1/completions``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key
        self.timeout = timeout

    def evaluate(
        self,
        context_tokens: int,
        *,
        needle: Optional[str] = None,
        position: float = 0.5,
        salt: int = 0,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> NiahResult:
        """Run one probe and return the recall score plus measured throughput."""
        if not self.base_url:
            raise ValueError("no base_url given to NeedleInHaystack")
        if not self.model:
            raise ValueError("no model given to NeedleInHaystack")
        needle = needle or make_needle(salt)
        prompt = build_needle_prompt(
            context_tokens, needle=needle, position=position, salt=salt
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": float(temperature),
            "seed": int(seed),
        }
        (text, prompt_tokens, generated_tokens, ttft_ms, decode_ms, total_ms) = \
            _stream_completion(self.base_url, payload, self.api_key, self.timeout)

        recall = needle_in_answer(needle, text)
        pp_tps = (
            prompt_tokens / (ttft_ms / 1000.0)
            if ttft_ms > 0 and prompt_tokens > 0 else 0.0
        )
        tg_tps = (
            generated_tokens / (decode_ms / 1000.0)
            if decode_ms > 0 and generated_tokens > 0 else 0.0
        )
        return NiahResult(
            quality=1.0 if recall else 0.0,
            recall=recall,
            answer=text,
            needle=needle,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
            pp_tps=pp_tps,
            tg_tps=tg_tps,
            prompt=prompt,
        )

    def score(self, context_tokens: int, **kwargs) -> float:
        """Convenience: run one probe and return just the 0/1 recall score."""
        return self.evaluate(context_tokens, **kwargs).quality


# ---------------------------------------------------------------------------
# RecallGate (quality-per-dollar deployment gate)
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    """RecallGate knobs. Defaults are safe and deterministic."""
    instance_type: Optional[str] = None      # cloud instance for compute_cost
    instance_hourly: Optional[float] = None  # or a raw $/hour override
    threshold: float = 0.10                  # max relative quality drop vs baseline
    baseline_budget: Optional[int] = None    # which budget is the baseline (default: max)
    needle: Optional[str] = None
    needle_position: float = 0.5             # where the needle sits in the haystack
    max_tokens: int = 64
    temperature: float = 0.0
    seed: int = 42
    repeats: int = 1
    budgets: Optional[list[int]] = None      # default budgets when evaluate() is called bare
    timeout: float = 600.0


@dataclass
class GateResult:
    """Verdict for one context/compression budget."""
    budget: int
    quality: float
    cost_per_1m_output: float
    quality_per_dollar: float
    verdict: str                       # "PASS" | "FAIL"
    relative_drop: float = 0.0
    pp_tps: float = 0.0
    tg_tps: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "quality": self.quality,
            "cost_per_1m_output": self.cost_per_1m_output,
            "quality_per_dollar": self.quality_per_dollar,
            "verdict": self.verdict,
            "relative_drop": self.relative_drop,
            "pp_tps": self.pp_tps,
            "tg_tps": self.tg_tps,
            "note": self.note,
        }


def judge(
    quality: float, baseline_quality: float, threshold: float = 0.10
) -> tuple[str, float, str]:
    """Verdict for one budget against the full-context baseline.

    Returns ``(verdict, relative_drop, note)``. ``relative_drop`` is
    ``(baseline_quality - quality) / baseline_quality``. A drop *strictly
    greater* than ``threshold`` is FAIL; everything else (no drop, a small drop
    within tolerance, an improvement, or a zero baseline) is PASS.
    """
    if baseline_quality <= 0.0:
        return ("PASS", 0.0, "baseline quality is zero; relative drop undefined")
    drop = (baseline_quality - quality) / baseline_quality
    if drop > threshold:
        return (
            "FAIL", drop,
            f"quality dropped {drop * 100:.2f}% (threshold {threshold * 100:.1f}%)",
        )
    return (
        "PASS", drop,
        f"drop {drop * 100:.2f}% within threshold {threshold * 100:.1f}%",
    )


class RecallGate:
    """Evaluate quality-per-dollar across context/compression budgets."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        config: Optional[GateConfig] = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key
        self.config = config or GateConfig()

    def evaluate(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        budgets: Optional[list[int]] = None,
        config: Optional[GateConfig] = None,
    ) -> list[GateResult]:
        """Run the gate across ``budgets`` and return one GateResult per budget.

        Each budget is a context length (in haystack tokens). The baseline is the
        largest budget (or ``config.baseline_budget``); every other budget is
        judged on how far its quality falls below the baseline's.
        """
        base_url = (base_url or self.base_url).rstrip("/")
        model = model or self.model
        cfg = config or self.config
        budgets = budgets if budgets is not None else (cfg.budgets or [])
        if not base_url:
            raise ValueError(
                "no base_url: pass it to RecallGate(base_url=...) or evaluate(base_url=...)"
            )
        if not model:
            raise ValueError(
                "no model: pass it to RecallGate(model=...) or evaluate(model=...)"
            )
        if not budgets:
            raise ValueError(
                "no budgets: pass budgets=[...] to evaluate() or set config.budgets"
            )

        budgets = sorted({int(b) for b in budgets})
        baseline_budget = (
            int(cfg.baseline_budget) if cfg.baseline_budget is not None else budgets[-1]
        )
        if baseline_budget not in budgets:
            raise ValueError(f"baseline_budget {baseline_budget} not in budgets {budgets}")

        niah = NeedleInHaystack(
            base_url, model, api_key=self.api_key, timeout=cfg.timeout
        )

        # Average over `repeats` (same needle across budgets, so the comparison
        # is fair: only the context length changes).
        per_budget: dict[int, tuple[float, float, float]] = {}
        for budget in budgets:
            qualities: list[float] = []
            pp_vals: list[float] = []
            tg_vals: list[float] = []
            for rep in range(max(1, int(cfg.repeats))):
                r = niah.evaluate(
                    budget,
                    needle=cfg.needle,
                    position=cfg.needle_position,
                    salt=rep,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    seed=cfg.seed,
                )
                qualities.append(r.quality)
                pp_vals.append(r.pp_tps)
                tg_vals.append(r.tg_tps)
            per_budget[budget] = (
                sum(qualities) / len(qualities),
                sum(pp_vals) / len(pp_vals),
                sum(tg_vals) / len(tg_vals),
            )

        baseline_quality = per_budget[baseline_budget][0]
        results: list[GateResult] = []
        for budget in budgets:
            quality, pp_tps, tg_tps = per_budget[budget]
            cost = compute_cost(
                pp_tps, tg_tps,
                instance_type=cfg.instance_type,
                instance_hourly=cfg.instance_hourly,
            )
            cpm = cost.price_per_output_1m
            qpd = quality_per_dollar(quality, cpm)
            if budget == baseline_budget:
                verdict, drop, note = "PASS", 0.0, "baseline (full context)"
            else:
                verdict, drop, note = judge(quality, baseline_quality, cfg.threshold)
            results.append(GateResult(
                budget=budget,
                quality=quality,
                cost_per_1m_output=cpm,
                quality_per_dollar=qpd,
                verdict=verdict,
                relative_drop=drop,
                pp_tps=pp_tps,
                tg_tps=tg_tps,
                note=note,
            ))
        return results


# ---------------------------------------------------------------------------
# CLI: python -m inferci.quality
# ---------------------------------------------------------------------------

def _fmt_table(results: list[GateResult]) -> str:
    header = ("budget", "quality", "cost_1m_out", "q_per_dollar", "verdict")
    rows = [header]
    for r in results:
        qpd = "inf" if math.isinf(r.quality_per_dollar) else f"{r.quality_per_dollar:.6f}"
        rows.append((
            str(r.budget),
            f"{r.quality:.3f}",
            f"{r.cost_per_1m_output:.6f}",
            qpd,
            r.verdict,
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(f"{cell:<{widths[i]}}" for i, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join(lines)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if math.isinf(obj) else "nan"
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inferci.quality",
        description="RecallGate: quality-per-dollar gate across context budgets.",
    )
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    p.add_argument("--model", required=True, help="model id/alias on the server")
    p.add_argument("--budgets", required=True,
                   help="comma-separated context lengths, e.g. 512,1024,2048")
    p.add_argument("--instance", default=None,
                   help="cloud instance type for cost (e.g. cpu.m7i.xlarge)")
    p.add_argument("--instance-hourly", type=float, default=None,
                   help="override $/hour directly (alternative to --instance)")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="max relative quality drop vs baseline before FAIL")
    p.add_argument("--baseline-budget", type=int, default=None,
                   help="which budget is the baseline (default: largest)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--position", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api-key", default=None)
    p.add_argument("--json", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    budgets = [
        int(x) for x in str(args.budgets).replace(" ", "").split(",") if x
    ]
    config = GateConfig(
        instance_type=args.instance,
        instance_hourly=args.instance_hourly,
        threshold=args.threshold,
        baseline_budget=args.baseline_budget,
        needle_position=args.position,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        repeats=args.repeats,
    )
    gate = RecallGate(args.base_url, args.model, api_key=args.api_key, config=config)
    results = gate.evaluate(budgets=budgets)
    if args.json:
        print(json.dumps(
            [_jsonable(r.to_dict()) for r in results],
            ensure_ascii=False, indent=2,
        ))
    else:
        print(_fmt_table(results))
    return 0


__all__ = [
    "quality_per_dollar",
    "make_needle",
    "build_needle_prompt",
    "needle_in_answer",
    "NiahResult",
    "NeedleInHaystack",
    "GateConfig",
    "GateResult",
    "judge",
    "RecallGate",
]


if __name__ == "__main__":
    raise SystemExit(main())
