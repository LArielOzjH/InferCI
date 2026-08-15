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
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
# Pluggable eval protocol + registry
# ---------------------------------------------------------------------------


class Eval(ABC):
    """Pluggable quality eval: one probe -> one ``[0, 1]`` quality score.

    A concrete eval exposes a stable ``name`` (the registry key) and implements
    ``score(base_url, model, config, **kw)``, which runs the eval against the
    given OpenAI-compatible endpoint and returns a quality score normalized to
    ``[0, 1]``. The registry lets callers swap probes without touching the gate.
    """

    name: str = ""

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
    ):
        # Standard constructor convention used by ``RecallGate``; concrete evals
        # may extend it (e.g. NeedleInHaystack adds ``chat``).
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    def score(
        self,
        base_url: str,
        model: str,
        config: GateConfig | None = None,
        **kw,
    ) -> float:
        """Run one probe and return a ``[0, 1]`` quality score."""
        raise NotImplementedError


EVAL_REGISTRY: dict[str, type] = {}


def register_eval(name: str, cls: type | None = None):
    """Register an eval class under ``name``.

    Usable either as a plain function ``register_eval("niah", NeedleInHaystack)``
    or as a class decorator ``@register_eval("niah")``.
    """

    def _register(eval_cls: type) -> type:
        EVAL_REGISTRY[name] = eval_cls
        return eval_cls

    if cls is None:
        return _register
    return _register(cls)


# ---------------------------------------------------------------------------
# Needle-In-A-Haystack (reference probe)
# ---------------------------------------------------------------------------

_NEEDLE_ADJECTIVES = (
    "crimson",
    "turquoise",
    "obsidian",
    "scarlet",
    "violet",
    "emerald",
    "cobalt",
    "golden",
)
_NEEDLE_NOUNS = (
    "falcon",
    "lantern",
    "orchid",
    "compass",
    "lighthouse",
    "thunder",
    "harbor",
    "comet",
)

_QUESTION = "Question: What is the passphrase mentioned in the document?\nAnswer:"

_SYSTEM_INSTRUCTION = (
    "You are given a document and a question. Answer the question using "
    "only the information in the document. If the answer is not in the "
    'document, respond with "not found".'
)


def make_needle(salt: int = 0) -> str:
    """Return a deterministic, unique passphrase derived from ``salt``.

    Deterministic so two runs of the same ``salt`` probe identical text (and two
    different salts probe different text, defeating prompt caching).
    """
    adj = _NEEDLE_ADJECTIVES[salt % len(_NEEDLE_ADJECTIVES)]
    noun = _NEEDLE_NOUNS[(salt // len(_NEEDLE_ADJECTIVES)) % len(_NEEDLE_NOUNS)]
    num = (salt * 7919 + 104729) % 10000
    return f"{adj}-{noun}-{num:04d}".upper()


def _build_document(
    context_tokens: int,
    *,
    needle: str | None = None,
    position: float = 0.5,
    salt: int = 0,
) -> str:
    """Build the deterministic ``<document>...</document>`` haystack body."""
    n = max(1, int(context_tokens))
    pos = min(1.0, max(0.0, float(position)))
    needle = needle or make_needle(salt)
    prefix_tokens = round(n * pos)
    suffix_tokens = max(0, n - prefix_tokens)
    prefix = _make_prompt(prefix_tokens, salt=salt)
    suffix = _make_prompt(suffix_tokens, salt=salt + 1)
    return f"{prefix}\n\nThe passphrase is {needle}.\n\n{suffix}"


def build_needle_prompt(
    context_tokens: int,
    *,
    needle: str | None = None,
    position: float = 0.5,
    salt: int = 0,
) -> str:
    """Build a deterministic NIAH prompt of ~``context_tokens`` haystack tokens.

    The haystack is a run of filler tokens (each filler word is one BPE token,
    via ``serving._make_prompt``). The unique ``needle`` sentence is buried at
    ``position`` (a fraction of the haystack in ``[0, 1]``), and a question is
    appended asking for the passphrase.
    """
    document = _build_document(context_tokens, needle=needle, position=position, salt=salt)
    return f"{_SYSTEM_INSTRUCTION}\n\n<document>\n{document}\n</document>\n\n{_QUESTION}"


def build_needle_messages(
    context_tokens: int,
    *,
    needle: str | None = None,
    position: float = 0.5,
    salt: int = 0,
) -> list[dict]:
    """Build the same NIAH probe as a chat ``messages`` list.

    Splits the instruction into a ``system`` message and the document+question
    into a ``user`` message, for use against ``/v1/chat/completions``.
    """
    document = _build_document(context_tokens, needle=needle, position=position, salt=salt)
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"<document>\n{document}\n</document>\n\n{_QUESTION}"},
    ]


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

    quality: float = 0.0  # 0.0 (miss) or 1.0 (hit); normalized 0..1
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


def _chat_completions_path(base_url: str) -> str:
    """Return the request path for ``/v1/chat/completions`` given a base URL.

    Mirrors ``serving._completions_path`` but targets the chat endpoint.
    """
    parts = urllib.parse.urlsplit(base_url)
    base = (parts.path or "").rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _open_connection(base_url: str, timeout: float, chat: bool = False):
    """Return ``(http.client.HTTPConnection, request_path)`` for ``base_url``.

    The path is ``/v1/completions`` (or ``/v1/chat/completions`` when ``chat``
    is true).
    """
    parts = urllib.parse.urlsplit(base_url)
    scheme = (parts.scheme or "http").lower()
    host = parts.hostname
    port = parts.port
    if not host:
        raise ValueError(f"invalid base_url: {base_url!r}")
    if port is None:
        port = 443 if scheme == "https" else 80
    if scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(host, port, timeout=timeout)
    elif scheme == "http":
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    else:
        raise ValueError(f"unsupported URL scheme: {scheme!r}")
    path = _chat_completions_path(base_url) if chat else _completions_path(base_url)
    return conn, path


def _stream_completion(
    base_url: str,
    payload: dict,
    api_key: str | None,
    timeout: float,
    chat: bool = False,
):
    """POST ``payload`` to the completions (or chat) endpoint and read the SSE stream.

    ``chat=True`` targets ``/v1/chat/completions`` and reads token text from
    ``choices[].delta.content`` (falling back to ``message.content``); otherwise
    ``/v1/completions`` is used and text comes from ``choices[].text``.

    Returns ``(text, prompt_tokens, generated_tokens, ttft_ms, decode_ms, total_ms)``.
    """
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    conn, path = _open_connection(base_url, timeout, chat=chat)
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
            raise RuntimeError(f"completion returned HTTP {resp.status}: {raw[:500]}")
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
                    data = line[len(b"data:") :].strip()
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
                    if chat:
                        msg = choice.get("delta") or choice.get("message") or {}
                        text = msg.get("content") or ""
                    else:
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


class NeedleInHaystack(Eval):
    """Reference NIAH probe against any OpenAI-compatible endpoint.

    Talks to ``/v1/completions`` by default; pass ``chat=True`` (or ``chat``
    through ``evaluate``/``build_payload``) to use ``/v1/chat/completions`` with
    a ``messages`` payload instead of a raw ``prompt``.
    """

    name = "niah"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        chat: bool = False,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key
        self.timeout = timeout
        self.chat = bool(chat)

    def build_payload(
        self,
        context_tokens: int,
        *,
        needle: str | None = None,
        position: float = 0.5,
        salt: int = 0,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 42,
        chat: bool | None = None,
    ) -> dict:
        """Build the request payload for one probe (``prompt`` or ``messages``)."""
        chat = self.chat if chat is None else bool(chat)
        common = {
            "model": self.model,
            "max_tokens": int(max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": float(temperature),
            "seed": int(seed),
        }
        if chat:
            return {
                **common,
                "messages": build_needle_messages(
                    context_tokens, needle=needle, position=position, salt=salt
                ),
            }
        return {
            **common,
            "prompt": build_needle_prompt(
                context_tokens, needle=needle, position=position, salt=salt
            ),
        }

    def evaluate(
        self,
        context_tokens: int,
        *,
        needle: str | None = None,
        position: float = 0.5,
        salt: int = 0,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 42,
        chat: bool | None = None,
    ) -> NiahResult:
        """Run one probe and return the recall score plus measured throughput."""
        if not self.base_url:
            raise ValueError("no base_url given to NeedleInHaystack")
        if not self.model:
            raise ValueError("no model given to NeedleInHaystack")
        chat = self.chat if chat is None else bool(chat)
        needle = needle or make_needle(salt)
        payload = self.build_payload(
            context_tokens,
            needle=needle,
            position=position,
            salt=salt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            chat=chat,
        )
        (text, prompt_tokens, generated_tokens, ttft_ms, decode_ms, total_ms) = _stream_completion(
            self.base_url, payload, self.api_key, self.timeout, chat=chat
        )

        recall = needle_in_answer(needle, text)
        pp_tps = prompt_tokens / (ttft_ms / 1000.0) if ttft_ms > 0 and prompt_tokens > 0 else 0.0
        tg_tps = (
            generated_tokens / (decode_ms / 1000.0)
            if decode_ms > 0 and generated_tokens > 0
            else 0.0
        )
        prompt_text = payload.get("prompt") or json.dumps(
            payload.get("messages", []), ensure_ascii=False
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
            prompt=prompt_text,
        )

    def score(
        self,
        base_url: str,
        model: str | None = None,
        config: GateConfig | None = None,
        **kw,
    ) -> float:
        """Protocol entry: run one probe and return the ``[0, 1]`` quality score.

        Backward compatible with the legacy ``score(context_tokens, **kwargs)``
        convenience call: a non-string first argument is treated as the context
        length for a probe against this instance's own ``base_url``/``model``.
        """
        if not isinstance(base_url, str):
            return self.evaluate(int(base_url), **kw).quality
        chat = bool(kw.pop("chat", getattr(config, "chat", False)))
        ev = NeedleInHaystack(
            base_url,
            model or self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            chat=chat,
        )
        return ev.evaluate(**kw).quality


register_eval("niah", NeedleInHaystack)


# ---------------------------------------------------------------------------
# RecallGate (quality-per-dollar deployment gate)
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    """RecallGate knobs. Defaults are safe and deterministic."""

    instance_type: str | None = None  # cloud instance for compute_cost
    instance_hourly: float | None = None  # or a raw $/hour override
    threshold: float = 0.10  # max relative quality drop vs baseline
    baseline_budget: int | None = None  # which budget is the baseline (default: max)
    needle: str | None = None
    needle_position: float = 0.5  # where the needle sits in the haystack
    max_tokens: int = 64
    temperature: float = 0.0
    seed: int = 42
    chat: bool = False  # probe via /v1/chat/completions messages
    repeats: int = 1
    budgets: list[int] | None = None  # default budgets when evaluate() is called bare
    timeout: float = 600.0


@dataclass
class GateResult:
    """Verdict for one context/compression budget."""

    budget: int
    quality: float
    cost_per_1m_output: float
    quality_per_dollar: float
    verdict: str  # "PASS" | "FAIL"
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
            "FAIL",
            drop,
            f"quality dropped {drop * 100:.2f}% (threshold {threshold * 100:.1f}%)",
        )
    return (
        "PASS",
        drop,
        f"drop {drop * 100:.2f}% within threshold {threshold * 100:.1f}%",
    )


def _resolve_eval(eval_name) -> type:
    """Resolve ``eval`` (a registry name, or an eval class) to an eval class."""
    if isinstance(eval_name, type):
        return eval_name
    try:
        return EVAL_REGISTRY[str(eval_name)]
    except KeyError:
        raise ValueError(
            f"unknown eval {eval_name!r}; registered: {sorted(EVAL_REGISTRY)}"
        ) from None


class RecallGate:
    """Evaluate quality-per-dollar across context/compression budgets."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        config: GateConfig | None = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key
        self.config = config or GateConfig()

    def evaluate(
        self,
        base_url: str | None = None,
        model: str | None = None,
        budgets: list[int] | None = None,
        config: GateConfig | None = None,
        eval: str = "niah",
    ) -> list[GateResult]:
        """Run the gate across ``budgets`` and return one GateResult per budget.

        Each budget is a context length (in haystack tokens). The baseline is the
        largest budget (or ``config.baseline_budget``); every other budget is
        judged on how far its quality falls below the baseline's.

        ``eval`` selects the probe class from :data:`EVAL_REGISTRY` (default
        ``"niah"``); it may also be an ``Eval`` class directly.
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
            raise ValueError("no model: pass it to RecallGate(model=...) or evaluate(model=...)")
        if not budgets:
            raise ValueError("no budgets: pass budgets=[...] to evaluate() or set config.budgets")

        budgets = sorted({int(b) for b in budgets})
        baseline_budget = (
            int(cfg.baseline_budget) if cfg.baseline_budget is not None else budgets[-1]
        )
        if baseline_budget not in budgets:
            raise ValueError(f"baseline_budget {baseline_budget} not in budgets {budgets}")

        eval_cls = _resolve_eval(eval)
        ev = eval_cls(base_url, model, api_key=self.api_key, timeout=cfg.timeout)

        # Average over `repeats` (same needle across budgets, so the comparison
        # is fair: only the context length changes). NeedleInHaystack is the
        # reference eval and also reports measured throughput; a generic eval
        # only exposes a quality score, so its throughput is recorded as 0
        # (which the cost model prices as local/free).
        per_budget: dict[int, tuple[float, float, float]] = {}
        for budget in budgets:
            qualities: list[float] = []
            pp_vals: list[float] = []
            tg_vals: list[float] = []
            for rep in range(max(1, int(cfg.repeats))):
                if isinstance(ev, NeedleInHaystack):
                    r = ev.evaluate(
                        budget,
                        needle=cfg.needle,
                        position=cfg.needle_position,
                        salt=rep,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        seed=cfg.seed,
                        chat=cfg.chat,
                    )
                    quality, pp, tg = r.quality, r.pp_tps, r.tg_tps
                else:
                    quality = float(
                        ev.score(
                            base_url,
                            model,
                            cfg,
                            context_tokens=budget,
                            needle=cfg.needle,
                            position=cfg.needle_position,
                            salt=rep,
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                            seed=cfg.seed,
                            chat=cfg.chat,
                        )
                    )
                    pp = tg = 0.0
                qualities.append(quality)
                pp_vals.append(pp)
                tg_vals.append(tg)
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
                pp_tps,
                tg_tps,
                instance_type=cfg.instance_type,
                instance_hourly=cfg.instance_hourly,
            )
            cpm = cost.price_per_output_1m
            qpd = quality_per_dollar(quality, cpm)
            if budget == baseline_budget:
                verdict, drop, note = "PASS", 0.0, "baseline (full context)"
            else:
                verdict, drop, note = judge(quality, baseline_quality, cfg.threshold)
            results.append(
                GateResult(
                    budget=budget,
                    quality=quality,
                    cost_per_1m_output=cpm,
                    quality_per_dollar=qpd,
                    verdict=verdict,
                    relative_drop=drop,
                    pp_tps=pp_tps,
                    tg_tps=tg_tps,
                    note=note,
                )
            )
        return results


# ---------------------------------------------------------------------------
# CLI: python -m inferci.quality
# ---------------------------------------------------------------------------


def _fmt_table(results: list[GateResult]) -> str:
    header = ("budget", "quality", "cost_1m_out", "q_per_dollar", "verdict")
    rows = [header]
    for r in results:
        qpd = "inf" if math.isinf(r.quality_per_dollar) else f"{r.quality_per_dollar:.6f}"
        rows.append(
            (
                str(r.budget),
                f"{r.quality:.3f}",
                f"{r.cost_per_1m_output:.6f}",
                qpd,
                r.verdict,
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["  ".join(f"{cell:<{widths[i]}}" for i, cell in enumerate(row)) for row in rows]
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
    p.add_argument(
        "--budgets", required=True, help="comma-separated context lengths, e.g. 512,1024,2048"
    )
    p.add_argument(
        "--instance", default=None, help="cloud instance type for cost (e.g. cpu.m7i.xlarge)"
    )
    p.add_argument(
        "--instance-hourly",
        type=float,
        default=None,
        help="override $/hour directly (alternative to --instance)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="max relative quality drop vs baseline before FAIL",
    )
    p.add_argument(
        "--baseline-budget",
        type=int,
        default=None,
        help="which budget is the baseline (default: largest)",
    )
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--position", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", default="niah", help="eval name from EVAL_REGISTRY (default: niah)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--json", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    budgets = [int(x) for x in str(args.budgets).replace(" ", "").split(",") if x]
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
    results = gate.evaluate(budgets=budgets, eval=args.eval)
    if args.json:
        print(
            json.dumps(
                [_jsonable(r.to_dict()) for r in results],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_fmt_table(results))
    return 0


__all__ = [
    "EVAL_REGISTRY",
    "Eval",
    "GateConfig",
    "GateResult",
    "NeedleInHaystack",
    "NiahResult",
    "RecallGate",
    "build_needle_messages",
    "build_needle_prompt",
    "judge",
    "make_needle",
    "needle_in_answer",
    "quality_per_dollar",
    "register_eval",
]


if __name__ == "__main__":
    raise SystemExit(main())
