"""The regression judge: compare two runs and flag regressions.

Neutral, deterministic, and documented. It never "picks a winner" — it reports
per-metric relative change against published thresholds, and distinguishes
real regressions from noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .methodology import MIN_MEANINGFUL_FRACTION, THRESHOLDS
from .schema import RunResult


class Verdict(str, Enum):
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NOISE = "noise"
    NO_DATA = "no_data"


@dataclass
class RegressionFinding:
    metric: str
    baseline: float
    candidate: float
    relative: float  # (candidate - baseline) / baseline
    threshold: float  # the boundary that would trigger regression
    verdict: Verdict
    note: str = ""

    def fmt(self) -> str:
        arrow = "▲" if self.relative >= 0 else "▼"
        return (
            f"{self.metric:<12} base={self.baseline:>10.3f} "
            f"cand={self.candidate:>10.3f} {arrow}{abs(self.relative) * 100:>6.2f}% "
            f"[{self.verdict.value}] {self.note}"
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "relative": self.relative,
            "threshold": self.threshold,
            "verdict": self.verdict.value,
            "note": self.note,
        }


# (metric, higher_is_better, threshold_key)
_METRICS = [
    ("tg_tps", True, "tg_tps"),
    ("pp_tps", True, "pp_tps"),
    ("ttft_ms", False, "ttft_ms"),
    ("itl_p95_ms", False, "itl_p95_ms"),
]


def _rel(base: float, cand: float) -> float:
    if base == 0.0:
        return 0.0
    return (cand - base) / base


def compare_runs(
    base: RunResult, cand: RunResult, thresholds: dict | None = None
) -> list[RegressionFinding]:
    t = dict(THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    findings: list[RegressionFinding] = []
    m1, m2 = base.metrics, cand.metrics
    for metric, higher_better, key in _METRICS:
        if metric == "itl_p95_ms":
            b = m1.itl.p95_ms
            c = m2.itl.p95_ms
        else:
            b = getattr(m1, metric)
            c = getattr(m2, metric)

        if b == 0.0 or c == 0.0:
            note = (
                "not measured"
                if b == 0.0 and c == 0.0
                else "zero/missing baseline (cannot compare)"
            )
            findings.append(
                RegressionFinding(metric, b, c, 0.0, t.get(key, 0.0), Verdict.NO_DATA, note)
            )
            continue

        rel = _rel(b, c)
        thr = t.get(key, 0.0)

        if higher_better:
            # regression = drop past the (negative) threshold
            if rel <= thr:
                verdict = Verdict.REGRESSION
                note = f"dropped {abs(rel) * 100:.2f}% (threshold {thr * 100:.1f}%)"
            elif rel >= MIN_MEANINGFUL_FRACTION:
                verdict = Verdict.IMPROVEMENT
                note = f"improved {rel * 100:.2f}%"
            else:
                # small change or small degradation, all within tolerance
                verdict = Verdict.NOISE
                note = f"within tolerance: {rel * 100:+.2f}% (threshold {thr * 100:.1f}%)"
        else:
            # regression = rise past the (positive) threshold
            if rel >= thr:
                verdict = Verdict.REGRESSION
                note = f"increased {rel * 100:.2f}% (threshold {thr * 100:.1f}%)"
            elif rel <= -MIN_MEANINGFUL_FRACTION:
                verdict = Verdict.IMPROVEMENT
                note = f"decreased {abs(rel) * 100:.2f}%"
            else:
                # small change or small degradation, all within tolerance
                verdict = Verdict.NOISE
                note = f"within tolerance: {rel * 100:+.2f}% (threshold {thr * 100:.1f}%)"

        findings.append(RegressionFinding(metric, b, c, rel, thr, verdict, note))

    return findings


def any_regression(findings: list[RegressionFinding]) -> bool:
    return any(f.verdict == Verdict.REGRESSION for f in findings)


def comparability_issues(base: RunResult, cand: RunResult) -> list[str]:
    """Return the reasons two runs are NOT fairly comparable (apples-to-oranges).

    Deliberately excludes `backend_version`: comparing the *same* workload across
    two versions of an engine is exactly what a regression gate does, so a
    version difference is the variable under test, not an incompatibility.
    """
    issues: list[str] = []
    if base.environment.accelerator.kind != cand.environment.accelerator.kind:
        issues.append(
            f"accelerator differs: {base.environment.accelerator.kind!r} vs "
            f"{cand.environment.accelerator.kind!r}"
        )
    s1, s2 = base.spec.sampling, cand.spec.sampling
    if (s1.temperature, s1.top_p, s1.top_k, s1.seed) != (
        s2.temperature,
        s2.top_p,
        s2.top_k,
        s2.seed,
    ):
        issues.append("sampling differs (temperature/top_p/top_k/seed)")
    d1 = base.spec.extra.get("device")
    d2 = cand.spec.extra.get("device")
    if d1 and d2 and d1 != d2:
        issues.append(f"device differs: {d1!r} vs {d2!r}")
    return issues
