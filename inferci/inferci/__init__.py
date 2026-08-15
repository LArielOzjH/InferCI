"""InferCI — neutral, reproducible inference performance & cost regression CI.

Design principle: the value is NOT owning GPUs; it is the neutral
orchestration + spec + aggregation. Runners (GPU/CPU/NPU) are contributed
resources. This package is the GPU-free core.
"""

__version__ = "0.2.0"

from .regression import RegressionFinding, Verdict, compare_runs
from .schema import (
    Accelerator,
    BenchmarkSpec,
    CostResult,
    Environment,
    Metrics,
    PerTokenLatency,
    RunResult,
    Sampling,
)

__all__ = [
    "Accelerator",
    "BenchmarkSpec",
    "CostResult",
    "Environment",
    "Metrics",
    "PerTokenLatency",
    "RegressionFinding",
    "RunResult",
    "Sampling",
    "Verdict",
    "__version__",
    "compare_runs",
]
