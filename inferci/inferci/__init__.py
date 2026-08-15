"""InferCI — neutral, reproducible inference performance & cost regression CI.

Design principle: the value is NOT owning GPUs; it is the neutral
orchestration + spec + aggregation. Runners (GPU/CPU/NPU) are contributed
resources. This package is the GPU-free core.
"""

__version__ = "0.1.0"

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
from .regression import compare_runs, RegressionFinding, Verdict

__all__ = [
    "__version__",
    "Accelerator",
    "BenchmarkSpec",
    "CostResult",
    "Environment",
    "Metrics",
    "PerTokenLatency",
    "RunResult",
    "Sampling",
    "compare_runs",
    "RegressionFinding",
    "Verdict",
]
