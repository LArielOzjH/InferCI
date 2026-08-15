"""Runner protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import BenchmarkSpec, Environment, RunResult


class Runner(ABC):
    id: str = "base"
    name: str = "Base runner"

    @abstractmethod
    def capture_environment(self) -> Environment:
        """Return a fully-populated Environment for this backend."""

    @abstractmethod
    def run(self, spec: BenchmarkSpec, environment: Environment) -> RunResult:
        """Execute the benchmark described by `spec` and return a RunResult."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Runner {self.id}>"
