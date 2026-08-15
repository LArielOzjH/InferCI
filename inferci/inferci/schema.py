"""Canonical data model for InferCI.

The schema is the *moat*: a single, neutral, machine-readable description of
"what was run, where, and what happened". Everything (store, regression, cost,
dashboard, BYO-runner protocol) speaks this schema.
"""
from __future__ import annotations

import datetime
import json
import platform
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

SPEC_VERSION = "0.1.0"


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_run_id() -> str:
    # full 128-bit UUID: a truncated id risks birthday collisions that would
    # collide inside the append-only ledger.
    return str(uuid.uuid4())


def _pick(d: dict, cls) -> dict:
    """Keep only keys that are declared fields of `cls` (tolerant deserialization)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class Accelerator:
    """The compute device the workload ran on."""
    kind: str = "cpu"          # cpu | cuda | metal | rocm | npu | ...
    name: str = ""             # e.g. "Apple M1 Pro", "NVIDIA H100"
    memory_gb: float = 0.0
    driver: str = ""           # e.g. CUDA 12.4, Metal 3


@dataclass
class Environment:
    """Where a run happened. Must be captured automatically, never typed by hand."""
    host: str = ""
    os: str = ""
    arch: str = ""
    cpu: str = ""
    ram_gb: float = 0.0
    accelerator: Accelerator = field(default_factory=Accelerator)
    backend: str = ""          # runner id: llama_cpp | vllm | sglang | ...
    backend_version: str = ""  # git commit / tag / version string
    lib_versions: dict = field(default_factory=dict)
    captured_at: str = field(default_factory=now_utc)


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 42


@dataclass
class BenchmarkSpec:
    """The *what* of a run. Two runs are comparable iff their specs match
    (same id), so reproducibility lives here."""
    spec_version: str = SPEC_VERSION
    id: str = ""               # canonical spec key, e.g. "llama_cpp.qwen25-0.5b.Q4_K_M.cpu.pp512.tg128"
    model_id: str = ""         # canonical model name, not a path
    model_file: str = ""       # local path or HF ref
    backend: str = ""          # runner id
    quantization: str = ""     # Q4_K_M | FP16 | FP8 | AWQ-4bit | ...
    prompt_tokens: int = 512   # prefill length
    gen_tokens: int = 128      # decode length
    repeats: int = 3
    warmup_repeats: int = 1
    batch: int = 1             # serving concurrency (1 = single stream)
    sampling: Sampling = field(default_factory=Sampling)
    prompt_set: str = "synthetic"   # which prompt/data set
    extra: dict = field(default_factory=dict)

    def canonical_id(self) -> str:
        if self.id:
            return self.id
        return ".".join([
            self.backend, self.model_id, self.quantization,
            f"pp{self.prompt_tokens}", f"tg{self.gen_tokens}", f"b{self.batch}",
        ])


@dataclass
class PerTokenLatency:
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


@dataclass
class Metrics:
    # Throughput (higher = better)
    pp_tps: float = 0.0       # prompt-processing tokens/sec (prefill)
    tg_tps: float = 0.0       # token-generation tokens/sec (decode)
    pp_tps_std: float = 0.0
    tg_tps_std: float = 0.0
    # Latency (lower = better) — filled by serving-style runners
    ttft_ms: float = 0.0      # time-to-first-token
    itl: PerTokenLatency = field(default_factory=PerTokenLatency)
    # Accounting
    total_seconds: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    peak_rss_mb: float = 0.0
    model_size_mb: float = 0.0


@dataclass
class CostResult:
    """$/token economics. For local CPU this may be 'local' with zero price;
    for cloud runners it is derived from instance price / measured throughput."""
    currency: str = "USD"
    price_per_input_1m: float = 0.0
    price_per_output_1m: float = 0.0
    instance_hourly: float = 0.0
    instance_type: str = ""
    source: str = ""          # where the price came from (catalog/aws/...)


@dataclass
class RunResult:
    run_id: str = field(default_factory=new_run_id)
    spec: BenchmarkSpec = field(default_factory=BenchmarkSpec)
    environment: Environment = field(default_factory=Environment)
    metrics: Metrics = field(default_factory=Metrics)
    cost: Optional[CostResult] = None
    created_at: str = field(default_factory=now_utc)
    raw: dict = field(default_factory=dict)   # runner-specific raw output (never parsed for judgments)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        # Tolerant: unknown/extra keys (from future or foreign BYO runners) are
        # dropped instead of crashing the whole read. Nested dataclasses are
        # rebuilt explicitly so their own fields are also filtered.
        spec_d = _pick(d.get("spec") or {}, BenchmarkSpec)
        sampling = Sampling(**_pick(spec_d.pop("sampling", None) or {}, Sampling))
        spec = BenchmarkSpec(sampling=sampling, **spec_d)
        env_d = _pick(d.get("environment") or {}, Environment)
        acc = Accelerator(**_pick(env_d.pop("accelerator", None) or {}, Accelerator))
        env = Environment(accelerator=acc, **env_d)
        m_d = _pick(d.get("metrics") or {}, Metrics)
        itl = PerTokenLatency(**_pick(m_d.pop("itl", None) or {}, PerTokenLatency))
        metrics = Metrics(itl=itl, **m_d)
        cost = CostResult(**_pick(d.get("cost") or {}, CostResult)) if d.get("cost") else None
        return cls(
            run_id=d.get("run_id") or new_run_id(),
            spec=spec, environment=env, metrics=metrics, cost=cost,
            created_at=d.get("created_at") or now_utc(),
            raw=d.get("raw") or {},
        )


def capture_local_environment(backend: str = "", backend_version: str = "") -> Environment:
    """Best-effort local environment capture (no GPU probing by default)."""
    acc = Accelerator(kind="cpu")
    cpu = platform.processor() or platform.machine() or ""
    # Apple Silicon name via sysctl when available
    try:
        import subprocess
        name = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if name:
            cpu = name
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        mem = ""
    ram_gb = float(mem) / 1e9 if mem.isdigit() else 0.0
    return Environment(
        host=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu=cpu,
        ram_gb=ram_gb,
        accelerator=acc,
        backend=backend,
        backend_version=backend_version,
        lib_versions={"python": platform.python_version()},
    )
