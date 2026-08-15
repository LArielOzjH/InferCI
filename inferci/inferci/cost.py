"""Cost model: turn measured throughput into $/1M tokens.

For cloud runners:  $/1M = (instance_hourly / 3600) * 1e6 / tokens_per_sec
For local runs:     price is 0 / "local" (you own the hardware).
Prices are approximate public on-demand USD and MUST be verified before quoting.
"""
from __future__ import annotations

from .schema import CostResult

# Approximate AWS on-demand USD/hour (verify before publishing). Keys are
# deliberately generic so other clouds / spot can be added.
PRICE_CATALOG = {
    "cpu.m7i.xlarge":   {"hourly": 0.2016, "kind": "cpu"},
    "gpu.t4.g4dn.xlarge": {"hourly": 0.526, "kind": "gpu"},
    "gpu.a10g.g5.xlarge": {"hourly": 1.006, "kind": "gpu"},
    "gpu.a10g.g5.2xlarge": {"hourly": 1.212, "kind": "gpu"},
    "gpu.v100.p3.2xlarge": {"hourly": 3.06, "kind": "gpu"},
    "gpu.a100.p4d.24xlarge": {"hourly": 32.77, "kind": "gpu"},
}


def lookup_price(instance_type: str) -> dict | None:
    entry = PRICE_CATALOG.get(instance_type)
    return dict(entry) if entry else None


def compute_cost(pp_tps: float, tg_tps: float,
                 instance_type: str | None = None,
                 instance_hourly: float | None = None) -> CostResult:
    """Derive $/1M tokens from throughput + instance price."""
    if instance_hourly is not None:
        hourly = instance_hourly
        source = "user-supplied"
    elif instance_type:
        entry = PRICE_CATALOG.get(instance_type)
        if entry is None:
            raise ValueError(
                f"unknown instance_type {instance_type!r}; available: "
                f"{sorted(PRICE_CATALOG)} (or pass instance_hourly)"
            )
        hourly = entry["hourly"]
        source = f"catalog:{instance_type}"
    else:
        # Only a truly local run (no instance requested) is priced as local.
        return CostResult(instance_type="local",
                          source="local (no cloud price; you own hardware)")

    per_sec = hourly / 3600.0
    price_input = (per_sec * 1e6 / pp_tps) if pp_tps > 0 else 0.0
    price_output = (per_sec * 1e6 / tg_tps) if tg_tps > 0 else 0.0
    return CostResult(
        price_per_input_1m=round(price_input, 6),
        price_per_output_1m=round(price_output, 6),
        instance_hourly=hourly,
        instance_type=instance_type or "",
        source=source,
    )
