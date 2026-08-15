"""SQLite persistence for runs. Zero external deps.

Storage is append-only history: every RunResult is a row. The neutral value is
the accumulated, reproducible history ("the ledger"), which later feeds the
dashboard and regression trend-lines.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from typing import Optional

from .schema import RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    created_at   TEXT,
    backend      TEXT,
    model_id     TEXT,
    quantization TEXT,
    spec_id      TEXT,
    tg_tps       REAL,
    pp_tps       REAL,
    ttft_ms      REAL,
    spec_json    TEXT,
    env_json     TEXT,
    metrics_json TEXT,
    cost_json    TEXT,
    raw_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_backend ON runs(backend);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_spec ON runs(spec_id);
"""


class Store:
    def __init__(self, path: str = "inferci.db"):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def insert(self, run: RunResult) -> None:
        spec = run.spec
        try:
            self.conn.execute(
                """INSERT INTO runs
                   (run_id, created_at, backend, model_id, quantization, spec_id,
                    tg_tps, pp_tps, ttft_ms, spec_json, env_json, metrics_json, cost_json, raw_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run.run_id, run.created_at, spec.backend, spec.model_id,
                    spec.quantization, spec.canonical_id(),
                    run.metrics.tg_tps, run.metrics.pp_tps, run.metrics.ttft_ms,
                    json.dumps(asdict(run.spec), ensure_ascii=False),
                    json.dumps(asdict(run.environment), ensure_ascii=False),
                    json.dumps(asdict(run.metrics), ensure_ascii=False),
                    json.dumps(asdict(run.cost), ensure_ascii=False) if run.cost else None,
                    json.dumps(run.raw, ensure_ascii=False),
                ),
            )
        except sqlite3.IntegrityError as e:
            # Append-only contract: a duplicate id must never silently overwrite.
            raise RuntimeError(
                f"run_id {run.run_id!r} already exists in the ledger; "
                "append-only — a new run must mint a new id"
            ) from e
        self.conn.commit()

    def get(self, run_id: str) -> Optional[RunResult]:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        cols = [c[0] for c in self.conn.execute("SELECT * FROM runs LIMIT 0").description]
        d = dict(zip(cols, row))
        return self._row_to_result(d)

    def list(self, limit: int | None = 50, backend: str | None = None,
             model_id: str | None = None) -> list[RunResult]:
        q = "SELECT * FROM runs"
        conds, args = [], []
        if backend:
            conds.append("backend=?")
            args.append(backend)
        if model_id:
            conds.append("model_id=?")
            args.append(model_id)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at DESC"
        if limit is not None:
            q += " LIMIT ?"
            args.append(limit)
        rows = self.conn.execute(q, args).fetchall()
        cols = [c[0] for c in self.conn.execute("SELECT * FROM runs LIMIT 0").description]
        return [self._row_to_result(dict(zip(cols, r))) for r in rows]

    def latest_baseline_for(self, spec_id: str) -> Optional[RunResult]:
        row = self.conn.execute(
            "SELECT * FROM runs WHERE spec_id=? ORDER BY created_at ASC LIMIT 1",
            (spec_id,),
        ).fetchone()
        if not row:
            return None
        cols = [c[0] for c in self.conn.execute("SELECT * FROM runs LIMIT 0").description]
        return self._row_to_result(dict(zip(cols, row)))

    @staticmethod
    def _row_to_result(d: dict) -> RunResult:
        full = {
            "run_id": d["run_id"],
            "spec": json.loads(d["spec_json"]),
            "environment": json.loads(d["env_json"]),
            "metrics": json.loads(d["metrics_json"]),
            "cost": json.loads(d["cost_json"]) if d["cost_json"] else None,
            "created_at": d["created_at"],
            "raw": json.loads(d["raw_json"]) if d["raw_json"] else {},
        }
        return RunResult.from_dict(full)
