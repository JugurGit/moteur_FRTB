# history_db.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def db_path() -> Path:
    # DB à côté de app.py / ui_common.py
    return Path(__file__).resolve().parent / "frtb_history.sqlite3"


def _connect() -> sqlite3.Connection:
    p = db_path()
    con = sqlite3.connect(str(p), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    con = _connect()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc     TEXT NOT NULL,
                created_at_local   TEXT NOT NULL,
                status            TEXT NOT NULL,            -- "ok" / "error"
                k_eq              REAL,
                k_sw              REAL,
                k_bo              REAL,
                k_total           REAL,
                portfolio_csv     TEXT,
                snapshot_json     TEXT,
                results_json      TEXT,
                logs_txt          TEXT,
                meta_json         TEXT,
                error_txt         TEXT
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at_utc);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);")
        con.commit()
    finally:
        con.close()


def insert_run(rec: Dict[str, Any]) -> int:
    init_db()
    con = _connect()
    try:
        cols = [
            "created_at_utc",
            "created_at_local",
            "status",
            "k_eq",
            "k_sw",
            "k_bo",
            "k_total",
            "portfolio_csv",
            "snapshot_json",
            "results_json",
            "logs_txt",
            "meta_json",
            "error_txt",
        ]
        vals = [rec.get(c) for c in cols]
        q = f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        cur = con.execute(q, vals)
        con.commit()
        return int(cur.lastrowid)
    finally:
        con.close()


def list_runs(limit: int = 200, status: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db()
    con = _connect()
    try:
        if status in ("ok", "error"):
            cur = con.execute(
                """
                SELECT id, created_at_local, created_at_utc, status, k_eq, k_sw, k_bo, k_total
                FROM runs
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (status, int(limit)),
            )
        else:
            cur = con.execute(
                """
                SELECT id, created_at_local, created_at_utc, status, k_eq, k_sw, k_bo, k_total
                FROM runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    con = _connect()
    try:
        cur = con.execute("SELECT * FROM runs WHERE id = ?", (int(run_id),))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def delete_run(run_id: int) -> None:
    init_db()
    con = _connect()
    try:
        con.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
        con.commit()
    finally:
        con.close()


def clear_all() -> None:
    init_db()
    con = _connect()
    try:
        con.execute("DELETE FROM runs")
        con.commit()
    finally:
        con.close()
