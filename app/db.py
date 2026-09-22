"""SQLite persistence: schema creation (on startup) and CRUD helpers."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import Request

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    wave_count INTEGER NOT NULL,
    optimal INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    bundle_sha256 TEXT
);
"""

# Content-addressing key for archived plans. NULL for plans created
# directly via POST /plans; unique among archived plans so an import with
# an existing digest is idempotent.
_BUNDLE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_plans_bundle_sha256
    ON plans(bundle_sha256)
    WHERE bundle_sha256 IS NOT NULL;
"""


def init_db(database_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(database_path))
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(database_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        # Lightweight migration for databases created before bundles: the
        # column must exist before the partial unique index is created.
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(plans)").fetchall()
        }
        if "bundle_sha256" not in existing:
            conn.execute("ALTER TABLE plans ADD COLUMN bundle_sha256 TEXT")
        conn.executescript(_BUNDLE_INDEX)
        conn.commit()
    finally:
        conn.close()


def _connect(database_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_db(request: Request) -> sqlite3.Connection:
    """FastAPI dependency yielding a per-request connection."""
    conn = _connect(request.app.state.settings.database_path)
    try:
        yield conn
    finally:
        conn.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def insert_plan(
    conn: sqlite3.Connection,
    name: str | None,
    request_payload: dict[str, Any],
    result_payload: dict[str, Any],
    wave_count: int,
    optimal: bool,
) -> int:
    cur = conn.execute(
        "INSERT INTO plans (name, request_json, result_json, wave_count, "
        "optimal, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            name,
            json.dumps(request_payload, ensure_ascii=False),
            json.dumps(result_payload, ensure_ascii=False),
            wave_count,
            int(optimal),
            _utcnow(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def fetch_plan(conn: sqlite3.Connection, plan_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()


def list_plans(
    conn: sqlite3.Connection, limit: int, offset: int
) -> tuple[int, list[sqlite3.Row]]:
    total = int(conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0])
    rows = conn.execute(
        "SELECT * FROM plans ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    return total, rows


def delete_plan(conn: sqlite3.Connection, plan_id: int) -> bool:
    cur = conn.execute("DELETE FROM plans WHERE id = ?", (plan_id,))
    conn.commit()
    return cur.rowcount > 0


def fetch_plan_by_digest(
    conn: sqlite3.Connection, digest: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM plans WHERE bundle_sha256 = ?", (digest,)
    ).fetchone()


def import_plan(
    conn: sqlite3.Connection,
    name: str | None,
    request_payload: dict[str, Any],
    result_payload: dict[str, Any],
    wave_count: int,
    optimal: bool,
    digest: str,
) -> tuple[sqlite3.Row, bool]:
    """Insert an archived plan in a single transaction.

    The ``bundle_sha256`` unique index makes the digest the idempotency
    key: if a plan with the same digest already exists its row is
    returned untouched (``reused=True``). A check-then-insert race is
    resolved by catching the integrity violation and re-reading the
    winner, so concurrent imports of one archive create exactly one row.
    Any failure before the commit leaves the database unchanged.
    """
    existing = fetch_plan_by_digest(conn, digest)
    if existing is not None:
        return existing, True

    try:
        cur = conn.execute(
            "INSERT INTO plans (name, request_json, result_json, wave_count, "
            "optimal, created_at, bundle_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                json.dumps(request_payload, ensure_ascii=False),
                json.dumps(result_payload, ensure_ascii=False),
                wave_count,
                int(optimal),
                _utcnow(),
                digest,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # A concurrent transaction imported the same digest first.
        conn.rollback()
        winner = fetch_plan_by_digest(conn, digest)
        assert winner is not None
        return winner, True

    row = fetch_plan(conn, int(cur.lastrowid))
    assert row is not None
    return row, False
