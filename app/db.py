"""SQLite persistence: schema creation (on startup) and CRUD helpers."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import Request

from . import API_VERSION
from .bundle import plan_digest

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    wave_count INTEGER NOT NULL,
    optimal INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    digest TEXT
);
"""

# Created only after the column migration below, so pre-digest databases
# (where CREATE TABLE IF NOT EXISTS is a no-op) get the column first.
_INDEX = "CREATE INDEX IF NOT EXISTS idx_plans_digest ON plans(digest);"


def init_db(database_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(database_path))
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(database_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        _migrate_schema(conn)
        conn.execute(_INDEX)
        _backfill_digests(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema (idempotent)."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(plans)")}
    if "digest" not in columns:
        conn.execute("ALTER TABLE plans ADD COLUMN digest TEXT")


def _backfill_digests(conn: sqlite3.Connection) -> None:
    """Content-address plans stored before the digest column existed."""
    rows = conn.execute(
        "SELECT id, name, result_json, wave_count, optimal FROM plans "
        "WHERE digest IS NULL"
    ).fetchall()
    for plan_id, name, result_json, wave_count, optimal in rows:
        digest = plan_digest(
            name,
            json.loads(result_json),
            wave_count,
            bool(optimal),
            API_VERSION,
        )
        conn.execute("UPDATE plans SET digest = ? WHERE id = ?", (digest, plan_id))


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
    digest: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO plans (name, request_json, result_json, wave_count, "
        "optimal, created_at, digest) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
    return int(cur.lastrowid)


def insert_plan_imported(
    conn: sqlite3.Connection,
    name: str | None,
    request_payload: dict[str, Any],
    result_payload: dict[str, Any],
    wave_count: int,
    optimal: bool,
    digest: str,
) -> tuple[int, bool]:
    """Insert an imported bundle as a new plan, atomically deduplicated.

    The digest lookup and the insert run in one immediate transaction: a
    bundle whose digest is already stored writes nothing and returns the id
    of the existing plan (``created=False``). Any failure rolls the
    transaction back, so a failed import never writes partial data.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT id FROM plans WHERE digest = ? ORDER BY id LIMIT 1",
            (digest,),
        ).fetchone()
        if row is not None:
            conn.rollback()
            return int(row[0]), False
        cur = conn.execute(
            "INSERT INTO plans (name, request_json, result_json, wave_count, "
            "optimal, created_at, digest) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
        return int(cur.lastrowid), True
    except BaseException:
        conn.rollback()
        raise


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
