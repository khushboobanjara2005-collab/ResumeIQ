"""SQLite storage for saved comparison results and shortlist selections."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "resumeiq_history.sqlite3"
)


def _database_path(database_path: str | Path | None) -> Path:
    if database_path is not None:
        return Path(database_path)
    return Path(os.environ.get("RESUMEIQ_DATABASE", DEFAULT_DATABASE_PATH))


def initialize_database(database_path: str | Path | None = None) -> None:
    """Create the local comparison-history table when it does not exist."""
    path = _database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_name TEXT NOT NULL,
                job_title TEXT,
                compared_at TEXT NOT NULL,
                keyword_match_percentage REAL,
                cosine_similarity REAL NOT NULL,
                matched_keywords TEXT NOT NULL,
                missing_keywords TEXT NOT NULL,
                resume_fingerprint TEXT NOT NULL,
                job_description_fingerprint TEXT NOT NULL,
                shortlisted INTEGER NOT NULL DEFAULT 0
                    CHECK (shortlisted IN (0, 1))
            )
            """
        )
        connection.commit()


def _connect(database_path: str | Path | None) -> sqlite3.Connection:
    path = _database_path(database_path)
    initialize_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _decode_comparison(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a database row into a safe UI-ready comparison record."""
    record = dict(row)
    record["matched_keywords"] = json.loads(record["matched_keywords"])
    record["missing_keywords"] = json.loads(record["missing_keywords"])
    record["shortlisted"] = bool(record["shortlisted"])
    return record


def save_comparison(
    resume_name: str,
    job_title: str,
    result: dict[str, Any],
    database_path: str | Path | None = None,
) -> int:
    """Save comparison metadata and metrics without retaining document text."""
    fingerprints = result["input_fingerprints"]
    with closing(_connect(database_path)) as connection:
        cursor = connection.execute(
            """
            INSERT INTO comparisons (
                resume_name, job_title, compared_at, keyword_match_percentage,
                cosine_similarity, matched_keywords, missing_keywords,
                resume_fingerprint, job_description_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resume_name.strip() or "Unnamed resume",
                job_title.strip() or None,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                result["keyword_coverage"],
                result["cosine_similarity"],
                json.dumps(result["matched_keywords"]),
                json.dumps(result["missing_keywords"]),
                fingerprints["resume"],
                fingerprints["job_description"],
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_comparisons(
    shortlisted_only: bool = False,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return saved comparisons in newest-first order."""
    query = "SELECT * FROM comparisons"
    if shortlisted_only:
        query += " WHERE shortlisted = 1"
    query += " ORDER BY compared_at DESC, id DESC"
    with closing(_connect(database_path)) as connection:
        rows = connection.execute(query).fetchall()
    return [_decode_comparison(row) for row in rows]


def get_comparison(
    comparison_id: int,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load one saved comparison by its database ID."""
    with closing(_connect(database_path)) as connection:
        row = connection.execute(
            "SELECT * FROM comparisons WHERE id = ?", (comparison_id,)
        ).fetchone()
    return _decode_comparison(row) if row is not None else None


def set_shortlisted(
    comparison_id: int,
    shortlisted: bool,
    database_path: str | Path | None = None,
) -> None:
    """Persist a shortlist selection for one comparison."""
    with closing(_connect(database_path)) as connection:
        cursor = connection.execute(
            "UPDATE comparisons SET shortlisted = ? WHERE id = ?",
            (int(shortlisted), comparison_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Comparison {comparison_id} was not found.")
        connection.commit()


def delete_comparison(
    comparison_id: int,
    database_path: str | Path | None = None,
) -> None:
    """Delete one saved comparison."""
    with closing(_connect(database_path)) as connection:
        cursor = connection.execute(
            "DELETE FROM comparisons WHERE id = ?", (comparison_id,)
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Comparison {comparison_id} was not found.")
        connection.commit()
