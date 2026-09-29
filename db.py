"""SQLite storage for users, jobs, and applications. No ORM — plain sqlite3
kept intentionally simple and dependency-light."""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_portal.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('recruiter', 'job_seeker')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recruiter_id INTEGER NOT NULL REFERENCES users(id),
    title TEXT NOT NULL,
    company TEXT,
    description TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    seeker_id INTEGER NOT NULL REFERENCES users(id),
    resume_filename TEXT,
    resume_text TEXT NOT NULL,
    match_score REAL NOT NULL,
    matched_keywords TEXT,
    llm_score REAL,
    llm_summary TEXT,
    llm_strengths TEXT,
    llm_gaps TEXT,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, seeker_id)
);
"""


def _add_column_if_missing(conn, table, column, coltype):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA)
    # Migrate older databases created before the Groq analysis columns existed.
    for col, coltype in [
        ("llm_score", "REAL"),
        ("llm_summary", "TEXT"),
        ("llm_strengths", "TEXT"),
        ("llm_gaps", "TEXT"),
    ]:
        _add_column_if_missing(conn, "applications", col, coltype)
    conn.commit()
    conn.close()
