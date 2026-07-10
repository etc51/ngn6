from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


UNKNOWN_COMMIT_HASH = "unknown"
_COMMIT_ENV_NAMES = ("NGN6_COMMIT_HASH", "GIT_COMMIT", "COMMIT_SHA")


@lru_cache(maxsize=1)
def current_commit_hash() -> str:
    for name in _COMMIT_ENV_NAMES:
        value = _clean_hash(os.environ.get(name))
        if value:
            return value

    project_root = _project_root()
    version_file_hash = _commit_hash_from_file(project_root / ".commit_hash")
    if version_file_hash:
        return version_file_hash

    git_hash = _commit_hash_from_git(project_root)
    if git_hash:
        return git_hash
    git_hash = _commit_hash_from_git(Path.cwd())
    if git_hash:
        return git_hash
    return UNKNOWN_COMMIT_HASH


def add_commit_hash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["commit_hash"] = current_commit_hash()
    return payload


def with_commit_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return add_commit_hash(dict(payload))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _commit_hash_from_file(path: Path) -> str | None:
    try:
        return _clean_hash(path.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def _commit_hash_from_git(cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return _clean_hash(completed.stdout.strip())


def _clean_hash(value: str | None) -> str | None:
    if not value:
        return None
    parsed = value.strip()
    if not parsed:
        return None
    if parsed.lower() in {"unknown", "none", "null"}:
        return None
    return parsed
