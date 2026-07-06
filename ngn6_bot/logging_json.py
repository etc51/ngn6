from __future__ import annotations

import logging
import json
from pathlib import Path

from ngn6_bot.runtime_metadata import current_commit_hash

try:
    from pythonjsonlogger import jsonlogger
except ModuleNotFoundError:
    jsonlogger = None


class _StdlibJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "asctime": self.formatTime(record, self.datefmt),
            "levelname": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "event": getattr(record, "event", None),
            "details": getattr(record, "details", {}),
            "commit_hash": current_commit_hash(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _CommitHashFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.commit_hash = current_commit_hash()
        return True


def setup_logging(level: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("ngn6_bot")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    if jsonlogger is not None:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(event)s %(details)s %(commit_hash)s"
        )
    else:
        formatter = _StdlibJsonFormatter()

    commit_filter = _CommitHashFilter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(commit_filter)
    logger.addHandler(stream_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(commit_filter)
        logger.addHandler(file_handler)

    return logger
