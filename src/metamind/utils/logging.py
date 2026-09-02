from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    logs_dir: Path,
    run_id: str,
    level: str = "INFO",
    console: bool = True,
) -> Path:
    """
    Creates a file logger at: logs_dir/<run_id>.log
    Returns log file path.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{run_id}.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Reset root handlers to avoid duplicate logs in notebooks / repeated runs
    root = logging.getLogger()
    root.setLevel(numeric_level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging initialized. File=%s", log_file)
    return log_file