"""Timestamped logging to terminal and a per-run log file.

Global context: every phase of the pipeline logs through `get_logger()`, so prompts sent to
OpenAI and the full outputs that come back are recorded, uncut, in `logs/<timestamp>.log`
and mirrored to the terminal in real time.
Source: https://docs.python.org/3/howto/logging.html#logging-to-multiple-destinations
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

FORMAT = "[%(asctime)s] %(levelname)s %(message)s"  # e.g. [2026-09-08 11:18:31] INFO extracting audio


def get_logger(log_dir: Path = Path("logs")) -> logging.Logger:
    """Return a logger writing to both stderr and a new timestamped file under `log_dir`."""
    log_dir.mkdir(parents=True, exist_ok=True)  # create logs/ if missing
    log = logging.getLogger("transcribe")  # one named logger, shared by all modules
    if log.handlers:  # already configured (e.g. called twice) -> reuse
        return log
    log.setLevel(logging.INFO)
    file = log_dir / f"transcribe_{datetime.now():%Y%m%d_%H%M%S}.log"  # unique file per run
    for h in (logging.FileHandler(file, encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        h.setFormatter(logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))  # same format in both sinks
        log.addHandler(h)
    log.info("log file: %s", file)
    return log
