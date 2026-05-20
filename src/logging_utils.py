"""
logging_utils.py
────────────────
Centralised logging configuration for the Anticipatory Energy Stress project.
Creates both a rotating file handler and a coloured console handler.
"""

import logging
import sys
from pathlib import Path


_RESET  = "\033[0m"
_GREY   = "\033[90m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BOLD   = "\033[1m"

_LEVEL_COLOURS = {
    logging.DEBUG:    _GREY,
    logging.INFO:     _CYAN,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _BOLD + _RED,
}


class _ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, _RESET)
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


def setup_logger(
    name: str = "energy_stress",
    log_file: str = "outputs/logs/step6_final_outputs.log",
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Return a logger that writes DEBUG+ to *log_file* and INFO+ to the console.

    Parameters
    ----------
    name     : logger name (use __name__ in callers)
    log_file : path to the log file (directories are created automatically)
    level    : minimum level captured in the file

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # ── File handler (full detail) ──────────────────────────────────────────
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # ── Console handler (INFO+, coloured) ───────────────────────────────────
    # Wrap stdout in a reconfigured writer that tolerates non-ASCII on Windows
    try:
        import io
        safe_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        safe_stdout = sys.stdout     # fallback if stdout has no .buffer

    ch = logging.StreamHandler(safe_stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColouredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve (or create) a child logger under the project root."""
    return logging.getLogger(f"energy_stress.{name}")
