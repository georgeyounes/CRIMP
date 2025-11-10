"""
Logging utilities
"""

from __future__ import annotations
import logging
from logging import StreamHandler, Logger
from logging.handlers import RotatingFileHandler

_FMT = "[%(asctime)s] %(levelname)8s %(message)s (%(name)s:%(lineno)s)"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
        *,
        console_level: str = "WARNING",
        file_path: str | None = None,
        file_level: str = "INFO",
        file_max_bytes: int = 10_000_000,
        file_backup_count: int = 3,
        force: bool = False,
) -> None:
    root = logging.getLogger()

    # Clear existing handlers if requested (important in notebooks / repeated runs)
    if force:
        for h in root.handlers[:]:
            root.removeHandler(h)

    # Always set root as permissive as possible; handler levels do the filtering
    root.setLevel(logging.DEBUG)

    # Console handler
    ch = StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.WARNING))
    ch.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(ch)

    # Optional file handler (truncate on each run)
    if file_path:
        open(file_path, "w").close()  # clean-up existing .log file mode="w" below was not enough
        fh = RotatingFileHandler(
            file_path,
            mode="w",           # truncate on first open in this process
            maxBytes=file_max_bytes,
            backupCount=file_backup_count
        )
        fh.setLevel(getattr(logging, file_level.upper(), logging.INFO))
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(fh)


def get_logger(name: str) -> Logger:
    """
    Return a module/package logger preloaded with a NullHandler so importing
    this module never configures global logging by accident
    """
    logger = logging.getLogger(name)
    # Ensure no 'No handler could be found' warnings in apps that never configure logging
    if not logger.handlers and not logger.propagate:
        # If a library wants to ensure silence by default, attach NullHandler
        logger.addHandler(logging.NullHandler())
    return logger
