"""Structured, deterministic logging."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "scroll_anchor") -> logging.Logger:
    return logging.getLogger(name)


def configure(level: str = "INFO") -> None:
    """Configure a single stderr handler with a compact structured format."""
    global _CONFIGURED
    root = logging.getLogger("scroll_anchor")
    root.setLevel(level.upper())
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
