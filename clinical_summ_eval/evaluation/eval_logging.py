"""
Tiny structured logger shared across the eval pipeline.

Replaces scattered print() calls with leveled, prefixed, timestamped output
that can be silenced or redirected with one switch. No external deps.

Usage:
    from eval_logging import get_logger
    log = get_logger(__name__)
    log.info("starting chunk %d", i)
    log.warn("length mismatch q=%d a=%d -> reconciled", nq, na)
    log.error("chunk failed: %s", exc)

Set EVAL_LOG_LEVEL=DEBUG|INFO|WARN|ERROR (env) to control verbosity.
"""

from __future__ import annotations

import logging
import os
import sys

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = _LEVELS.get(os.environ.get("EVAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("eval")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


class _Adapter:
    """Thin wrapper so call sites can use .warn() and printf-style args."""

    def __init__(self, name: str):
        _configure_root()
        short = name.split(".")[-1]
        self._log = logging.getLogger(f"eval.{short}")

    def debug(self, msg, *a):
        self._log.debug(msg, *a)

    def info(self, msg, *a):
        self._log.info(msg, *a)

    def warn(self, msg, *a):
        self._log.warning(msg, *a)

    warning = warn

    def error(self, msg, *a):
        self._log.error(msg, *a)


def get_logger(name: str = "eval") -> _Adapter:
    return _Adapter(name)