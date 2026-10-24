"""Logging for the command-line scripts in ``scripts/`` (private).

The library only calls ``logging.getLogger(__name__)`` and never configures handlers. A script
calls :func:`setup_script_logging` once, which attaches handlers to the ``lunarscout`` logger so
messages from the library and from the script go to the same places:

* a **log file** (when a path is given), in plain text with a full timestamp, the level and the
  logger name. It never contains color codes, so it replaces ``| tee``;
* the **console**, on standard output. With ``color="auto"`` (the default) the console uses
  ``rich`` (colored levels, readable layout) when standard output is a terminal and ``NO_COLOR``
  is not set, and a plain one-line-per-message handler otherwise, so a pipe or a redirect gets
  clean text. ``color="always"`` forces ``rich`` even into a pipe, ``color="never"`` forces the
  plain handler.

``rich`` is an optional dependency (``pip install "lunarscout[console]"``). Without it the console
is always plain and a single notice says so. Importing this module imports neither ``rich`` nor
anything heavy.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT_LOGGER = "lunarscout"
COLOR_MODES = ("auto", "always", "never")
_MARK = "_lunarscout_script_handler"
_saved_state: list = []          # [level, propagate] of the logger before the first setup
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"
#: Rich wraps at the terminal width; when it is forced into a pipe there is no terminal, so
#: it gets a width wide enough for the longest progress lines.
PIPED_RICH_WIDTH = 200


def rich_available() -> bool:
    """Whether ``rich`` can be imported."""
    try:
        import rich.logging  # noqa: F401
    except ImportError:
        return False
    return True


class _StdoutHandler(logging.StreamHandler):
    """A stream handler on whatever ``sys.stdout`` is when a message is written (so a captured
    or replaced stdout is followed, as ``logging.lastResort`` does for stderr)."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stdout


def _level(level) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level).upper())
    if not isinstance(value, int):
        raise ValueError(f"unknown log level {level!r}")
    return value


def _console_handler(color: str, announce: list[str]) -> logging.Handler:
    is_terminal = bool(getattr(sys.stdout, "isatty", lambda: False)())
    want_rich = color == "always" or (color == "auto" and is_terminal and "NO_COLOR" not in os.environ)
    if want_rich and not rich_available():
        announce.append(
            "rich is not installed, so the console output is not colored "
            '(pip install "lunarscout[console]")'
        )
        want_rich = False
    if want_rich:
        from rich.console import Console
        from rich.logging import RichHandler

        console = Console(
            file=None,                                   # sys.stdout as it is at each write
            force_terminal=True if color == "always" else None,
            width=None if is_terminal else PIPED_RICH_WIDTH,
        )
        handler: logging.Handler = RichHandler(
            console=console, markup=False, show_path=False, rich_tracebacks=True,
            log_time_format=CONSOLE_DATE_FORMAT, omit_repeated_times=False,
        )
    else:
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT))
    return handler


def reset_script_logging() -> None:
    """Remove the handlers :func:`setup_script_logging` attached and close the log file."""
    logger = logging.getLogger(ROOT_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, _MARK, False):
            logger.removeHandler(handler)
            handler.close()
    if _saved_state:
        logger.setLevel(_saved_state[0])
        logger.propagate = _saved_state[1]
        _saved_state.clear()


def setup_script_logging(log_path: str | Path | None = None, *, color: str = "auto",
                         level: str | int = "INFO") -> logging.Logger:
    """Attach the console handler, and a plain log file if ``log_path`` is given, to the
    ``lunarscout`` logger and return it. Calling it again replaces the earlier handlers."""
    if color not in COLOR_MODES:
        raise ValueError(f"color must be one of {COLOR_MODES}, not {color!r}")
    numeric = _level(level)
    if log_path is not None:
        log_path = Path(log_path)
        if not log_path.parent.is_dir():
            raise ValueError(f"the directory of the log file {log_path} does not exist")
    reset_script_logging()
    logger = logging.getLogger(ROOT_LOGGER)
    _saved_state[:] = [logger.level, logger.propagate]
    logger.setLevel(numeric)
    logger.propagate = False
    announce: list[str] = []
    handlers = [_console_handler(color, announce)]
    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT))
        handlers.append(file_handler)
    for handler in handlers:
        setattr(handler, _MARK, True)
        logger.addHandler(handler)
    for message in announce:
        logger.warning(message)
    return logger
