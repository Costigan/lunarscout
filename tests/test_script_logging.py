"""The logging set-up the command-line scripts use: a plain log file, and a console that is
colored with rich on a terminal and plain otherwise."""

from __future__ import annotations

import io
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from lunarscout import _script_logging as script_logging

ESCAPE = "\x1b["
LIBRARY = "lunarscout._numba_horizon.safe_haven_windows"
ORIGINAL_LEVEL = logging.getLogger("lunarscout").level          # NOTSET unless something set it


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def clean_logging():
    script_logging.reset_script_logging()
    yield
    script_logging.reset_script_logging()


@pytest.fixture()
def make_terminal(monkeypatch):
    """Install a fake terminal as stdout. Called inside the test, because pytest's capture
    puts its own stdout back between fixture set-up and the test body."""
    def install():
        stream = Terminal()
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setenv("TERM", "xterm-256color")
        for name in ("NO_COLOR", "FORCE_COLOR", "COLUMNS"):
            monkeypatch.delenv(name, raising=False)
        return stream
    return install


def test_a_pipe_gets_plain_one_line_messages_and_the_file_a_full_timestamp(tmp_path, capsys) -> None:
    log = tmp_path / "run.log"
    script_logging.setup_script_logging(log)
    logging.getLogger("lunarscout.scripts.demo").info("Stage 3: 1/4 blocks")
    logging.getLogger(LIBRARY).warning("pixel (1, 2) month 0: stage 1 holds 12.0 h")
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[0].endswith("INFO    Stage 3: 1/4 blocks")
    assert out[1].endswith("WARNING pixel (1, 2) month 0: stage 1 holds 12.0 h")
    assert len(out[0].split()[0]) == 8 and ESCAPE not in "\n".join(out)         # HH:MM:SS, no color codes
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and ESCAPE not in log.read_text()
    assert lines[0].split()[0].count("-") == 2 and lines[0].endswith("INFO    lunarscout.scripts.demo: Stage 3: 1/4 blocks")
    assert lines[1].endswith(f"WARNING {LIBRARY}: pixel (1, 2) month 0: stage 1 holds 12.0 h")


def test_a_second_setup_replaces_the_handlers_and_the_log_file_is_appended_to(tmp_path, capsys) -> None:
    log = tmp_path / "run.log"
    for round_number in (1, 2):
        script_logging.setup_script_logging(log)
        logging.getLogger("lunarscout.scripts.demo").info("round %d", round_number)
    assert capsys.readouterr().out.count("round") == 2          # once per round, not duplicated
    assert [line.split(": ")[-1] for line in log.read_text().splitlines()] == ["round 1", "round 2"]


def test_always_forces_rich_into_a_pipe_without_wrapping_or_touching_the_file(tmp_path, capsys) -> None:
    pytest.importorskip("rich")
    log = tmp_path / "run.log"
    script_logging.setup_script_logging(log, color="always")
    message = "Stage 3: 12/340 blocks, " + "x" * 150
    logging.getLogger("lunarscout.scripts.demo").info(message)
    out = capsys.readouterr().out
    assert ESCAPE in out and "INFO" in out
    assert len(out.strip().splitlines()) == 1 and "x" * 150 in out       # not wrapped at 80 columns
    assert ESCAPE not in log.read_text() and log.read_text().strip().endswith(message)


def test_auto_colors_a_terminal_but_not_when_no_color_is_set_or_color_is_never(make_terminal, monkeypatch) -> None:
    pytest.importorskip("rich")
    terminal = make_terminal()
    log = logging.getLogger("lunarscout.scripts.demo")
    script_logging.setup_script_logging()
    log.info("on a terminal")
    colored = terminal.getvalue()
    assert ESCAPE in colored and "on a terminal" in colored
    terminal.truncate(0)
    terminal.seek(0)
    script_logging.setup_script_logging(color="never")
    log.info("forced plain")
    assert ESCAPE not in terminal.getvalue() and terminal.getvalue().rstrip().endswith("INFO    forced plain")
    terminal.truncate(0)
    terminal.seek(0)
    monkeypatch.setenv("NO_COLOR", "1")
    script_logging.setup_script_logging()
    log.info("no color requested")
    assert ESCAPE not in terminal.getvalue() and "no color requested" in terminal.getvalue()


def test_without_rich_the_console_is_plain_and_says_so_once(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(script_logging, "rich_available", lambda: False)
    log = tmp_path / "run.log"
    script_logging.setup_script_logging(log, color="always")
    logging.getLogger("lunarscout.scripts.demo").info("hello")
    out = capsys.readouterr().out
    assert ESCAPE not in out and out.count("rich is not installed") == 1 and 'lunarscout[console]' in out
    assert "rich is not installed" in log.read_text() and log.read_text().strip().endswith("hello")


def test_the_level_filters_both_handlers(tmp_path, capsys) -> None:
    log = tmp_path / "run.log"
    script_logging.setup_script_logging(log, level="WARNING")
    logging.getLogger("lunarscout.scripts.demo").info("quiet")
    logging.getLogger("lunarscout.scripts.demo").warning("loud")
    assert "quiet" not in capsys.readouterr().out and "quiet" not in log.read_text() and "loud" in log.read_text()
    script_logging.setup_script_logging(log, level=logging.DEBUG)
    logging.getLogger("lunarscout.scripts.demo").debug("detail")
    assert "detail" in log.read_text()


def test_bad_arguments_are_refused_without_changing_the_logging(tmp_path) -> None:
    script_logging.setup_script_logging(tmp_path / "ok.log")
    before = list(logging.getLogger("lunarscout").handlers)
    for kwargs in (dict(color="rainbow"), dict(level="LOUD"), dict(log_path=tmp_path / "nowhere" / "x.log")):
        with pytest.raises(ValueError):
            script_logging.setup_script_logging(**{"log_path": None, **kwargs})
    assert logging.getLogger("lunarscout").handlers == before


def test_reset_removes_the_handlers_closes_the_file_and_restores_the_logger(tmp_path, caplog) -> None:
    logger = logging.getLogger("lunarscout")
    original = (logger.level, logger.propagate, list(logger.handlers))
    handlers = []
    script_logging.setup_script_logging(tmp_path / "run.log")
    handlers = [h for h in logger.handlers if getattr(h, script_logging._MARK, False)]
    assert len(handlers) == 2 and not logger.propagate
    script_logging.reset_script_logging()
    assert (logger.level, logger.propagate, list(logger.handlers)) == original
    assert logger.level == ORIGINAL_LEVEL == logging.NOTSET
    assert all(getattr(h, "stream", None) is None or h.stream is None or getattr(h.stream, "closed", True) for h in handlers
               if isinstance(h, logging.FileHandler))
    with caplog.at_level(logging.WARNING, logger=LIBRARY):
        logging.getLogger(LIBRARY).warning("seen by pytest again")
    assert "seen by pytest again" in caplog.text


def test_importing_the_module_does_not_import_rich() -> None:
    code = "import sys, lunarscout._script_logging; print('rich' in sys.modules)"
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(root / "src"), "PATH": ""}, cwd=root)
    assert done.returncode == 0 and done.stdout.strip() == "False"
