"""Command-line options for the scripts in ``scripts/`` (private).

A script keeps its settings as constants at the top of the file. Every one that can vary is also an *option*: the
command line gives ``--constant-name value`` (the constant's name in lower case with hyphens), and the constant is the
default. Each script defines

* ``OPTIONS``: a tuple of :class:`Option` (the constant's name, how to read it, a help line);
* ``default_options()``: ``{option: default}`` read from the constants *when it is called*, so editing the file (or a
  test or the driver patching a constant) changes the defaults;
* a procedure, ``run_<script>(**options)``, whose parameters are exactly those option names in lower case, so another
  program can load the script and call it with the values it wants (``scripts/run_safe_haven.py`` does);
* ``build_parser()`` and ``main(argv=())``: ``main`` reads the command line (none: the defaults), sets up logging and
  calls the procedure. ``main()`` with no argument never reads ``sys.argv``; the ``__main__`` block passes it.

Structured settings (``FloodSettings``, ``WindowParameters``) are one option group each, with one flag per field
(:func:`add_flood_settings`, :func:`add_window_parameters`).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._script_logging import COLOR_MODES, reset_script_logging, setup_script_logging

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _path(text: str) -> Path:
    return Path(text).expanduser()


def _optional(convert: Callable[[str], Any]) -> Callable[[str], Any]:
    def read(text: str):
        return None if text.strip().lower() in ("none", "") else convert(text)

    return read


def _pair(text: str) -> tuple[int, int]:
    try:
        row, column = text.split(",")
        return int(row), int(column)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not 'row,column'") from error


#: How an option's value is read: the key of ``Option.kind``.
KINDS: dict[str, Callable[[str], Any]] = {
    "path": _path, "optional_path": _optional(_path), "str": str, "optional_str": _optional(str), "int": int,
    "optional_int": _optional(int), "float": float, "optional_float": _optional(float), "pair": _pair, "id": int,
}


@dataclass(frozen=True)
class Option:
    """One option of a script: the constant ``name`` (``"DEM_PATH"``), how to read it (a key of :data:`KINDS`,
    ``"bool"`` for a ``--flag/--no-flag`` pair, ``"pairs"`` / ``"ids"`` for a list that may be ``None``, or ``"strs"``
    for a tuple of words), a help line
    and, for a fixed set of values, ``choices``."""

    name: str
    kind: str
    help: str = ""
    choices: tuple | None = None

    @property
    def flag(self) -> str:
        return "--" + self.name.lower().replace("_", "-")

    @property
    def key(self) -> str:
        return self.name.lower()


def _show(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], tuple):
        return " ".join(f"{a},{b}" for a, b in value)
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value) or "none"
    return "none" if value is None else str(value)


def add_options(parser: argparse.ArgumentParser, options: Sequence[Option], defaults: Mapping[str, Any]) -> None:
    """Add one flag per option, with ``defaults[option.key]`` as its default (shown in the help)."""
    for option in options:
        default = defaults[option.key]
        text = f"{option.help} (default: {_show(default)})".strip()
        if option.kind == "bool":
            parser.add_argument(option.flag, dest=option.key, action=argparse.BooleanOptionalAction, default=default,
                                help=text)
        elif option.kind == "strs":
            parser.add_argument(option.flag, dest=option.key, nargs="*", type=str, default=list(default),
                                choices=option.choices, metavar="VALUE", help=text)
        elif option.kind in ("pairs", "ids"):
            reader = _pair if option.kind == "pairs" else int
            parser.add_argument(option.flag, dest=option.key, nargs="*", type=reader, default=default, metavar="VALUE",
                                help=text + "; give no value for none")
        else:
            parser.add_argument(option.flag, dest=option.key, type=KINDS[option.kind], default=default,
                                choices=option.choices, help=text, metavar=option.kind.upper().replace("_", "-")
                                if option.choices is None else None)


def options_from(args: argparse.Namespace, options: Sequence[Option]) -> dict[str, Any]:
    """``{option key: value}`` of the parsed ``args``; an empty list (``--cells`` with no value) is ``None``."""
    values: dict[str, Any] = {}
    for option in options:
        value = getattr(args, option.key)
        if option.kind in ("pairs", "ids") and value is not None and len(value) == 0:
            value = None
        if option.kind == "strs":
            value = tuple(value)
        values[option.key] = value
    return values


def defaults_from(options: Sequence[Option], constants: Mapping[str, Any]) -> dict[str, Any]:
    """``{option key: constant}`` for every option, read from ``constants`` (the script's ``globals()``) now."""
    return {option.key: constants[option.name] for option in options}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def add_logging_options(parser: argparse.ArgumentParser, constants: Mapping[str, Any]) -> None:
    """``--log-path``, ``--color`` and ``--log-level``, defaulting to the script's ``LOG_PATH``, ``COLOR`` and
    ``LOG_LEVEL`` constants (a ``LOG_PATH`` of None writes next to the output)."""
    parser.add_argument("--log-path", type=_optional(_path), default=constants["LOG_PATH"], metavar="PATH",
                        help=f"plain-text log file (default: {_show(constants['LOG_PATH'])}: next to the output)")
    parser.add_argument("--color", choices=COLOR_MODES, default=constants["COLOR"],
                        help=f"console colour (default: {constants['COLOR']})")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default=constants["LOG_LEVEL"],
                        help=f"log level (default: {constants['LOG_LEVEL']})")


def run_with_logging(args: argparse.Namespace, *, default_log_path: Path, logger: logging.Logger, label: str,
                     procedure: Callable[[], int]) -> int:
    """Set up logging from ``args``, run ``procedure()``, log a failure as ``"<label> stopped: ..."`` and re-raise it,
    and always tear the logging down."""
    log_path = args.log_path if args.log_path is not None else default_log_path
    setup_script_logging(log_path, color=args.color, level=args.log_level)
    try:
        return procedure()
    except Exception as error:
        logger.error("%s stopped: %s: %s", label, type(error).__name__, error)
        raise
    finally:
        reset_script_logging()


# ---------------------------------------------------------------------------
# Structured settings
# ---------------------------------------------------------------------------

#: (field, kind, help) of ``FloodSettings``: what the rover's flood and everything built on it depends on.
FLOOD_FIELDS = (
    ("max_days", "float", "the flood and the lit endurance end here (days)"),
    ("moves_per_step", "int", "one-pixel moves the rover makes in each step"),
    ("connectivity", "int", "4 or 8 neighbours per move"),
    ("max_slope_deg", "float", "the rover only enters or stays on pixels whose Horn slope is at most this (degrees)"),
    ("psr_small_max_m2", "float", "PSR size classes: small is up to this area (m2); 400 is one pixel at 20 m"),
    ("psr_medium_max_m2", "float", "PSR size classes: medium is up to this area (m2), large above"),
    ("reach_hours", "float", "the reach in pixels is counted this many hours after the boundary"),
    ("edge_rule", "str", "which seeds a flood may arrive at: next_cycle or any_later"),
    ("unit_cache_mib", "int", "lit-unit cache per worker (MiB)"),
)

#: (field, kind, help) of ``WindowParameters`` that the scripts expose.
WINDOW_FIELDS = (
    ("flood_step_minutes", "int", "the light store's step in minutes"),
    ("survival_limit_hours", "float", "the longest hibernation (outage in shadow) a haven may have (hours)"),
    ("operational_hours", "float", "operating time wanted on each side of a hibernation (hours)"),
    ("moves_per_step", "int", "one-pixel moves the rover makes in each step"),
    ("max_seeds_per_cell_month", "int", "seeds per 5.12 km cell and month"),
    ("seed_separation_m", "float", "seeds are at least this far apart unless their outage timing differs (m)"),
    ("seed_timing_tolerance_hours", "float", "outage timing difference that lets seeds be closer (hours)"),
    ("haven_neighbourhood_m", "float", "radius of the neighbourhood checked for other havens (m)"),
    ("min_nearby_haven_fraction", "float", "the share of that neighbourhood that must be havens"),
    ("boundary_search_hours", "float", "how far to look for the lit boundary around an outage (hours)"),
    ("operating_sunlight_fraction", "float", "Sun fraction at or above which the rover operates"),
    ("cell_patches", "int", "cell size in patches of 128 pixels"),
    ("degenerate_outage_window_hours", "float", "the one-sample outage rule where Earth never sets (hours; 0 disables)"),
)


def _fields(fields, defaults) -> list[tuple[str, str, str, Any]]:
    return [(name, kind, text, getattr(defaults, name)) for name, kind, text in fields]


def _add_group(parser, title: str, rows) -> None:
    group = parser.add_argument_group(title)
    for name, kind, text, default in rows:
        group.add_argument("--" + name.replace("_", "-"), dest=name, type=KINDS[kind], default=default,
                           help=f"{text} (default: {default})", metavar=kind.upper())


def add_flood_settings(parser: argparse.ArgumentParser, defaults) -> None:
    """One flag per ``FloodSettings`` field, defaulting to the fields of ``defaults`` (a ``FloodSettings``)."""
    _add_group(parser, "rover flood settings (must be the same in stages 3 and 4 and the chain maps)",
               _fields(FLOOD_FIELDS, defaults))


def flood_settings_from(args: argparse.Namespace, defaults):
    """The ``FloodSettings`` the flags of ``args`` give (``defaults`` for any flag the parser did not have)."""
    return dataclasses.replace(defaults, **{name: getattr(args, name) for name, _k, _t in FLOOD_FIELDS
                                            if hasattr(args, name)})


def add_window_parameters(parser: argparse.ArgumentParser, defaults) -> None:
    """One flag per exposed ``WindowParameters`` field, defaulting to the fields of ``defaults``."""
    _add_group(parser, "seed rules (stage 2)", _fields(WINDOW_FIELDS, defaults))


def window_parameters_from(args: argparse.Namespace, defaults):
    """The ``WindowParameters`` the flags of ``args`` give (``defaults`` for any flag the parser did not have)."""
    return dataclasses.replace(defaults, **{name: getattr(args, name) for name, _k, _t in WINDOW_FIELDS
                                            if hasattr(args, name)})
