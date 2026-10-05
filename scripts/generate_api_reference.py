#!/usr/bin/env python3
"""Generate the public API reference from checkout signatures and docstrings.

Run with the project's Python environment. No public functions, properties,
CUDA probes, or SPICE operations are executed. Only the output file is written.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
from pathlib import Path
import re
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = ("lunarscout", "lunarscout.cuda", "lunarscout.spice",
              "lunarscout.trajectory", "lunarscout.map_algebra")


def public_members(module: types.ModuleType) -> list[tuple[str, object]]:
    """Honor curated exports; otherwise select package-owned public callables."""
    exports = getattr(module, "__all__", None)
    names = sorted(exports if exports is not None else vars(module))
    result = []
    for name in names:
        if name.startswith("_"):
            continue
        value = getattr(module, name)
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            continue
        if not getattr(value, "__module__", "").startswith("lunarscout."):
            continue
        result.append((name, value))
    return result


def signature(value: object) -> inspect.Signature | None:
    try:
        return inspect.signature(value)
    except (TypeError, ValueError):
        return None


def documented_parameters(doc: str) -> set[str]:
    """Find explicit NumPy, Google, and reST parameter entries.

    This intentionally does not infer documentation from prose or references
    to another function. Grouped NumPy entries separated by commas or slashes
    are supported.
    """
    names: set[str] = set()
    in_parameters = False
    lines = doc.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {"Parameters", "Other Parameters", "Args:", "Arguments:", "Keyword Args:"}:
            in_parameters = True
            continue
        if index + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[index + 1].strip()):
            in_parameters = stripped in {"Parameters", "Other Parameters"}
        if stripped in {"Returns:", "Raises:", "Notes:", "Examples:"}:
            in_parameters = False
        rst = re.match(r"\s*:param\s+(?:\S+\s+)?(\w+)\s*:", line)
        if rst:
            names.add(rst.group(1))
        if in_parameters and ":" in line and (not line.startswith(" ") or line.startswith("    ")):
            entry = line.split(":", 1)[0].strip()
            entry = re.sub(r"\s*\([^)]*\)$", "", entry)
            parts = re.split(r"\s*[,/]\s*", entry)
            if all(re.fullmatch(r"\*{0,2}[A-Za-z_]\w*", part) for part in parts):
                names.update(part.lstrip("*") for part in parts)
    return names


def fenced(text: str, language: str) -> str:
    runs = re.findall(r"`+", text)
    fence = "`" * max(3, 1 + max(map(len, runs), default=0))
    return f"{fence}{language}\n{text}\n{fence}\n"


def render_entry(name: str, value: object, level: int = 3) -> str:
    lines = [f"{'#' * level} `{name}`\n"]
    sig = signature(value)
    if sig is not None:
        # Default callable/sentinel reprs may contain process-specific addresses.
        display = re.sub(r" at 0x[0-9a-fA-F]+", "", str(sig))
        lines.append(fenced(name + display, "python"))
    else:
        lines.append("Signature unavailable through Python inspection.\n")
    source = inspect.unwrap(value)
    try:
        path = Path(inspect.getsourcefile(source)).resolve().relative_to(ROOT)
        line = inspect.getsourcelines(source)[1]
    except (TypeError, OSError, ValueError):
        pass
    else:
        lines.append(f"Source: [{path.as_posix()}](../{path.as_posix()}#L{line}).\n")
    # Avoid implicitly borrowing an inherited class docstring.
    doc = inspect.cleandoc(getattr(value, "__doc__", None) or "")
    if doc:
        lines.append(fenced(doc, "text"))
    else:
        lines.append("**Documentation gap:** no source docstring.\n")
    if sig is not None:
        documented = documented_parameters(doc)
        missing = [p for p in sig.parameters if p not in {"self", "cls"} and p not in documented]
        if missing:
            lines.append("**Parameters without explicit entries in this docstring:** "
                         + ", ".join(f"`{p}`" for p in missing) + ".\n")
    return "\n".join(lines)


def render_class(name: str, cls: type) -> str:
    lines = [render_entry(name, cls)]
    if dataclasses.is_dataclass(cls):
        lines.append("Dataclass fields (types and constructor defaults appear above):\n")
        lines.extend(f"- `{field.name}`" for field in dataclasses.fields(cls))
        lines.append("")
    # Read dictionaries directly: descriptors and properties must not execute.
    members = {}
    for base in reversed(cls.__mro__):
        if getattr(base, "__module__", "").startswith("lunarscout."):
            members.update(vars(base))
    for member_name, member in sorted(members.items()):
        if member_name.startswith("_"):
            continue
        if isinstance(member, property):
            lines.append(render_entry(name + "." + member_name, member.fget, 4))
            lines.append("This member is a property.\n")
        else:
            if isinstance(member, (classmethod, staticmethod)):
                member = member.__func__
            if inspect.isfunction(member):
                lines.append(render_entry(name + "." + member_name, member, 4))
    return "\n".join(lines)


def build_reference() -> str:
    # Always document this checkout rather than an installed distribution.
    sys.path.insert(0, str(ROOT / "src"))
    lines = ["# Lunarscout API Reference\n",
             "Generated by `scripts/generate_api_reference.py`. Edit source docstrings, "
             "then regenerate this file; do not edit it by hand.\n",
             "This reference contains public functions, classes, public methods, "
             "properties, and dataclass fields. Private members and imported third-party "
             "helpers are excluded. Signatures come from Python inspection (including "
             "wrapped functions); docstrings are preserved as text.\n",
             "Parameter gap notices identify missing explicit NumPy, Google, or reST "
             "entries in the local docstring. Prose and cross-references may describe "
             "some of these arguments. This generated reference does not establish "
             "that the source documentation is complete.\n",
             "For workflows and shared contracts, see the [User Guide](USER_GUIDE.md).\n",
             "## Namespaces\n"]
    lines.extend(f"- [{name}](#{name.replace('.', '-')})" for name in NAMESPACES)
    lines.append("")
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        lines.append(f"<a id=\"{namespace.replace('.', '-')}\"></a>\n\n## `{namespace}`\n")
        for name, value in public_members(module):
            qualified = namespace + "." + name
            lines.append(render_class(qualified, value) if inspect.isclass(value)
                         else render_entry(qualified, value))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "API_REFERENCE.md")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the output is missing or stale; write nothing.")
    args = parser.parse_args()
    content = build_reference()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
            print(f"API reference is missing or stale: {args.output}", file=sys.stderr)
            return 1
        print(f"API reference is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Complete rendering before writing; atomic replacement preserves old output
    # if generation fails and avoids exposing a partially written reference.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Generated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
