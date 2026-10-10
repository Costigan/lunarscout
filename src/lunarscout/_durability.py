"""Durable-write helpers: data reaches storage before the record that claims it.

A journal, manifest, or completion marker is only trustworthy if the data it
describes was flushed first, and a rename only survives power loss once its
directory is flushed. These helpers make that ordering explicit.
"""

from __future__ import annotations

import os
from pathlib import Path


def fsync_file(path: str | Path) -> None:
    """Flush one existing file's contents to storage."""
    descriptor = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: str | Path) -> None:
    """Flush a directory's entries (creations and renames) to storage.

    Platforms or filesystems that cannot open or sync a directory are skipped.
    """
    try:
        descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_bytes_durably(path: str | Path, data: bytes) -> None:
    """Write a file and flush it before returning."""
    with open(path, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
