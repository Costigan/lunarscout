"""Stage 0 of the safe-haven analysis: the thresholded-lighting store (private).

Stage 0 lights every 128 x 128 patch once, for the whole time range, and keeps three yes/no
planes per time step so that later stages never read the horizons again:

* ``lit_partial``   Sun visible fraction at least the shadow threshold (0.2); its complement is
  the power-negative shadow of stage 1;
* ``lit_full``      Sun visible fraction at least the operating fraction (1.0) less the stage-1
  fuzz of 1e-6;
* ``earth_visible`` Earth-centre elevation above the pixel's own horizon at least the outage
  threshold (2 degrees); its complement is the stage-1 outage.

Each plane is packed as the stage-3 flood packs a lit step: ``(128, 2)`` 64-bit words, bit ``k``
of word ``w`` in a row being column ``64 * w + k`` (``safe_haven_flood_packed``). A pixel outside
the valid part of a patch has all three bits clear.

One file holds one patch. The layout, little-endian throughout::

    magic         8 bytes    b"LSLITE01"
    header size   uint64     bytes of the JSON header, before padding
    header        JSON, padded with spaces to a multiple of 8 bytes
    offsets       int64[chunk_count + 1], absolute byte offsets of the chunks, then the file size
    chunks        one compressed block per run of ``steps_per_chunk`` steps

A chunk is the planes of its steps, array shape ``(3, steps, 128, 2)`` of ``<u8``, compressed on
its own so that a reader needs only the chunks of the days it asks for: it slices the memory-mapped
file between two offsets and decompresses those. Measured on eight real patches (8 years at 2 hours),
zstd level 3 on 8-day chunks stores a patch in 3.6 MB on average (60 times smaller than the raw
215 MB); storing each step as the exclusive-or of the one before made zstd files about 10% larger,
so the planes are stored as they are. The header names the codec, the time grid, the thresholds and
the patch, and a reader that is asked for different ones refuses the file. A file is written to a staging name and renamed when
complete, so a failure never leaves a partial file or damages an earlier one.

The producer, :func:`light_patch_planes`, is the CPU lightmap followed by the three thresholds in
compiled loops that release the GIL, or, when given a session that has a ``light_patch_planes``
method (``safe_haven_lit_cuda.LitCudaSession``), that session's CUDA route. Both use the stage-1
comparisons exactly; they can differ at pixel-steps whose value lies within a few float32 steps of
a threshold.
"""

from __future__ import annotations

import json
import mmap
import os
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from ..errors import InputError, ProductStorageError
from .file_format import PATCH_SIZE
from .safe_haven_flood import OPERATING_FUZZ
from .safe_haven_litstore import _threshold_and_pack

FORMAT = "lunarscout-light-store-v1"
MAGIC = b"LSLITE01"
PLANE_NAMES = ("lit_partial", "lit_full", "earth_visible")
PLANE_COUNT = len(PLANE_NAMES)
WORDS_PER_ROW = PATCH_SIZE // 64
_WORD = np.dtype("<u8")


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


def _split_codec(name: str) -> tuple[str, int | None]:
    base, _, level = str(name).partition("-")
    try:
        return base, (int(level) if level else None)
    except ValueError:
        raise InputError(f"unknown codec {name!r}", code="light_store_codec_unknown",
                         details={"codec": name}) from None


def _missing(base: str):
    return ProductStorageError(
        f"the {base!r} codec needs its Python package, which is not installed"
        + (' (pip install "lunarscout[lightstore]")' if base == "zstd" else ""),
        code="light_store_codec_unavailable", details={"codec": base},
    )


def codec_functions(name: str) -> tuple[Callable[[bytes], bytes], Callable[[bytes, int], bytes]]:
    """``(compress, decompress)`` for a codec name such as ``"zstd-3"``, ``"lz4"``, ``"snappy"``,
    ``"zlib-6"`` or ``"none"``. ``decompress(data, raw_size)`` needs the uncompressed size."""
    base, level = _split_codec(name)
    if base == "none":
        return bytes, lambda data, size: bytes(data)
    if base == "zlib":
        level = 6 if level is None else level
        return (lambda data: zlib.compress(data, level)), (lambda data, size: zlib.decompress(data))
    if base == "zstd":
        try:
            import zstandard
        except ImportError:
            raise _missing(base) from None
        level = 3 if level is None else level
        compressor = zstandard.ZstdCompressor(level=level, write_content_size=True)
        decompressor = zstandard.ZstdDecompressor()
        return compressor.compress, lambda data, size: decompressor.decompress(data, max_output_size=size)
    if base == "lz4":
        try:
            import lz4.block as lz4_block
        except ImportError:
            raise _missing(base) from None
        if level is None:
            return ((lambda data: lz4_block.compress(data, store_size=False)),
                    (lambda data, size: lz4_block.decompress(data, uncompressed_size=size)))
        return ((lambda data: lz4_block.compress(data, mode="high_compression", compression=level, store_size=False)),
                (lambda data, size: lz4_block.decompress(data, uncompressed_size=size)))
    if base == "snappy":
        try:
            import snappy
        except ImportError:
            raise _missing(base) from None
        return snappy.compress, lambda data, size: snappy.uncompress(data)
    raise InputError(f"unknown codec {name!r}", code="light_store_codec_unknown", details={"codec": name})


# ---------------------------------------------------------------------------
# Settings and the producer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LightStoreSettings:
    """What a store holds: the time grid, the thresholds and how chunks are encoded."""

    step_minutes: int = 120
    steps_per_chunk: int = 96                 # 8 days at two hours
    shadow_fraction: float = 0.2
    operating_fraction: float = 1.0
    earth_elevation_deg: float = 2.0
    observer_height_m: float = 0.0
    codec: str = "zstd-3"

    def __post_init__(self) -> None:
        if self.step_minutes < 1 or 1440 % self.step_minutes:
            raise InputError("the step must divide 24 hours", code="light_store_step_invalid",
                             details={"step_minutes": self.step_minutes})
        if self.steps_per_chunk < 1:
            raise InputError("steps_per_chunk must be positive", code="light_store_chunk_invalid",
                             details={"steps_per_chunk": self.steps_per_chunk})
        if not 0.0 <= self.shadow_fraction <= self.operating_fraction <= 1.0:
            raise InputError("need 0 <= shadow_fraction <= operating_fraction <= 1",
                             code="light_store_threshold_invalid")
        codec_functions(self.codec)           # an unknown or missing codec fails here, not mid-file


def patch_geometry(dem, patch_row: int, patch_column: int) -> tuple[int, int, int, int]:
    """``(tile_y, tile_x, valid_height, valid_width)`` of a patch, which must overlap the DEM."""
    tile_y, tile_x = patch_row * PATCH_SIZE, patch_column * PATCH_SIZE
    valid_height = min(PATCH_SIZE, dem.elevation_m.shape[0] - tile_y)
    valid_width = min(PATCH_SIZE, dem.elevation_m.shape[1] - tile_x)
    if valid_height < 1 or valid_width < 1 or tile_y < 0 or tile_x < 0:
        raise InputError("the patch lies outside the DEM", code="light_store_patch_outside",
                         details={"patch": [patch_row, patch_column]})
    return tile_y, tile_x, valid_height, valid_width


def plane_thresholds(settings: LightStoreSettings) -> tuple[float, float, float]:
    """``(shadow, operating, earth)``: a plane bit is set where the value is at least its threshold.
    The operating threshold carries the stage-1 fuzz."""
    return (float(settings.shadow_fraction), float(settings.operating_fraction) - OPERATING_FUZZ,
            float(settings.earth_elevation_deg))


def checked_vectors(sun_vectors_m: npt.ArrayLike, earth_vectors_m: npt.ArrayLike):
    sun = np.asarray(sun_vectors_m, dtype=np.float64)
    earth = np.asarray(earth_vectors_m, dtype=np.float64)
    if sun.ndim != 2 or sun.shape != earth.shape or sun.shape[1] != 3:
        raise InputError("the Sun and Earth vectors must both have shape (steps, 3)",
                         code="light_store_vectors_invalid",
                         details={"sun_shape": list(sun.shape), "earth_shape": list(earth.shape)})
    return sun, earth


def light_patch_planes(
    session,
    dem,
    horizons: npt.NDArray[np.float32],
    patch_row: int,
    patch_column: int,
    sun_vectors_m: npt.ArrayLike,
    earth_vectors_m: npt.ArrayLike,
    settings: LightStoreSettings = LightStoreSettings(),
) -> npt.NDArray[np.uint64]:
    """The three packed planes of one patch for the given Sun and Earth vectors.

    Returns ``(3, steps, 128, 2)`` uint64 in ``PLANE_NAMES`` order. ``session`` is a
    ``LightmapCpuSession``, or a session with its own ``light_patch_planes`` method (the CUDA
    route); the vectors are Moon-ME, one row per step. Pixels outside the DEM stay clear in every
    plane.
    """
    sun, earth = checked_vectors(sun_vectors_m, earth_vectors_m)
    tile_y, tile_x, valid_height, valid_width = patch_geometry(dem, patch_row, patch_column)
    if hasattr(session, "light_patch_planes"):
        return session.light_patch_planes(
            dem, horizons, sun, earth, settings, tile_y=tile_y, tile_x=tile_x,
            valid_height=valid_height, valid_width=valid_width)
    pack = _threshold_and_pack(True)
    shadow, operating, earth_threshold = plane_thresholds(settings)
    bits = np.zeros((PLANE_COUNT, sun.shape[0], PATCH_SIZE, WORDS_PER_ROW), dtype=np.uint64)
    done = 0
    for fractions in session.iter_patch_fraction_batches(
        dem, horizons, sun, tile_y=tile_y, tile_x=tile_x, valid_height=valid_height, valid_width=valid_width,
    ):
        stop = done + fractions.shape[0]
        pack(fractions, valid_height, valid_width, shadow, bits[0, done:stop])
        pack(fractions, valid_height, valid_width, operating, bits[1, done:stop])
        done = stop
    done = 0
    for margins in session.iter_patch_margin_batches(
        dem, horizons, earth, tile_y=tile_y, tile_x=tile_x, valid_height=valid_height, valid_width=valid_width,
    ):
        stop = done + margins.shape[0]
        pack(margins, valid_height, valid_width, earth_threshold, bits[2, done:stop])
        done = stop
    return bits


# ---------------------------------------------------------------------------
# Chunk encoding
# ---------------------------------------------------------------------------


def chunk_raw_size(steps: int) -> int:
    return PLANE_COUNT * int(steps) * PATCH_SIZE * WORDS_PER_ROW * _WORD.itemsize


def encode_chunk(planes: npt.NDArray[np.uint64], settings: LightStoreSettings) -> bytes:
    """The bytes stored for one chunk of planes, shape ``(3, steps, 128, 2)``."""
    planes = np.asarray(planes)
    if planes.ndim != 4 or planes.shape[0] != PLANE_COUNT or planes.shape[2:] != (PATCH_SIZE, WORDS_PER_ROW):
        raise InputError("a chunk must have shape (3, steps, 128, 2)", code="light_store_chunk_shape",
                         details={"shape": list(planes.shape)})
    compress, _ = codec_functions(settings.codec)
    return compress(np.ascontiguousarray(planes, dtype=_WORD).tobytes())


def decode_chunk(data: bytes | memoryview, steps: int, settings: LightStoreSettings) -> npt.NDArray[np.uint64]:
    """Inverse of :func:`encode_chunk` for a chunk of ``steps`` steps."""
    _, decompress = codec_functions(settings.codec)
    raw = decompress(bytes(data), chunk_raw_size(steps))
    if len(raw) != chunk_raw_size(steps):
        raise ProductStorageError("a chunk decompressed to the wrong size", code="light_store_corrupt",
                                  details={"expected": chunk_raw_size(steps), "actual": len(raw)})
    return np.frombuffer(raw, dtype=_WORD).reshape(PLANE_COUNT, steps, PATCH_SIZE, WORDS_PER_ROW).astype(np.uint64)


# ---------------------------------------------------------------------------
# File writer and reader
# ---------------------------------------------------------------------------


def _header_document(settings: LightStoreSettings, patch_row: int, patch_column: int,
                     first_step: int, step_count: int) -> dict:
    chunks = -(-step_count // settings.steps_per_chunk)
    return {
        "format": FORMAT, "patch_row": int(patch_row), "patch_column": int(patch_column),
        "first_step": int(first_step), "step_count": int(step_count), "chunk_count": int(chunks),
        "planes": list(PLANE_NAMES), "settings": asdict(settings),
    }


def _same_settings(left: dict, right: dict) -> bool:
    return left == right


class LightStoreWriter:
    """Writes one patch's file: ``write_chunk`` once per chunk in order, then close.

    Use as a context manager. The file appears at ``path`` only when every chunk has been written;
    an exception removes the staging file and leaves any existing file as it was.
    """

    def __init__(self, path: str | Path, settings: LightStoreSettings, *, patch_row: int, patch_column: int,
                 first_step: int, step_count: int) -> None:
        if step_count < 1:
            raise InputError("step_count must be positive", code="light_store_step_count")
        self.path = Path(path)
        if not self.path.parent.is_dir():
            raise ProductStorageError("the directory of the light-store file does not exist",
                                      code="light_store_directory_missing", details={"path": str(self.path)})
        self.settings = settings
        self.document = _header_document(settings, patch_row, patch_column, first_step, step_count)
        self._staging = self.path.with_name(self.path.name + f".partial-{os.getpid()}")
        self._written = 0
        self._offsets: list[int] = []
        self._handle = None
        self._table_at = 0

    def __enter__(self) -> "LightStoreWriter":
        header = json.dumps(self.document, sort_keys=True).encode("utf-8")
        padded = header + b" " * (-len(header) % 8)
        self._handle = self._staging.open("wb")
        try:
            self._handle.write(MAGIC)
            self._handle.write(np.uint64(len(header)).tobytes())
            self._handle.write(padded)
            self._table_at = self._handle.tell()
            self._handle.write(bytes(8 * (self.document["chunk_count"] + 1)))
        except BaseException:
            self._abandon()
            raise
        return self

    def steps_in_chunk(self, index: int) -> int:
        spc = self.settings.steps_per_chunk
        return min(spc, self.document["step_count"] - index * spc)

    def write_chunk(self, planes: npt.NDArray[np.uint64]) -> int:
        """Append the next chunk; returns the bytes stored."""
        if self._written >= self.document["chunk_count"]:
            raise InputError("every chunk has been written", code="light_store_too_many_chunks")
        steps = self.steps_in_chunk(self._written)
        planes = np.asarray(planes)
        if planes.ndim == 4 and planes.shape[1] != steps:
            raise InputError(f"chunk {self._written} needs {steps} steps", code="light_store_chunk_steps",
                             details={"expected": steps, "actual": int(planes.shape[1])})
        data = encode_chunk(planes, self.settings)
        self._offsets.append(self._handle.tell())
        self._handle.write(data)
        self._written += 1
        return len(data)

    def _abandon(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._staging.unlink(missing_ok=True)

    def __exit__(self, kind, value, trace) -> None:
        if kind is not None:
            self._abandon()
            return
        try:
            if self._written != self.document["chunk_count"]:
                raise ProductStorageError("the file was closed before every chunk was written",
                                          code="light_store_incomplete",
                                          details={"written": self._written, "expected": self.document["chunk_count"]})
            self._offsets.append(self._handle.tell())
            self._handle.seek(self._table_at)
            self._handle.write(np.asarray(self._offsets, dtype="<i8").tobytes())
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
            os.replace(self._staging, self.path)
        except BaseException:
            self._abandon()
            raise


class LightStore:
    """One patch's file, opened for reading. Use as a context manager or call :meth:`close`.

    ``read_steps(first, stop)`` returns the planes for global steps ``first <= step < stop`` as
    ``(3, stop - first, 128, 2)`` uint64, decompressing only the chunks that cover them.
    """

    def __init__(self, path: str | Path, settings: LightStoreSettings | None = None) -> None:
        self.path = Path(path)
        try:
            self._file = self.path.open("rb")
        except OSError as error:
            raise ProductStorageError(f"cannot open the light-store file {self.path}",
                                      code="light_store_unreadable", details={"path": str(self.path)}) from error
        try:
            if os.fstat(self._file.fileno()).st_size == 0:          # mmap refuses an empty file
                raise self._corrupt("the file is empty")
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._parse(settings)
        except BaseException:
            self.close()
            raise

    def _corrupt(self, message: str, **details) -> ProductStorageError:
        return ProductStorageError(f"{message}: {self.path}", code="light_store_corrupt",
                                   details={"path": str(self.path), **details})

    def _parse(self, expected: LightStoreSettings | None) -> None:
        view = memoryview(self._map)
        if len(view) < 16 or bytes(view[:8]) != MAGIC:
            raise self._corrupt("not a light-store file")
        header_size = int(np.frombuffer(view[8:16], dtype="<u8")[0])
        header_end = 16 + header_size + (-header_size % 8)
        if header_end > len(view):
            raise self._corrupt("the header is truncated")
        try:
            document = json.loads(bytes(view[16:16 + header_size]).decode("utf-8"))
            if document["format"] != FORMAT:
                raise ValueError(document["format"])
            self.settings = LightStoreSettings(**document["settings"])
            self.patch = (int(document["patch_row"]), int(document["patch_column"]))
            self.first_step = int(document["first_step"])
            self.step_count = int(document["step_count"])
            self.chunk_count = int(document["chunk_count"])
        except (ValueError, KeyError, TypeError) as error:
            raise self._corrupt("the header is unreadable") from error
        if expected is not None and self.settings != expected:
            raise ProductStorageError(
                f"the light-store file was written with different settings: {self.path}",
                code="light_store_settings_mismatch",
                details={"path": str(self.path), "file": asdict(self.settings), "expected": asdict(expected)},
            )
        table_end = header_end + 8 * (self.chunk_count + 1)
        if table_end > len(view):
            raise self._corrupt("the offset table is truncated")
        self._offsets = np.frombuffer(view[header_end:table_end], dtype="<i8").astype(np.int64)
        if (self._offsets[-1] != len(view) or self._offsets[0] < table_end
                or np.any(np.diff(self._offsets) < 0)):
            raise self._corrupt("the offsets do not match the file (truncated or damaged)",
                                file_size=len(view), last_offset=int(self._offsets[-1]))
        del view

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            try:
                self._map.close()
            except BufferError:
                pass
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "LightStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def stop_step(self) -> int:
        return self.first_step + self.step_count

    def _steps_in_chunk(self, index: int) -> int:
        spc = self.settings.steps_per_chunk
        return min(spc, self.step_count - index * spc)

    def stored_bytes(self, first: int | None = None, stop: int | None = None) -> int:
        """Compressed bytes of the chunks covering steps ``first <= step < stop`` (default all)."""
        low, high = self._chunk_span(self.first_step if first is None else first,
                                     self.stop_step if stop is None else stop)
        return int(self._offsets[high] - self._offsets[low])

    def _chunk_span(self, first: int, stop: int) -> tuple[int, int]:
        if not self.first_step <= first < stop <= self.stop_step:
            raise InputError("the steps asked for lie outside the file", code="light_store_range",
                             details={"first": first, "stop": stop, "file": [self.first_step, self.stop_step]})
        spc = self.settings.steps_per_chunk
        return (first - self.first_step) // spc, -(-(stop - self.first_step) // spc)

    def read_steps(self, first: int, stop: int) -> npt.NDArray[np.uint64]:
        low, high = self._chunk_span(int(first), int(stop))
        spc = self.settings.steps_per_chunk
        pieces = []
        for index in range(low, high):
            data = self._map[int(self._offsets[index]):int(self._offsets[index + 1])]
            pieces.append(decode_chunk(data, self._steps_in_chunk(index), self.settings))
        planes = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
        start = int(first) - self.first_step - low * spc
        return planes[:, start:start + int(stop) - int(first)]


def write_patch(path: str | Path, planes: npt.NDArray[np.uint64], settings: LightStoreSettings, *,
                patch_row: int, patch_column: int, first_step: int) -> int:
    """Write ``(3, steps, 128, 2)`` planes as one file; returns the file size in bytes."""
    planes = np.asarray(planes)
    spc = settings.steps_per_chunk
    with LightStoreWriter(path, settings, patch_row=patch_row, patch_column=patch_column,
                          first_step=first_step, step_count=planes.shape[1]) as writer:
        for start in range(0, planes.shape[1], spc):
            writer.write_chunk(planes[:, start:start + spc])
    return Path(path).stat().st_size


__all__ = [
    "FORMAT", "PLANE_NAMES", "LightStore", "LightStoreSettings", "LightStoreWriter", "codec_functions",
    "decode_chunk", "encode_chunk", "light_patch_planes", "patch_geometry", "plane_thresholds", "write_patch",
]
