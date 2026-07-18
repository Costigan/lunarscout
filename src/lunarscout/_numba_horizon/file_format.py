"""Production horizon tile naming, validation, compression, and staged writes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
import threading
import time
from typing import BinaryIO
from uuid import uuid4

import numpy as np
import numpy.typing as npt


PATCH_SIZE = 128
AZIMUTH_COUNT = 1440
PIXEL_COUNT = PATCH_SIZE * PATCH_SIZE
TOTAL_SAMPLES = PIXEL_COUNT * AZIMUTH_COUNT
RAW_FILE_BYTES = TOTAL_SAMPLES * np.dtype("<f4").itemsize
MAX_COMPRESSED_BLOCK_BYTES = 2 * AZIMUTH_COUNT
_SHORT_SCALE = np.float32(32767.0) / np.float32(50.0)
_ELEVATION_SCALE = np.float32(50.0) / np.float32(32767.0)
_COMPILED_ENCODER = None
_COMPILED_DECODER = None
_COMPILED_DECODER_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class HorizonReadTimings:
    """Wall-clock boundaries for one complete horizon-tile read."""

    file_read_seconds: float
    decompression_seconds: float


def _encode_horizons_python(
    degrees: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]:
    """Reference implementation of the C# HorizonCompressor encoder."""
    encoded = np.empty((degrees.shape[0], MAX_COMPRESSED_BLOCK_BYTES), dtype=np.uint8)
    lengths = np.empty(degrees.shape[0], dtype=np.uint16)
    for horizon_index, horizon in enumerate(degrees):
        output_index = 0
        previous = 0
        for sample_index, raw_value in enumerate(horizon):
            value = float(raw_value)
            value = max(-50.0, min(50.0, value))
            scaled = float(np.float32(value) * _SHORT_SCALE)
            quantized = int(np.floor(scaled + 0.5) if scaled >= 0 else np.ceil(scaled - 0.5))
            quantized = max(-32767, min(32767, quantized))
            if sample_index == 0:
                previous = quantized
                encoded[horizon_index, output_index] = (quantized >> 8) & 0xFF
                encoded[horizon_index, output_index + 1] = quantized & 0xFF
                output_index += 2
                continue
            delta = max(-16384, min(16383, quantized - previous))
            previous = ((previous + delta + 32768) % 65536) - 32768
            if 0 <= delta < 63:
                encoded[horizon_index, output_index] = delta
                output_index += 1
            elif -64 <= delta < 0:
                encoded[horizon_index, output_index] = delta & 0x7F
                output_index += 1
            else:
                encoded[horizon_index, output_index] = ((delta >> 8) & 0xFF) | 0x80
                encoded[horizon_index, output_index + 1] = delta & 0xFF
                output_index += 2
        lengths[horizon_index] = output_index
    return encoded, lengths


def _encode_horizons(
    degrees: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]:
    """Encode pixel-major horizons, compiling the CPU loop lazily when available."""
    global _COMPILED_ENCODER
    if _COMPILED_ENCODER is None:
        try:
            from numba import njit
        except ImportError:
            _COMPILED_ENCODER = _encode_horizons_python
        else:
            # Defining this lazily keeps an ordinary ``import lunarscout``
            # independent of Numba while making production compression practical.
            _COMPILED_ENCODER = njit(cache=False)(_encode_horizons_python)
    return _COMPILED_ENCODER(degrees)


def _normalize_tile(
    degrees: npt.ArrayLike,
    *,
    valid_width: int = PATCH_SIZE,
    valid_height: int = PATCH_SIZE,
) -> npt.NDArray[np.float32]:
    if not 1 <= valid_width <= PATCH_SIZE or not 1 <= valid_height <= PATCH_SIZE:
        raise ValueError("valid tile dimensions must be between 1 and 128")
    source = np.asarray(degrees, dtype=np.float32)
    expected = (valid_width * valid_height, AZIMUTH_COUNT)
    full_shape = (PIXEL_COUNT, AZIMUTH_COUNT)
    if source.shape not in (expected, full_shape):
        raise ValueError(f"degrees must have shape {expected} or {full_shape}")
    if np.any(np.isnan(source)):
        raise ValueError("degrees must not contain NaN values")
    if valid_width == PATCH_SIZE and valid_height == PATCH_SIZE:
        return np.ascontiguousarray(source)
    # Existing readers require a complete 128x128 tile. Pixels outside the DEM
    # are never addressable and use the compressor's minimum representable angle.
    padded = np.full((PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT), -50.0, dtype=np.float32)
    if source.shape == full_shape:
        valid = source.reshape(PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT)[
            :valid_height, :valid_width
        ]
    else:
        valid = source.reshape(valid_height, valid_width, AZIMUTH_COUNT)
    padded[:valid_height, :valid_width] = valid
    return padded.reshape(PIXEL_COUNT, AZIMUTH_COUNT)


def _write_uncompressed(handle: BinaryIO, degrees: npt.NDArray[np.float32]) -> None:
    handle.write(degrees.astype("<f4", copy=False).tobytes(order="C"))


def _write_compressed(handle: BinaryIO, degrees: npt.NDArray[np.float32]) -> None:
    encoded, lengths = _encode_horizons(degrees)
    for index, length_value in enumerate(lengths):
        length = int(length_value)
        handle.write(struct.pack("<H", length))
        handle.write(encoded[index, :length].tobytes())


def _decode_compressed_payload_into_python(
    payload: npt.NDArray[np.uint8],
    output: npt.NDArray[np.float32],
) -> None:
    """Decode and structurally validate a complete C# ``.cbin`` payload."""
    offset = 0
    payload_size = payload.size
    for pixel_index in range(PIXEL_COUNT):
        if offset + 2 > payload_size:
            raise ValueError("compressed horizon ended while reading a block length")
        length = int(payload[offset]) | (int(payload[offset + 1]) << 8)
        offset += 2
        if length < 2 or length > MAX_COMPRESSED_BLOCK_BYTES:
            raise ValueError("compressed horizon block length is invalid")
        block_end = offset + length
        if block_end > payload_size:
            raise ValueError("compressed horizon ended inside a block")

        accumulator = (int(payload[offset]) << 8) | int(payload[offset + 1])
        if accumulator >= 32768:
            accumulator -= 65536
        offset += 2
        output[pixel_index, 0] = np.float32(accumulator) * _ELEVATION_SCALE
        sample_index = 1
        while offset < block_end and sample_index < AZIMUTH_COUNT:
            first = int(payload[offset])
            offset += 1
            if first & 0x80:
                if offset >= block_end:
                    raise ValueError("compressed horizon ended inside a two-byte delta")
                high = ((first << 1) & 0x80) | (first & 0x7F)
                delta = (high << 8) | int(payload[offset])
                offset += 1
                if delta >= 32768:
                    delta -= 65536
            else:
                delta = first & 0x7F
                if delta & 0x40:
                    delta -= 0x80
            accumulator = ((accumulator + delta + 32768) % 65536) - 32768
            output[pixel_index, sample_index] = (
                np.float32(accumulator) * _ELEVATION_SCALE
            )
            sample_index += 1
        if sample_index != AZIMUTH_COUNT or offset != block_end:
            raise ValueError("compressed horizon block did not decode to 1440 samples")
    if offset != payload_size:
        raise ValueError("compressed horizon contains trailing bytes")


def _decode_compressed_payload_python(
    payload: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    output = np.empty((PIXEL_COUNT, AZIMUTH_COUNT), dtype=np.float32)
    _decode_compressed_payload_into_python(payload, output)
    return output


def _decode_compressed_payload(
    payload: npt.NDArray[np.uint8],
    output: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.float32]:
    """Decode a complete payload, compiling the bounded loop when available."""
    global _COMPILED_DECODER
    if _COMPILED_DECODER is None:
        with _COMPILED_DECODER_LOCK:
            if _COMPILED_DECODER is None:
                try:
                    from numba import njit
                except ImportError:
                    _COMPILED_DECODER = _decode_compressed_payload_into_python
                else:
                    # The bounded downstream pipeline decompresses ahead while
                    # the caller drives CUDA and durable output. Releasing the
                    # GIL permits overlap without changing decoder arithmetic.
                    _COMPILED_DECODER = njit(cache=False, nogil=True)(
                        _decode_compressed_payload_into_python
                    )
    if output is None:
        output = np.empty((PIXEL_COUNT, AZIMUTH_COUNT), dtype=np.float32)
    _COMPILED_DECODER(payload, output)
    return output


def read_horizon_tile_with_timings(
    path: str | Path,
    *,
    output: npt.NDArray[np.float32] | None = None,
) -> tuple[npt.NDArray[np.float32], HorizonReadTimings]:
    """Read one horizon tile and return separate file/decompression timings."""
    if output is not None:
        if (
            output.dtype != np.float32
            or output.shape
            != (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT)
            or not output.flags.c_contiguous
            or not output.flags.writeable
        ):
            raise ValueError(
                "output must be a writable contiguous float32 (128, 128, 1440) array"
            )
        flat_output = output.reshape(PIXEL_COUNT, AZIMUTH_COUNT)
    else:
        flat_output = None
    candidate = Path(path)
    suffix = candidate.suffix.lower()
    read_started = time.perf_counter()
    if suffix == ".bin":
        try:
            size = candidate.stat().st_size
            if size not in (RAW_FILE_BYTES, RAW_FILE_BYTES + 28):
                raise ValueError("uncompressed horizon file size is invalid")
            with candidate.open("rb") as handle:
                payload = handle.read(RAW_FILE_BYTES)
        except OSError as exc:
            raise ValueError(f"unable to read horizon tile: {candidate}") from exc
        file_read_seconds = time.perf_counter() - read_started
        decode_started = time.perf_counter()
        source = np.frombuffer(payload, dtype="<f4", count=TOTAL_SAMPLES)
        if flat_output is None:
            values = source.copy()
        else:
            np.copyto(flat_output.reshape(-1), source)
            values = flat_output
    elif suffix == ".cbin":
        try:
            payload = np.frombuffer(candidate.read_bytes(), dtype=np.uint8)
        except OSError as exc:
            raise ValueError(f"unable to read horizon tile: {candidate}") from exc
        file_read_seconds = time.perf_counter() - read_started
        decode_started = time.perf_counter()
        values = _decode_compressed_payload(payload, flat_output)
    else:
        raise ValueError("horizon tile must have a .bin or .cbin extension")
    decompression_seconds = time.perf_counter() - decode_started
    result = values.reshape(PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT)
    return result, HorizonReadTimings(
        file_read_seconds=file_read_seconds,
        decompression_seconds=decompression_seconds,
    )


def read_horizon_tile(path: str | Path) -> npt.NDArray[np.float32]:
    """Read one complete production horizon tile as ``float32[y, x, azimuth]``."""
    values, _timings = read_horizon_tile_with_timings(path)
    return values


class HorizonTileStore:
    """Private Python equivalent of the production C# ``HorizonTileStore``."""

    def __init__(self, root_directory: str | Path, *, read_legacy_flat_files: bool = True):
        self.root = Path(root_directory)
        if not str(self.root):
            raise ValueError("horizon root directory must be provided")
        self.read_legacy_flat_files = bool(read_legacy_flat_files)

    @staticmethod
    def _elevation_decimeters(observer_elevation_m: float) -> int:
        # int() truncates toward zero, matching the C# float-to-int cast.
        value = int(float(observer_elevation_m) * 10.0)
        if not 0 <= value <= 999:
            raise ValueError("observer elevation must fit the D3 filename field")
        return value

    @staticmethod
    def _coordinate(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 99999:
            raise ValueError("horizon tile coordinates must fit the D5 filename field")
        return value

    def build_file_name(
        self, tile_y: int, tile_x: int, observer_elevation_m: float, *, compress: bool
    ) -> str:
        y = self._coordinate(tile_y)
        x = self._coordinate(tile_x)
        elevation = self._elevation_decimeters(observer_elevation_m)
        extension = ".cbin" if compress else ".bin"
        return f"horizon_{y:05d}_{x:05d}_{elevation:03d}{extension}"

    def build_path(
        self, tile_y: int, tile_x: int, observer_elevation_m: float, *, compress: bool
    ) -> Path:
        name = self.build_file_name(
            tile_y, tile_x, observer_elevation_m, compress=compress
        )
        return self.root / f"{tile_y:05d}" / name

    def candidate_paths(
        self, tile_y: int, tile_x: int, observer_elevation_m: float
    ) -> tuple[Path, ...]:
        compressed = self.build_file_name(
            tile_y, tile_x, observer_elevation_m, compress=True
        )
        raw = self.build_file_name(tile_y, tile_x, observer_elevation_m, compress=False)
        candidates = [self.root / f"{tile_y:05d}" / compressed, self.root / f"{tile_y:05d}" / raw]
        if self.read_legacy_flat_files:
            candidates.extend((self.root / compressed, self.root / raw))
        return tuple(candidates)

    @staticmethod
    def is_complete(path: str | Path) -> bool:
        candidate = Path(path)
        try:
            if candidate.suffix.lower() == ".bin":
                # The second size is the legacy payload plus seven metadata floats.
                return candidate.stat().st_size in (RAW_FILE_BYTES, RAW_FILE_BYTES + 28)
            if candidate.suffix.lower() != ".cbin" or candidate.stat().st_size <= 0:
                return False
            with candidate.open("rb") as handle:
                for _ in range(PIXEL_COUNT):
                    prefix = handle.read(2)
                    if len(prefix) != 2:
                        return False
                    length = int.from_bytes(prefix, "little")
                    if not 1 <= length <= MAX_COMPRESSED_BLOCK_BYTES:
                        return False
                    if len(handle.read(length)) != length:
                        return False
                return handle.read(1) == b""
        except (OSError, ValueError):
            return False

    def find_existing_path(
        self,
        tile_y: int,
        tile_x: int,
        observer_elevation_m: float,
        *,
        require_complete: bool = True,
    ) -> Path | None:
        for path in self.candidate_paths(tile_y, tile_x, observer_elevation_m):
            if path.is_file() and (not require_complete or self.is_complete(path)):
                return path
        return None

    def read(
        self, tile_y: int, tile_x: int, observer_elevation_m: float
    ) -> npt.NDArray[np.float32] | None:
        """Read the preferred complete tile, or return ``None`` when absent."""
        path = self.find_existing_path(tile_y, tile_x, observer_elevation_m)
        return None if path is None else read_horizon_tile(path)

    def write(
        self,
        tile_y: int,
        tile_x: int,
        observer_elevation_m: float,
        degrees: npt.ArrayLike,
        *,
        compress: bool,
        valid_width: int = PATCH_SIZE,
        valid_height: int = PATCH_SIZE,
    ) -> Path:
        final_path = self.build_path(
            tile_y, tile_x, observer_elevation_m, compress=compress
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        extension = ".cbin" if compress else ".bin"
        staging = final_path.parent / (
            f"{final_path.stem}.{uuid4().hex}.tmp{extension}"
        )
        normalized = _normalize_tile(
            degrees, valid_width=valid_width, valid_height=valid_height
        )
        try:
            with staging.open("wb", buffering=1024 * 1024) as handle:
                if compress:
                    _write_compressed(handle, normalized)
                else:
                    _write_uncompressed(handle, normalized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staging, final_path)
        except BaseException:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return final_path
