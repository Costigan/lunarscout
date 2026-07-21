"""Shared public-API-only support for the Lunarscout example scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import lunarscout as ls
import numpy as np


_WKT = (
    'PROJCS["ESRI:103878",'
    'GEOGCS["Moon_2000",DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["Meter",1]]'
)
_PROJ4 = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs"
)


def example_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("LUNARSCOUT_EXAMPLE_WORKSPACE", "/tmp/lunarscout_examples")),
        help="Directory for deterministic fixtures and generated outputs.",
    )
    return parser


def synthetic_georef(
    *,
    width: int = 64,
    height: int = 64,
    origin_x: float = -320.0,
    origin_y: float = 320.0,
    pixel_size: float = 10.0,
    nodata: int | float | None = -9999.0,
) -> ls.GeoReference:
    return ls.GeoReference(
        projection_wkt=_WKT,
        projection_proj4=_PROJ4,
        affine_transform=(origin_x, pixel_size, 0.0, origin_y, 0.0, -pixel_size),
        width=width,
        height=height,
        pixel_size_x=pixel_size,
        pixel_size_y=-pixel_size,
        nodata=nodata,
    )


def synthetic_dem() -> np.ndarray:
    rows, columns = np.indices((64, 64), dtype=np.float32)
    dem = 100.0 + 0.15 * columns + 0.08 * rows
    dem += 15.0 * np.exp(-((columns - 48.0) ** 2 + (rows - 18.0) ** 2) / 45.0)
    dem[12:30, 10:30] = 104.0
    dem[38:55, 35:57] = 112.0
    dem[0, 0] = -9999.0
    return dem.astype(np.float32)


def ensure_synthetic_scenario(workspace: Path) -> ls.Scenario:
    root = workspace.expanduser().resolve() / "synthetic_scenario"
    root.mkdir(parents=True, exist_ok=True)
    scenario = ls.open_scenario(root)
    if not scenario.dem_path().is_file():
        ls.write_geotiff(scenario.dem_path(), synthetic_dem(), synthetic_georef())
    return scenario


def synthetic_times() -> ls.TimeRange:
    return ls.times(
        "2027-01-01T00:00:00Z",
        "2027-01-01T05:00:00Z",
        step_hours=1,
    )


def synthetic_temporal_cube(georef: ls.GeoReference) -> ls.TemporalCube:
    times = synthetic_times()
    rows, columns = np.indices((georef.height, georef.width), dtype=np.float32)
    spatial = np.clip(0.35 + columns / max(1, georef.width - 1) * 0.45, 0.0, 1.0)
    layers = [
        np.clip(spatial + 0.08 * np.sin(index * np.pi / 3) - rows * 0.001, 0.0, 1.0)
        for index in range(times.time_count)
    ]
    return ls.TemporalCube(
        np.asarray(layers, dtype=np.float32),
        times,
        georef.with_nodata(None),
    )


def ensure_synthetic_series(workspace: Path) -> ls.TemporalGeoTiffSeries:
    scenario = ensure_synthetic_scenario(workspace)
    _dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise RuntimeError("Synthetic DEM unexpectedly lacks georeferencing.")
    path = scenario.output_path("analysis/synthetic_sun.temporal")
    if path.is_dir():
        return ls.open_temporal_cube(path)
    return ls.write_temporal_cube(
        path,
        synthetic_temporal_cube(georef),
        signal_name="sun_fraction",
        units="fraction",
        provenance={"source": "deterministic Lunarscout example"},
    )


# ---------------------------------------------------------------------------
# Synthetic horizon scenario (GPU-prebuilt data from GitHub Releases)
# ---------------------------------------------------------------------------

_DEFAULT_EXAMPLE_DATA_DIR = Path.home() / ".local/share/lunarscout/examples"

_GH_RELEASE_BASE = (
    "https://github.com/Costigan/lunarscout"
    "/releases/download"
)


def _example_data_dir() -> Path:
    env = os.environ.get("LUNARSCOUT_EXAMPLE_DATA_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "lunarscout" / "examples"
    return _DEFAULT_EXAMPLE_DATA_DIR


def _manifest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "synthetic_horizon_manifest.json"


def _load_manifest() -> dict:
    with open(_manifest_path(), "r") as fh:
        return json.load(fh)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(manifest: dict) -> str:
    override = os.environ.get("LUNARSCOUT_SYNTHETIC_HORIZON_URL")
    if override:
        return override
    explicit = manifest.get("download_url")
    if explicit:
        return explicit
    tag = manifest["release_tag"]
    name = manifest["asset_name"]
    return f"{_GH_RELEASE_BASE}/{tag}/{name}"


def ensure_synthetic_horizon_scenario(workspace: Path) -> ls.Scenario | None:
    """Obtain the synthetic 256×256 DEM and four pregenerated horizon tiles.

    Downloads from GitHub Releases on first use and caches under
    ``$LUNARSCOUT_EXAMPLE_DATA_DIR`` (default:
    ``$XDG_DATA_HOME/lunarscout/examples/``, fallback:
    ``~/.local/share/lunarscout/examples/``).  Copies into *workspace* as
    ``horizon_scenario/``.  Returns the open :class:`Scenario` on success,
    or ``None`` if the download failed (caller prints a message and exits
    cleanly).
    """
    cache_dir = _example_data_dir()
    manifest = _load_manifest()
    cache_root = cache_dir / "synthetic_horizon_scenario"
    dem_path = cache_root / "dem.tif"

    # -- Already cached? ---------------------------------------------------
    if dem_path.is_file():
        expected = manifest["files"]
        all_present = True
        for rel, expected_hash in expected.items():
            fp = cache_root / rel
            if not fp.is_file():
                all_present = False
                break
            if _sha256_file(fp) != expected_hash:
                all_present = False
                break
        if all_present:
            return _copy_to_workspace(cache_root, workspace)

    # -- Download -----------------------------------------------------------
    url = _download_url(manifest)
    expected_archive_hash = manifest["asset_sha256"]
    print(f"Downloading synthetic horizon data ({manifest['asset_name']}) ...")
    print(f"  from: {url}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / manifest["asset_name"]

    try:
        _download_with_progress(url, archive_path)
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("To skip this example, run with a real scenario instead.")
        return None

    archive_hash = _sha256_file(archive_path)
    if archive_hash != expected_archive_hash:
        print(
            f"Checksum mismatch for {manifest['asset_name']}.\n"
            f"  expected: {expected_archive_hash}\n"
            f"  got:      {archive_hash}\n"
            "The remote asset may have been updated.  Delete the cache and retry."
        )
        return None
    print("  checksum: ok")

    # -- Extract ------------------------------------------------------------
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(cache_root)

    # -- Verify extracted files ---------------------------------------------
    for rel, expected_hash in manifest["files"].items():
        fp = cache_root / rel
        actual = _sha256_file(fp)
        if actual != expected_hash:
            print(f"Checksum mismatch for extracted file: {rel}")
            return None

    print(f"  extracted to {cache_root}")
    return _copy_to_workspace(cache_root, workspace)


def _copy_to_workspace(cache_root: Path, workspace: Path) -> ls.Scenario:
    dest = workspace.expanduser().resolve() / "horizon_scenario"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(cache_root, dest, symlinks=False)
    return ls.open_scenario(str(dest))


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest*, printing progress dots."""
    tmp_path = Path(str(dest) + ".downloading")

    def _report(block_count: int, block_size: int, total_size: int) -> None:
        if block_count == 0:
            return
        downloaded = block_count * block_size
        pct = int(round(downloaded / max(1, total_size) * 100))
        if block_count % 50 == 0:
            print(f"  {pct}% ({downloaded:,} / {total_size:,} bytes)")

    urlretrieve(url, str(tmp_path), reporthook=_report)
    print("  100%")
    tmp_path.rename(dest)
