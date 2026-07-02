from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys

import numpy as np
import pytest

import lunarscout as ls
from lunarscout import spice


class FakeSpice:
    def __init__(self) -> None:
        self.furnished: list[str] = []
        self.unloaded: list[str] = []
        self.cleared = False
        self.positions: list[np.ndarray] = []
        self.utc_values: list[str] = []

    def furnsh(self, path: str) -> None:
        self.furnished.append(path)

    def unload(self, path: str) -> None:
        self.unloaded.append(path)

    def kclear(self) -> None:
        self.cleared = True

    def bodvrd(self, body: str, item: str, maxn: int):
        assert (body, item, maxn) == ("MOON", "RADII", 3)
        return 3, np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

    def utc2et(self, value: str) -> float:
        self.utc_values.append(value)
        return float(len(self.utc_values))

    def spkpos(
        self,
        target: str,
        et: float,
        frame: str,
        abcorr: str,
        observer: str,
    ):
        assert target in {"SUN", "EARTH"}
        assert frame == "MOON_ME"
        assert abcorr == "LT+S"
        assert observer == "MOON"
        if self.positions:
            return self.positions.pop(0), 0.0
        return np.asarray([1.0, 3.0, 2.0], dtype=np.float64), 0.0


class FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


@pytest.fixture()
def fake_spiceypy(monkeypatch):
    fake = FakeSpice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    spice.set_autoload_enabled(True)
    yield fake
    spice._DEFAULT_FURNISHED_PATHS.clear()
    spice._DEFAULT_KERNELS_LOADED = False
    spice.set_autoload_enabled(True)


def test_default_kernel_manifest_loads_in_order() -> None:
    entries = spice.default_kernel_entries()

    assert len(entries) == 10
    assert [entry["load_order"] for entry in entries] == sorted(
        entry["load_order"] for entry in entries
    )
    assert entries[0]["filename"] == "naif0012.tls.pc"


def test_furnish_accepts_single_path_and_disables_autoload(fake_spiceypy) -> None:
    spice.furnish("/kernels/example.tm")

    assert fake_spiceypy.furnished == ["/kernels/example.tm"]
    assert spice.autoload_enabled() is False


def test_furnish_accepts_path_list(fake_spiceypy) -> None:
    spice.furnish(["/kernels/a.bsp", "/kernels/b.tpc"], disable_autoload=False)

    assert fake_spiceypy.furnished == ["/kernels/a.bsp", "/kernels/b.tpc"]
    assert spice.autoload_enabled() is True


def test_clear_kernels_resets_bookkeeping(fake_spiceypy) -> None:
    spice.furnish("/kernels/example.tm")
    spice.clear_kernels()

    assert fake_spiceypy.cleared is True
    assert spice.default_kernels_loaded() is False
    assert spice.autoload_enabled() is True


def test_ensure_default_kernels_uses_environment_meta_kernel(
    fake_spiceypy,
    monkeypatch,
    tmp_path,
) -> None:
    meta_kernel = tmp_path / "defaults.tm"
    meta_kernel.write_text("KPL/MK\n", encoding="utf-8")
    monkeypatch.setenv("LUNARSCOUT_SPICE_META_KERNEL", str(meta_kernel))

    spice.ensure_default_kernels()
    spice.ensure_default_kernels()

    assert fake_spiceypy.furnished == [str(meta_kernel)]
    assert spice.default_kernels_loaded() is True


def test_download_default_kernels_downloads_missing_files(
    monkeypatch,
    tmp_path,
) -> None:
    urls: list[str] = []
    entries = [
        {
            "id": "example",
            "filename": "example.bsp",
            "url": "https://example.test/example.bsp",
            "kind": "spk",
            "load_order": 1,
        }
    ]

    def fake_urlopen(request, timeout):
        urls.append(request.full_url)
        return FakeHttpResponse(f"payload:{request.full_url}".encode("utf-8"))

    monkeypatch.setattr(spice, "default_kernel_entries", lambda: entries)
    monkeypatch.setattr(spice, "urlopen", fake_urlopen)

    downloaded = spice.download_default_kernels(kernel_directory=tmp_path)

    assert len(downloaded) == len(entries)
    assert urls == [entry["url"] for entry in entries]
    for entry in entries:
        assert (tmp_path / entry["filename"]).exists()


def test_download_default_kernels_uses_cache_without_overwrite(
    monkeypatch,
    tmp_path,
) -> None:
    entries = [
        {
            "id": "example",
            "filename": "example.bsp",
            "url": "https://example.test/example.bsp",
            "kind": "spk",
            "load_order": 1,
        }
    ]
    monkeypatch.setattr(spice, "default_kernel_entries", lambda: entries)
    for entry in spice.default_kernel_entries():
        (tmp_path / entry["filename"]).write_text("cached", encoding="utf-8")

    def fake_urlopen(request, timeout):
        raise AssertionError("cached kernels should not be downloaded")

    monkeypatch.setattr(spice, "urlopen", fake_urlopen)

    downloaded = spice.download_default_kernels(kernel_directory=tmp_path)

    assert downloaded == []


def test_download_default_kernels_verifies_cached_checksum(
    monkeypatch,
    tmp_path,
) -> None:
    manifest = {
        "kernels": [
            {
                "id": "example",
                "filename": "example.bsp",
                "url": "https://example.test/example.bsp",
                "kind": "spk",
                "load_order": 1,
                "sha256": (
                    "2cf24dba5fb0a30e26e83b2ac5b9e29e"
                    "1b161e5c1fa7425e73043362938b9824"
                ),
            }
        ]
    }
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        "\n".join(
            [
                "[[kernels]]",
                'id = "example"',
                'filename = "example.bsp"',
                'url = "https://example.test/example.bsp"',
                'kind = "spk"',
                "load_order = 1",
                f'sha256 = "{manifest["kernels"][0]["sha256"]}"',
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "example.bsp").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(spice, "default_kernel_manifest_path", lambda: manifest_path)

    downloaded = spice.download_default_kernels(kernel_directory=tmp_path)

    assert downloaded == []


def test_download_default_kernels_rejects_bad_cached_checksum(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        "\n".join(
            [
                "[[kernels]]",
                'id = "example"',
                'filename = "example.bsp"',
                'url = "https://example.test/example.bsp"',
                'kind = "spk"',
                "load_order = 1",
                'sha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "example.bsp").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(spice, "default_kernel_manifest_path", lambda: manifest_path)

    with pytest.raises(ls.SpiceKernelError) as exc:
        spice.download_default_kernels(kernel_directory=tmp_path)

    assert exc.value.code == "spice_kernel_checksum_mismatch"


def test_ensure_default_kernels_reports_download_failure(
    fake_spiceypy,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("LUNARSCOUT_SPICE_META_KERNEL", raising=False)
    monkeypatch.setenv("LUNARSCOUT_SPICE_KERNEL_DIR", str(tmp_path))

    def fake_urlopen(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr(spice, "urlopen", fake_urlopen)

    with pytest.raises(ls.SpiceKernelError) as exc:
        spice.ensure_default_kernels()

    assert exc.value.code == "spice_kernel_download_failed"
    assert exc.value.details["destination"].startswith(str(tmp_path))


def test_ensure_default_kernels_generates_meta_kernel_from_manifest(
    fake_spiceypy,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("LUNARSCOUT_SPICE_META_KERNEL", raising=False)
    monkeypatch.setenv("LUNARSCOUT_SPICE_KERNEL_DIR", str(tmp_path))
    entries = [
        {
            "id": "de440s",
            "filename": "de440s.bsp",
            "url": "https://example.test/de440s.bsp",
            "kind": "spk",
            "load_order": 1,
        }
    ]
    monkeypatch.setattr(spice, "default_kernel_entries", lambda: entries)
    for entry in spice.default_kernel_entries():
        (tmp_path / entry["filename"]).write_text("", encoding="utf-8")

    def fake_urlopen(request, timeout):
        raise AssertionError("cached kernels should not be downloaded")

    monkeypatch.setattr(spice, "urlopen", fake_urlopen)

    spice.ensure_default_kernels()

    assert len(fake_spiceypy.furnished) == 1
    meta_kernel = fake_spiceypy.furnished[0]
    assert meta_kernel.endswith("lunarscout_default.tm")
    text = Path(meta_kernel).read_text(encoding="utf-8")
    assert "KERNELS_TO_LOAD" in text
    assert "de440s.bsp" in text
    assert spice.default_kernels_loaded() is True


def test_unload_default_kernels_unloads_only_default_meta_kernel(
    fake_spiceypy,
    monkeypatch,
    tmp_path,
) -> None:
    meta_kernel = tmp_path / "defaults.tm"
    meta_kernel.write_text("KPL/MK\n", encoding="utf-8")
    monkeypatch.setenv("LUNARSCOUT_SPICE_META_KERNEL", str(meta_kernel))
    spice.ensure_default_kernels()

    spice.unload_default_kernels()

    assert fake_spiceypy.unloaded == [str(meta_kernel)]
    assert spice.default_kernels_loaded() is False


def test_reload_default_kernels_unloads_then_loads_default_meta_kernel(
    fake_spiceypy,
    monkeypatch,
    tmp_path,
) -> None:
    meta_kernel = tmp_path / "defaults.tm"
    meta_kernel.write_text("KPL/MK\n", encoding="utf-8")
    monkeypatch.setenv("LUNARSCOUT_SPICE_META_KERNEL", str(meta_kernel))
    spice.ensure_default_kernels()
    fake_spiceypy.furnished.clear()

    spice.reload_default_kernels()

    assert fake_spiceypy.unloaded == [str(meta_kernel)]
    assert fake_spiceypy.furnished == [str(meta_kernel)]
    assert spice.default_kernels_loaded() is True


def test_iter_times_includes_aligned_stop() -> None:
    values = list(
        ls.iter_times(
            "2027-01-01T00:00:00Z",
            "2027-01-01T03:00:00Z",
            timedelta(hours=1),
        )
    )

    assert values == [
        datetime(2027, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2027, 1, 1, 1, tzinfo=timezone.utc),
        datetime(2027, 1, 1, 2, tzinfo=timezone.utc),
        datetime(2027, 1, 1, 3, tzinfo=timezone.utc),
    ]


def test_iter_times_rejects_non_positive_step() -> None:
    with pytest.raises(ls.TimeRangeError, match="step must be positive"):
        list(
            ls.iter_times(
                "2027-01-01T00:00:00Z",
                "2027-01-01T03:00:00Z",
                timedelta(0),
            )
        )


def test_body_vectors_ned_returns_float64_time_by_three(fake_spiceypy) -> None:
    result = ls.body_vectors_ned(
        ls.LonLat(0.0, 0.0),
        "sun",
        [datetime(2027, 1, 1, tzinfo=timezone.utc)],
        ensure_kernels=False,
    )

    assert result.dtype == np.float64
    assert result.shape == (1, 3)
    np.testing.assert_allclose(result, [[2.0, 3.0, 0.0]])


def test_body_azimuth_elevation_uses_ned_convention(fake_spiceypy) -> None:
    fake_spiceypy.positions = [
        np.asarray([1.0 + math.sqrt(2.0), 1.0, 1.0], dtype=np.float64)
    ]

    result = ls.body_azimuth_elevation(
        ls.LonLat(0.0, 0.0),
        "earth",
        [datetime(2027, 1, 1, tzinfo=timezone.utc)],
        ensure_kernels=False,
    )

    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[45.0, 45.0]])


def test_body_name_validation_happens_before_spice_call(fake_spiceypy) -> None:
    with pytest.raises(ls.SpiceGeometryError) as exc:
        ls.body_vectors_ned(
            ls.LonLat(0.0, 0.0),
            "mars",
            [datetime(2027, 1, 1, tzinfo=timezone.utc)],
            ensure_kernels=False,
        )

    assert exc.value.code == "spice_unsupported_body"


def test_dataframe_helpers(fake_spiceypy) -> None:
    pd = pytest.importorskip("pandas")
    values = [datetime(2027, 1, 1, tzinfo=timezone.utc)]

    vectors = ls.body_vectors_ned_dataframe(
        ls.LonLat(0.0, 0.0),
        "sun",
        values,
        ensure_kernels=False,
    )
    angles = ls.body_azimuth_elevation_dataframe(
        ls.LonLat(0.0, 0.0),
        "sun",
        values,
        ensure_kernels=False,
    )

    assert isinstance(vectors, pd.DataFrame)
    assert list(vectors.columns) == ["time", "x", "y", "z"]
    assert len(vectors) == 1
    assert list(angles.columns) == ["time", "azimuth", "elevation"]
    assert len(angles) == 1


def test_plot_helper_returns_figure_and_axis(fake_spiceypy) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = ls.plot_body_elevation(
        ls.LonLat(0.0, 0.0),
        "sun",
        [datetime(2027, 1, 1, tzinfo=timezone.utc)],
        grid=False,
        ensure_kernels=False,
    )

    assert fig is not None
    assert ax.get_ylabel() == "Elevation (deg)"
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    plt.close(fig)


def test_plot_helper_can_enable_grid(fake_spiceypy) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = ls.plot_body_elevation(
        ls.LonLat(0.0, 0.0),
        "sun",
        [datetime(2027, 1, 1, tzinfo=timezone.utc)],
        grid=True,
        ensure_kernels=False,
    )

    assert any(line.get_visible() for line in ax.get_xgridlines())
    plt.close(fig)
