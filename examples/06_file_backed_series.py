"""Create and query a file-backed timestamped GeoTIFF series.

Inputs: deterministic six-layer synthetic temporal cube under --workspace.
Outputs: analysis/synthetic_sun.temporal with manifest, VRT, and layer TIFFs.
Resources: small fixture; demonstrates bounded caches without a GPU.
"""

from __future__ import annotations

import lunarscout as ls

from _example_support import ensure_synthetic_series, example_parser


def main() -> None:
    args = example_parser(__doc__).parse_args()
    series = ensure_synthetic_series(args.workspace)
    first, first_georef = series.read_layer(0)
    nearest, _ = series.read_time("2027-01-01T02:20:00Z", method="nearest")

    print(f"root={series.root}")
    print(f"shape={series.shape}, dtype={series.dtype}, signal={series.signal_name}")
    print(f"first layer range=({first.min():.3f}, {first.max():.3f})")
    print(f"nearest lookup mean={nearest.mean():.3f}")
    print(f"georeference grid={first_georef.width} x {first_georef.height}")
    print(f"VRT={series.vrt_path}")


if __name__ == "__main__":
    main()
