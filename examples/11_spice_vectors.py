"""Compute Sun and Earth local-frame vectors and azimuth/elevation histories.

Requires SPICE kernel download on first use (cached after that).  No DEM,
scenario, or GPU is needed.  Prints NED vectors and azimuth/elevation angles
for a lunar south-polar point over a short time window.
"""

from __future__ import annotations

from datetime import timedelta

import lunarscout as ls


def main() -> None:
    point = ls.LonLat(longitude=0.0, latitude=-89.5)
    sample_times = list(
        ls.iter_times(
            "2027-01-01T00:00:00Z",
            "2027-01-01T06:00:00Z",
            timedelta(hours=2),
        )
    )

    # -- Vectors ------------------------------------------------------------

    print("=== Sun NED vectors (x=north, y=east, z=down, km) ===")
    sun_vectors = ls.body_vectors_ned(point, "sun", sample_times)
    for time, (x, y, z) in zip(sample_times, sun_vectors):
        mag = (x * x + y * y + z * z) ** 0.5
        print(f"  {time:%Y-%m-%dT%H:%M:%SZ}  x={x: 10.2f}  y={y: 10.2f}  z={z: 10.2f}  |v|={mag: 9.2f}")

    print("\n=== Earth NED vectors ===")
    earth_vectors = ls.body_vectors_ned(point, "earth", sample_times)
    for time, (x, y, z) in zip(sample_times, earth_vectors):
        print(f"  {time:%Y-%m-%dT%H:%M:%SZ}  x={x: 10.2f}  y={y: 10.2f}  z={z: 10.2f}")

    # -- DataFrame variant --------------------------------------------------

    print("\n=== Sun vectors as DataFrame ===")
    df = ls.body_vectors_ned_dataframe(point, "sun", sample_times)
    print(df.to_string(index=False))

    # -- Azimuth / elevation ------------------------------------------------

    print("\n=== Sun azimuth and elevation (deg; az=0 north, az=90 east; el=+90 zenith) ===")
    sun_angles = ls.body_azimuth_elevation(point, "sun", sample_times)
    for time, (az, el) in zip(sample_times, sun_angles):
        print(f"  {time:%Y-%m-%dT%H:%M:%SZ}  azimuth={az: 7.2f}  elevation={el: 7.2f}")

    print("\n=== Earth azimuth and elevation as DataFrame ===")
    df = ls.body_azimuth_elevation_dataframe(point, "earth", sample_times)
    print(df.to_string(index=False))

    # -- TimeRange convenience ----------------------------------------------

    print("\n=== Same results using ls.times() ===")
    time_range = ls.times("2027-01-01T00:00:00Z", "2027-01-01T06:00:00Z", step_hours=2)
    angles2 = ls.body_azimuth_elevation(point, "sun", time_range)
    assert len(sun_angles) == len(angles2)
    print(f"  TimeRange produced {time_range.time_count} samples")


if __name__ == "__main__":
    main()
