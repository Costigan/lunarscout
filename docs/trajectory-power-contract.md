# Initial Trajectory Power Contract

Status: Phase 4A accepted and implemented.
Date: 2026-09-07.

Refer to the [trajectory API design](trajectory-api-design.md) for rationale and
future SOC-planner context. This record captures the initial decisions supplied
for Phase 4A.

## Public models and units

`SolarPowerModel(rated_power_w=...)` is orientation-independent. Operations are
assumed to manage array orientation for maximum generation. `rated_power_w` is
non-negative post-conversion electrical output at full illumination, so the
initial operation is:

```text
watts_in = rated_power_w * sunlight_fraction
```

Sunlight fraction is finite and in `[0, 1]`. Array geometry, attitude, thermal
effects, degradation, and a separate photovoltaic conversion-efficiency
parameter are absent from this first model.

`BatteryModel` requires capacity, initial stored energy, and minimum stored
energy in Wh, satisfying:

```text
0 <= minimum_energy_wh <= initial_energy_wh <= capacity_wh
```

Charge and discharge efficiencies are required arguments. The initial model
accepts only `1.0` for each; this makes the ideal assumption explicit at every
construction site. A future model may replace constant charge efficiency with
an SOC-indexed lookup table without reinterpreting this model.

`RoverPowerModel` has separate non-negative constant `drive_power_w` and
`idle_power_w` loads. It does not define science, communications, heater, or
other future operating modes.

## Energy transitions

Private shared accounting integrates one constant-signal segment of finite
non-negative duration in hours. Solar generation serves the active load first.
Surplus energy charges the battery; a deficit discharges it. Stored energy is
clipped to `[0, capacity_wh]`. Energy above capacity is reported as discarded
and represents unmodeled array heating or other rejection. There is no separate
charge-power limit.

A transition is feasible when its unconstrained ending energy is at least the
minimum, including equality. The numerical comparison uses an absolute
`1e-12 * max(1, capacity_wh)` Wh tolerance and snaps only a feasible value just
below the minimum back to the minimum. Because power is constant within this
scalar segment, checking its endpoint also checks the segment minimum.

Sunlight rasters are piecewise constant over half-open intervals `[t[i],
t[i+1])`. A movement or wait that crosses a boundary is split there, and the
new interval's value applies exactly at the boundary. Feasibility is checked
after every segment.

Waiting uses the occupied cell's sunlight. Movement uses the source cell's
sunlight from departure through arrival; the destination value first applies
to a subsequent wait or movement. There is no interpolation or source/destination
aggregation.

`EnergySegment` is the immutable public record for one such constant-signal
segment. It records UTC start/stop times, mode, sampled `(x, y)` cell and
sunlight fraction, and starting/ending, generated, consumed, and discarded
energy in Wh. `EnergyTimelineResult` owns an immutable tuple of continuous
segments and reports feasibility, initial/final energy, and aggregate energy
properties. A failed candidate includes its first infeasible segment and no
later segments. A zero-duration candidate has no segments and is feasible only
when its starting energy satisfies the minimum-energy rule.

The internal sunlight timeline copies input values to a read-only float64
`(interval, y, x)` array and requires finite fractions in `[0, 1]` on its
associated georeferenced grid. SOC search is not part of Phase 4A.
