# Initial Trajectory Power Contract

Status: accepted model and scalar-accounting foundation; movement sampling is
not yet frozen.
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

## Scalar energy transition

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

Movement and waiting that cross environmental boundaries will be integrated as
consecutive constant-signal segments, checking feasibility after every segment.
Waiting uses the occupied cell's sunlight. The raster sampling rule for sunlight
while moving between two cells remains unresolved and blocks SOC search; it is
not part of `SolarPowerModel`.
