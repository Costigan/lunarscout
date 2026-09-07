from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..errors import PlanningError, TrajectoryInputError
from ._validation import StaticProblem


_MAX_FLOAT64 = np.finfo(np.float64).max


@dataclass(frozen=True, slots=True)
class CompiledStaticProblem:
    available: NDArray[np.bool_]
    elevation_m: NDArray[np.float64]
    step_dx: NDArray[np.int8]
    step_dy: NDArray[np.int8]
    step_distance_m: NDArray[np.float64]
    speed_m_per_h: float
    slip_slopes: NDArray[np.float64]
    slip_factors: NDArray[np.float64]
    slip_mode: int


@dataclass(frozen=True, slots=True)
class BlockStaticResult:
    travel_time_hours: NDArray[np.float64]
    predecessor_x: NDArray[np.int64] | None
    predecessor_y: NDArray[np.int64] | None
    block_visits: NDArray[np.int64]
    block_improvements: NDArray[np.int64]
    activation_count: int
    relaxation_pass_count: int
    block_width: int
    block_height: int


def compile_static_problem(problem: StaticProblem) -> CompiledStaticProblem:
    if problem.elevation_m is None:
        elevation = np.zeros(problem.available.shape, dtype=np.float64)
    else:
        elevation = np.ascontiguousarray(problem.elevation_m, dtype=np.float64)
    step_dx = np.asarray([step.dx for step in problem.steps], dtype=np.int8)
    step_dy = np.asarray([step.dy for step in problem.steps], dtype=np.int8)
    distances = np.asarray(
        [step.distance_m for step in problem.steps], dtype=np.float64
    )
    if problem.model.slip is None:
        slopes = np.empty(0, dtype=np.float64)
        factors = np.empty(0, dtype=np.float64)
        slip_mode = 0
    else:
        slopes = np.asarray(problem.model.slip.signed_slopes, dtype=np.float64)
        factors = np.asarray(problem.model.slip.factors, dtype=np.float64)
        slip_mode = 1 if problem.model.slip.extrapolation == "infeasible" else 2
    return CompiledStaticProblem(
        available=np.ascontiguousarray(problem.available, dtype=np.bool_),
        elevation_m=elevation,
        step_dx=step_dx,
        step_dy=step_dy,
        step_distance_m=distances,
        speed_m_per_h=problem.model.speed_m_per_h,
        slip_slopes=slopes,
        slip_factors=factors,
        slip_mode=slip_mode,
    )


@njit(cache=True)
def _slip_factor(
    signed_slope: float,
    slopes: NDArray[np.float64],
    factors: NDArray[np.float64],
    mode: int,
) -> float:
    if mode == 0:
        return 1.0
    if signed_slope < slopes[0]:
        return factors[0] if mode == 2 else np.inf
    if signed_slope > slopes[-1]:
        return factors[-1] if mode == 2 else np.inf
    if signed_slope == slopes[-1]:
        return factors[-1]
    for index in range(slopes.size - 1):
        left = slopes[index]
        right = slopes[index + 1]
        if signed_slope <= right:
            fraction = (signed_slope - left) / (right - left)
            return factors[index] + fraction * (factors[index + 1] - factors[index])
    return factors[-1]


@njit(cache=True)
def _relax_block_to_quiescence(
    travel_time: NDArray[np.float64],
    predecessor_x: NDArray[np.int64],
    predecessor_y: NDArray[np.int64],
    available: NDArray[np.bool_],
    elevation_m: NDArray[np.float64],
    step_dx: NDArray[np.int8],
    step_dy: NDArray[np.int8],
    step_distance_m: NDArray[np.float64],
    speed_m_per_h: float,
    slip_slopes: NDArray[np.float64],
    slip_factors: NDArray[np.float64],
    slip_mode: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    track_predecessors: bool,
) -> tuple[bool, int, bool, bool]:
    height, width = available.shape
    maximum_passes = max(2, (x1 - x0) * (y1 - y0) + 1)
    any_improvement = False
    for pass_index in range(maximum_passes):
        improved = False
        for y in range(y0, y1):
            for x in range(x0, x1):
                if not available[y, x]:
                    continue
                current = travel_time[y, x]
                for step_index in range(step_dx.size):
                    dx = int(step_dx[step_index])
                    dy = int(step_dy[step_index])
                    source_x = x - dx
                    source_y = y - dy
                    if (
                        source_x < 0
                        or source_x >= width
                        or source_y < 0
                        or source_y >= height
                        or not available[source_y, source_x]
                    ):
                        continue
                    source_cost = travel_time[source_y, source_x]
                    if not np.isfinite(source_cost):
                        continue
                    if dx != 0 and dy != 0:
                        if not available[source_y, x] or not available[y, source_x]:
                            continue
                    distance = step_distance_m[step_index]
                    factor = 1.0
                    if slip_mode != 0:
                        signed_slope = (
                            elevation_m[y, x] - elevation_m[source_y, source_x]
                        ) / distance
                        factor = _slip_factor(
                            signed_slope,
                            slip_slopes,
                            slip_factors,
                            slip_mode,
                        )
                    if not np.isfinite(factor):
                        continue
                    edge = distance / speed_m_per_h * factor
                    if source_cost > _MAX_FLOAT64 - edge:
                        return any_improvement, pass_index + 1, True, False
                    candidate = source_cost + edge
                    if candidate < current:
                        travel_time[y, x] = candidate
                        if track_predecessors:
                            predecessor_x[y, x] = source_x
                            predecessor_y[y, x] = source_y
                        current = candidate
                        improved = True
                        any_improvement = True
        if not improved:
            return any_improvement, pass_index + 1, False, True
    return any_improvement, maximum_passes, False, False


def _validate_block_dimension(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TrajectoryInputError(
            f"{name} must be a positive integer.",
            code="trajectory_invalid_block_geometry",
            details={"name": name, "value": value},
        )
    result = int(value)
    if result <= 0:
        raise TrajectoryInputError(
            f"{name} must be a positive integer.",
            code="trajectory_invalid_block_geometry",
            details={"name": name, "value": value},
        )
    return result


def block_static_travel_time(
    problem: StaticProblem,
    *,
    block_width: int = 32,
    block_height: int = 32,
    maximum_activations: int | None = None,
    track_predecessors: bool = True,
) -> BlockStaticResult:
    """Run private Numba block-relaxation for a validated static problem."""

    block_width = _validate_block_dimension(block_width, name="block_width")
    block_height = _validate_block_dimension(block_height, name="block_height")
    compiled = compile_static_problem(problem)
    height, width = compiled.available.shape
    block_columns = (width + block_width - 1) // block_width
    block_rows = (height + block_height - 1) // block_height
    block_count = block_rows * block_columns
    if maximum_activations is None:
        maximum_activations = max(1_000, height * width * block_count * 4)
    maximum_activations = _validate_block_dimension(
        maximum_activations,
        name="maximum_activations",
    )

    travel_time = np.full((height, width), np.inf, dtype=np.float64)
    if track_predecessors:
        predecessor_x = np.full((height, width), -1, dtype=np.int64)
        predecessor_y = np.full((height, width), -1, dtype=np.int64)
    else:
        predecessor_x = np.empty((1, 1), dtype=np.int64)
        predecessor_y = np.empty((1, 1), dtype=np.int64)
    start_x, start_y = problem.start
    travel_time[start_y, start_x] = 0.0
    start_block = (start_x // block_width, start_y // block_height)
    queue: deque[tuple[int, int]] = deque([start_block])
    queued = np.zeros((block_rows, block_columns), dtype=np.bool_)
    queued[start_block[1], start_block[0]] = True
    visits = np.zeros((block_rows, block_columns), dtype=np.int64)
    improvements = np.zeros((block_rows, block_columns), dtype=np.int64)
    activation_count = 0
    relaxation_pass_count = 0

    while queue:
        block_x, block_y = queue.popleft()
        queued[block_y, block_x] = False
        activation_count += 1
        if activation_count > maximum_activations:
            raise PlanningError(
                "Static block relaxation exceeded its activation safety bound.",
                code="trajectory_block_activation_limit",
                details={
                    "maximum_activations": maximum_activations,
                    "block_columns": block_columns,
                    "block_rows": block_rows,
                },
            )
        visits[block_y, block_x] += 1
        x0 = block_x * block_width
        y0 = block_y * block_height
        x1 = min(width, x0 + block_width)
        y1 = min(height, y0 + block_height)
        improved, passes, overflow, converged = _relax_block_to_quiescence(
            travel_time,
            predecessor_x,
            predecessor_y,
            compiled.available,
            compiled.elevation_m,
            compiled.step_dx,
            compiled.step_dy,
            compiled.step_distance_m,
            compiled.speed_m_per_h,
            compiled.slip_slopes,
            compiled.slip_factors,
            compiled.slip_mode,
            x0,
            y0,
            x1,
            y1,
            track_predecessors,
        )
        relaxation_pass_count += passes
        if overflow:
            raise PlanningError(
                "Accumulated trajectory travel time overflowed float64.",
                code="trajectory_cost_overflow",
            )
        if not converged:
            raise PlanningError(
                "Static block relaxation did not reach local quiescence.",
                code="trajectory_block_nonconvergence",
                details={"block": [block_x, block_y], "passes": passes},
            )
        if improved:
            improvements[block_y, block_x] += 1

        propagate_seed = (block_x, block_y) == start_block and visits[block_y, block_x] == 1
        if not improved and not propagate_seed:
            continue
        for neighbor_y in range(max(0, block_y - 1), min(block_rows, block_y + 2)):
            for neighbor_x in range(
                max(0, block_x - 1), min(block_columns, block_x + 2)
            ):
                if neighbor_x == block_x and neighbor_y == block_y:
                    continue
                if not queued[neighbor_y, neighbor_x]:
                    queue.append((neighbor_x, neighbor_y))
                    queued[neighbor_y, neighbor_x] = True

    return BlockStaticResult(
        travel_time_hours=travel_time,
        predecessor_x=predecessor_x if track_predecessors else None,
        predecessor_y=predecessor_y if track_predecessors else None,
        block_visits=visits,
        block_improvements=improvements,
        activation_count=activation_count,
        relaxation_pass_count=relaxation_pass_count,
        block_width=block_width,
        block_height=block_height,
    )


def reconstruct_block_path(
    result: BlockStaticResult,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> NDArray[np.int64] | None:
    if result.predecessor_x is None or result.predecessor_y is None:
        raise PlanningError(
            "Static block predecessors were not retained.",
            code="trajectory_predecessors_unavailable",
        )
    goal_x, goal_y = goal
    if not np.isfinite(result.travel_time_hours[goal_y, goal_x]):
        return None
    reversed_path = [goal]
    x, y = goal
    maximum_length = result.travel_time_hours.size
    while (x, y) != start:
        next_x = int(result.predecessor_x[y, x])
        next_y = int(result.predecessor_y[y, x])
        if next_x < 0 or next_y < 0 or len(reversed_path) >= maximum_length:
            raise PlanningError(
                "Static block predecessor chain is invalid.",
                code="trajectory_invalid_predecessor_chain",
                details={"cell": [x, y]},
            )
        x, y = next_x, next_y
        reversed_path.append((x, y))
    return np.asarray(reversed_path[::-1], dtype=np.int64)
