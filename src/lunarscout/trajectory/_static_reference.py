from __future__ import annotations

from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray

from ..errors import PlanningError
from ._geometry import cell_distance_m
from ._validation import StaticProblem


_MAX_FLOAT64 = np.finfo(np.float64).max


def _add_cost(left: float, right: float) -> float:
    if left > _MAX_FLOAT64 - right:
        raise PlanningError(
            "Accumulated trajectory travel time overflowed float64.",
            code="trajectory_cost_overflow",
            details={"accumulated_hours": left, "edge_hours": right},
        )
    return left + right


def dijkstra_field(problem: StaticProblem) -> NDArray[np.float64]:
    height, width = problem.available.shape
    result = np.full((height, width), np.inf, dtype=np.float64)
    start_x, start_y = problem.start
    result[start_y, start_x] = 0.0
    frontier: list[tuple[float, int, int]] = [(0.0, start_y, start_x)]
    while frontier:
        cost, y, x = heappop(frontier)
        if cost != result[y, x]:
            continue
        for step in problem.steps:
            edge = problem.transition_time(x, y, step)
            if not np.isfinite(edge):
                continue
            nx = x + step.dx
            ny = y + step.dy
            candidate = _add_cost(cost, edge)
            if candidate < result[ny, nx]:
                result[ny, nx] = candidate
                heappush(frontier, (candidate, ny, nx))
    return result


def _heuristic(problem: StaticProblem, cell: tuple[int, int]) -> float:
    assert problem.goal is not None
    distance = cell_distance_m(
        problem.georef,
        cell,
        problem.goal,
        units_to_metres=problem.units_to_metres,
    )
    heuristic = distance / problem.model.speed_m_per_h * problem.minimum_factor
    if not np.isfinite(heuristic):
        raise PlanningError(
            "The trajectory heuristic overflowed float64.",
            code="trajectory_cost_overflow",
            details={"distance_m": distance},
        )
    return heuristic


def astar_path(problem: StaticProblem) -> tuple[float | None, NDArray[np.int64] | None]:
    assert problem.goal is not None
    if problem.start == problem.goal:
        return 0.0, np.asarray([problem.start], dtype=np.int64)

    height, width = problem.available.shape
    costs = np.full((height, width), np.inf, dtype=np.float64)
    predecessor_x = np.full((height, width), -1, dtype=np.int64)
    predecessor_y = np.full((height, width), -1, dtype=np.int64)
    start_x, start_y = problem.start
    costs[start_y, start_x] = 0.0
    frontier: list[tuple[float, float, int, int]] = [
        (_heuristic(problem, problem.start), 0.0, start_y, start_x)
    ]

    while frontier:
        _priority, cost, y, x = heappop(frontier)
        if cost != costs[y, x]:
            continue
        if (x, y) == problem.goal:
            break
        for step in problem.steps:
            edge = problem.transition_time(x, y, step)
            if not np.isfinite(edge):
                continue
            nx = x + step.dx
            ny = y + step.dy
            candidate = _add_cost(cost, edge)
            if candidate < costs[ny, nx]:
                costs[ny, nx] = candidate
                predecessor_x[ny, nx] = x
                predecessor_y[ny, nx] = y
                priority = _add_cost(candidate, _heuristic(problem, (nx, ny)))
                heappush(frontier, (priority, candidate, ny, nx))

    goal_x, goal_y = problem.goal
    goal_cost = costs[goal_y, goal_x]
    if not np.isfinite(goal_cost):
        return None, None

    reversed_path = [problem.goal]
    x, y = problem.goal
    while (x, y) != problem.start:
        previous_x = int(predecessor_x[y, x])
        previous_y = int(predecessor_y[y, x])
        if previous_x < 0 or previous_y < 0:
            raise RuntimeError("A* predecessor chain is incomplete")
        x, y = previous_x, previous_y
        reversed_path.append((x, y))
    reversed_path.reverse()
    return float(goal_cost), np.asarray(reversed_path, dtype=np.int64)
