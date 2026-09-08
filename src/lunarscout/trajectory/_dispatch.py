from __future__ import annotations

from dataclasses import dataclass

from ..errors import PlanningError, TrajectoryInputError


@dataclass(frozen=True, slots=True)
class DynamicDispatch:
    algorithm: str
    backend: str


def resolve_dynamic_dispatch(algorithm: str, backend: str) -> DynamicDispatch:
    supported_algorithms = {"gridrunner", "safe_interval"}
    if not isinstance(algorithm, str) or algorithm not in supported_algorithms:
        raise TrajectoryInputError(
            "Unknown dynamic trajectory algorithm.",
            code="trajectory_unknown_algorithm",
            details={
                "algorithm": algorithm,
                "supported": sorted(supported_algorithms),
            },
        )
    if not isinstance(backend, str) or backend not in {"auto", "cpu", "cuda"}:
        raise TrajectoryInputError(
            "Unknown dynamic trajectory backend.",
            code="trajectory_unknown_backend",
            details={"backend": backend, "supported": ["auto", "cpu", "cuda"]},
        )
    if backend == "cuda":
        raise PlanningError(
            "The selected dynamic algorithm has no CUDA implementation.",
            code="trajectory_backend_unavailable",
            details={"algorithm": algorithm, "backend": backend},
        )
    return DynamicDispatch(algorithm=algorithm, backend="cpu")
