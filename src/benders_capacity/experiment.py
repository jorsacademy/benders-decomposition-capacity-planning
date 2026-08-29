from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .benders import BendersResult, solve_benders
from .data import CapacityPlanningInstance, demo_instance
from .deterministic_equivalent import (
    DeterministicEquivalentResult,
    solve_deterministic_equivalent,
)


def comparison_report(
    instance: CapacityPlanningInstance | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> dict[str, object]:
    """Solve by Benders and extensive form and return a JSON-serializable comparison."""
    model = demo_instance() if instance is None else instance
    benders = solve_benders(model, tolerance=tolerance, max_iterations=max_iterations)
    extensive = solve_deterministic_equivalent(model)

    return {
        "benders": _serialize_benders(benders),
        "deterministic_equivalent": _serialize_extensive(extensive),
        "objective_difference": float(benders.objective - extensive.objective),
    }


def _serialize_benders(result: BendersResult) -> dict[str, object]:
    return {
        "objective": result.objective,
        "first_stage_cost": result.first_stage_cost,
        "expected_recourse": result.expected_recourse,
        "open_facilities": list(result.open_facilities),
        "capacity": result.capacity.tolist(),
        "scenario_recourse": result.scenario_recourse.tolist(),
        "converged": result.converged,
        "iteration_count": len(result.iterations),
        "cut_count": len(result.cuts),
        "iterations": [asdict(item) for item in result.iterations],
    }


def _serialize_extensive(result: DeterministicEquivalentResult) -> dict[str, object]:
    return {
        "objective": result.objective,
        "first_stage_cost": result.first_stage_cost,
        "expected_recourse": result.expected_recourse,
        "open_facilities": list(result.open_facilities),
        "capacity": np.asarray(result.capacity).tolist(),
    }
