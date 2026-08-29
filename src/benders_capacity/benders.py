from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import CapacityPlanningInstance
from .master import BendersCut, solve_master
from .subproblem import SubproblemResult, solve_scenario_subproblem


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    lower_bound: float
    incumbent_upper_bound: float
    relative_gap: float
    master_objective: float
    candidate_objective: float
    cuts_added: int


@dataclass(frozen=True)
class BendersResult:
    objective: float
    first_stage_cost: float
    expected_recourse: float
    open_facilities: tuple[int, ...]
    capacity: np.ndarray
    scenario_recourse: np.ndarray
    iterations: tuple[IterationRecord, ...]
    cuts: tuple[BendersCut, ...]
    converged: bool


def _first_stage_cost(instance: CapacityPlanningInstance, y: np.ndarray, x: np.ndarray) -> float:
    return float(instance.fixed_cost @ y + instance.capacity_cost @ x)


def solve_benders(
    instance: CapacityPlanningInstance,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    mip_rel_gap: float = 0.0,
) -> BendersResult:
    """Solve the two-stage stochastic capacity model by multi-cut Benders decomposition."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    cuts: list[BendersCut] = []
    history: list[IterationRecord] = []
    best_upper = float("inf")
    best_y: np.ndarray | None = None
    best_x: np.ndarray | None = None
    best_q: np.ndarray | None = None
    converged = False

    for iteration in range(1, max_iterations + 1):
        master = solve_master(instance, cuts, mip_rel_gap=mip_rel_gap)
        y = (master.open_facilities > 0.5).astype(float)
        x = np.maximum(master.capacity, 0.0)
        first_stage = _first_stage_cost(instance, y, x)

        subproblems: list[SubproblemResult] = [
            solve_scenario_subproblem(instance, x, scenario)
            for scenario in range(instance.n_scenarios)
        ]
        q = np.array([sub.objective for sub in subproblems], dtype=float)
        expected_recourse = float(instance.scenario_probability @ q)
        candidate = first_stage + expected_recourse

        if candidate < best_upper - 1e-9:
            best_upper = candidate
            best_y = y.copy()
            best_x = x.copy()
            best_q = q.copy()

        lower = float(master.objective)
        if lower > best_upper and lower - best_upper < 1e-6 * max(1.0, abs(best_upper)):
            lower = best_upper
        relative_gap = max(0.0, best_upper - lower) / max(1.0, abs(best_upper))

        violated: list[BendersCut] = []
        for scenario, sub in enumerate(subproblems):
            if master.theta[scenario] + max(1e-8, tolerance * 0.1) < sub.objective:
                violated.append(
                    BendersCut(
                        scenario=scenario,
                        intercept=sub.cut_intercept(x),
                        gradient=sub.capacity_subgradient.copy(),
                    )
                )

        history.append(
            IterationRecord(
                iteration=iteration,
                lower_bound=lower,
                incumbent_upper_bound=best_upper,
                relative_gap=relative_gap,
                master_objective=float(master.objective),
                candidate_objective=candidate,
                cuts_added=len(violated),
            )
        )

        if relative_gap <= tolerance and not violated:
            converged = True
            break
        if not violated:
            converged = relative_gap <= max(tolerance, 1e-8)
            break

        cuts.extend(violated)

    if best_y is None or best_x is None or best_q is None:
        raise RuntimeError("Benders decomposition failed to produce an incumbent")

    best_first_stage = _first_stage_cost(instance, best_y, best_x)
    best_expected_recourse = float(instance.scenario_probability @ best_q)
    return BendersResult(
        objective=float(best_upper),
        first_stage_cost=best_first_stage,
        expected_recourse=best_expected_recourse,
        open_facilities=tuple(int(i) for i in np.flatnonzero(best_y > 0.5)),
        capacity=best_x,
        scenario_recourse=best_q,
        iterations=tuple(history),
        cuts=tuple(cuts),
        converged=converged,
    )
