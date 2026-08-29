from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .data import CapacityPlanningInstance


@dataclass(frozen=True)
class SubproblemResult:
    scenario: int
    objective: float
    flow: np.ndarray
    shortage: np.ndarray
    capacity_subgradient: np.ndarray

    def cut_intercept(self, capacity: np.ndarray) -> float:
        """Return alpha in theta >= alpha + g^T x at the evaluated capacity."""
        x = np.asarray(capacity, dtype=float)
        return float(self.objective - self.capacity_subgradient @ x)


def solve_scenario_subproblem(
    instance: CapacityPlanningInstance,
    capacity: np.ndarray,
    scenario: int,
) -> SubproblemResult:
    """Solve one linear recourse problem and extract a capacity subgradient."""
    x = np.asarray(capacity, dtype=float)
    if x.shape != (instance.n_facilities,):
        raise ValueError("capacity has the wrong shape")
    if np.any(x < -1e-10):
        raise ValueError("capacity must be nonnegative")
    if not 0 <= scenario < instance.n_scenarios:
        raise IndexError("scenario index out of range")

    f = instance.n_facilities
    j = instance.n_markets
    n_flow = f * j
    n_vars = n_flow + j

    objective = np.empty(n_vars, dtype=float)
    objective[:n_flow] = instance.shipping_cost.reshape(-1)
    objective[n_flow:] = instance.shortage_penalty

    # Capacity rows: sum_j flow_ij <= x_i.
    a_ub = lil_matrix((f, n_vars), dtype=float)
    for i in range(f):
        start = i * j
        a_ub[i, start : start + j] = 1.0
    b_ub = x.copy()

    # Demand balance: sum_i flow_ij + shortage_j = demand_sj.
    a_eq = lil_matrix((j, n_vars), dtype=float)
    for market in range(j):
        for i in range(f):
            a_eq[market, i * j + market] = 1.0
        a_eq[market, n_flow + market] = 1.0
    b_eq = instance.scenario_demand[scenario]

    result = linprog(
        c=objective,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"recourse LP failed: {result.message}")

    vector = np.asarray(result.x, dtype=float)
    flow = vector[:n_flow].reshape(f, j)
    shortage = vector[n_flow:]
    gradient = np.asarray(result.ineqlin.marginals, dtype=float)

    # SciPy defines marginals as d objective / d b_ub. Here b_ub is capacity.
    # Hence the capacity-row marginals are a subgradient of Q_s(x).
    if gradient.shape != (f,):
        raise RuntimeError("unexpected dual vector shape returned by linprog")

    return SubproblemResult(
        scenario=int(scenario),
        objective=float(result.fun),
        flow=flow,
        shortage=shortage,
        capacity_subgradient=gradient,
    )
