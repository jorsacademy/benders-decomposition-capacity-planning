from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .data import CapacityPlanningInstance


@dataclass(frozen=True)
class DeterministicEquivalentResult:
    objective: float
    first_stage_cost: float
    expected_recourse: float
    open_facilities: tuple[int, ...]
    capacity: np.ndarray
    status: int
    message: str


def solve_deterministic_equivalent(
    instance: CapacityPlanningInstance,
    mip_rel_gap: float = 0.0,
    time_limit: float | None = 30.0,
) -> DeterministicEquivalentResult:
    """Solve the extensive-form MILP used as a benchmark for Benders."""
    f = instance.n_facilities
    j = instance.n_markets
    s = instance.n_scenarios

    y_start = 0
    x_start = f
    flow_start = 2 * f
    n_flow = s * f * j
    shortage_start = flow_start + n_flow
    n_shortage = s * j
    n_vars = shortage_start + n_shortage

    def flow_idx(sc: int, i: int, market: int) -> int:
        return flow_start + sc * f * j + i * j + market

    def shortage_idx(sc: int, market: int) -> int:
        return shortage_start + sc * j + market

    c = np.zeros(n_vars, dtype=float)
    c[y_start:x_start] = instance.fixed_cost
    c[x_start:flow_start] = instance.capacity_cost
    for sc in range(s):
        p = instance.scenario_probability[sc]
        for i in range(f):
            for market in range(j):
                c[flow_idx(sc, i, market)] = p * instance.shipping_cost[i, market]
        for market in range(j):
            c[shortage_idx(sc, market)] = p * instance.shortage_penalty[market]

    integrality = np.zeros(n_vars, dtype=int)
    integrality[y_start:x_start] = 1

    lower = np.zeros(n_vars, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)
    upper[y_start:x_start] = 1.0
    upper[x_start:flow_start] = instance.max_capacity

    n_rows = f + s * f + s * j
    A = lil_matrix((n_rows, n_vars), dtype=float)
    lb = np.full(n_rows, -np.inf, dtype=float)
    ub = np.full(n_rows, np.inf, dtype=float)
    row = 0

    for i in range(f):
        A[row, x_start + i] = 1.0
        A[row, y_start + i] = -instance.max_capacity[i]
        ub[row] = 0.0
        row += 1

    for sc in range(s):
        for i in range(f):
            for market in range(j):
                A[row, flow_idx(sc, i, market)] = 1.0
            A[row, x_start + i] = -1.0
            ub[row] = 0.0
            row += 1

        for market in range(j):
            for i in range(f):
                A[row, flow_idx(sc, i, market)] = 1.0
            A[row, shortage_idx(sc, market)] = 1.0
            demand = instance.scenario_demand[sc, market]
            lb[row] = demand
            ub[row] = demand
            row += 1

    assert row == n_rows

    options: dict[str, float | bool] = {"disp": False, "mip_rel_gap": float(mip_rel_gap)}
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        options["time_limit"] = float(time_limit)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        options=options,
    )
    if result.x is None or result.fun is None:
        raise RuntimeError(f"extensive-form MILP returned no solution: {result.message}")
    if result.status not in (0, 1):
        raise RuntimeError(f"extensive-form MILP failed with status {result.status}: {result.message}")

    vector = np.asarray(result.x, dtype=float)
    y = vector[y_start:x_start]
    x = vector[x_start:flow_start]
    first_stage = float(instance.fixed_cost @ y + instance.capacity_cost @ x)
    expected_recourse = float(result.fun - first_stage)

    return DeterministicEquivalentResult(
        objective=float(result.fun),
        first_stage_cost=first_stage,
        expected_recourse=expected_recourse,
        open_facilities=tuple(int(i) for i in np.flatnonzero(y > 0.5)),
        capacity=x.copy(),
        status=int(result.status),
        message=str(result.message),
    )
