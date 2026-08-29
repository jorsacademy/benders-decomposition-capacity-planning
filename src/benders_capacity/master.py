from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .data import CapacityPlanningInstance


@dataclass(frozen=True)
class BendersCut:
    scenario: int
    intercept: float
    gradient: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "gradient", np.asarray(self.gradient, dtype=float))


@dataclass(frozen=True)
class MasterSolution:
    objective: float
    open_facilities: np.ndarray
    capacity: np.ndarray
    theta: np.ndarray
    status: int
    message: str


def solve_master(
    instance: CapacityPlanningInstance,
    cuts: list[BendersCut],
    mip_rel_gap: float = 0.0,
    time_limit: float | None = 30.0,
) -> MasterSolution:
    """Solve the Benders master problem with accumulated scenario cuts."""
    if mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be nonnegative")

    f = instance.n_facilities
    s = instance.n_scenarios
    y_start = 0
    x_start = f
    theta_start = 2 * f
    n_vars = 2 * f + s

    c = np.zeros(n_vars, dtype=float)
    c[y_start:x_start] = instance.fixed_cost
    c[x_start:theta_start] = instance.capacity_cost
    c[theta_start:] = instance.scenario_probability

    integrality = np.zeros(n_vars, dtype=int)
    integrality[y_start:x_start] = 1

    lower = np.zeros(n_vars, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)
    upper[y_start:x_start] = 1.0
    upper[x_start:theta_start] = instance.max_capacity

    rows = f + len(cuts)
    A = lil_matrix((rows, n_vars), dtype=float)
    lb = np.full(rows, -np.inf, dtype=float)
    ub = np.full(rows, np.inf, dtype=float)
    row = 0

    for i in range(f):
        A[row, x_start + i] = 1.0
        A[row, y_start + i] = -instance.max_capacity[i]
        ub[row] = 0.0
        row += 1

    for cut in cuts:
        if not 0 <= cut.scenario < s:
            raise ValueError("cut scenario index out of range")
        if cut.gradient.shape != (f,):
            raise ValueError("cut gradient has the wrong shape")
        A[row, theta_start + cut.scenario] = 1.0
        for i in range(f):
            A[row, x_start + i] = -cut.gradient[i]
        lb[row] = cut.intercept
        row += 1

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
        raise RuntimeError(f"master MILP returned no solution: {result.message}")
    if result.status not in (0, 1):
        raise RuntimeError(f"master MILP failed with status {result.status}: {result.message}")

    vector = np.asarray(result.x, dtype=float)
    return MasterSolution(
        objective=float(result.fun),
        open_facilities=vector[y_start:x_start],
        capacity=vector[x_start:theta_start],
        theta=vector[theta_start:],
        status=int(result.status),
        message=str(result.message),
    )
