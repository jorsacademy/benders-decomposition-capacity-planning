from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapacityPlanningInstance:
    """Two-stage capacitated distribution planning data."""

    fixed_cost: np.ndarray
    capacity_cost: np.ndarray
    max_capacity: np.ndarray
    shipping_cost: np.ndarray
    shortage_penalty: np.ndarray
    scenario_demand: np.ndarray
    scenario_probability: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "fixed_cost": self.fixed_cost,
            "capacity_cost": self.capacity_cost,
            "max_capacity": self.max_capacity,
            "shipping_cost": self.shipping_cost,
            "shortage_penalty": self.shortage_penalty,
            "scenario_demand": self.scenario_demand,
            "scenario_probability": self.scenario_probability,
        }
        for name, value in arrays.items():
            object.__setattr__(self, name, np.asarray(value, dtype=float))

        f = self.fixed_cost.size
        if self.fixed_cost.ndim != 1 or f == 0:
            raise ValueError("fixed_cost must be a nonempty one-dimensional array")
        if self.capacity_cost.shape != (f,) or self.max_capacity.shape != (f,):
            raise ValueError("capacity vectors must match the number of facilities")
        if np.any(self.fixed_cost < 0) or np.any(self.capacity_cost < 0):
            raise ValueError("first-stage costs must be nonnegative")
        if np.any(self.max_capacity <= 0):
            raise ValueError("max_capacity must be strictly positive")

        if self.shipping_cost.ndim != 2 or self.shipping_cost.shape[0] != f:
            raise ValueError("shipping_cost must have one row per facility")
        j = self.shipping_cost.shape[1]
        if j == 0 or self.shortage_penalty.shape != (j,):
            raise ValueError("shortage_penalty must match the number of markets")
        if np.any(self.shipping_cost < 0) or np.any(self.shortage_penalty <= 0):
            raise ValueError("recourse costs must be nonnegative and penalties positive")

        if self.scenario_demand.ndim != 2 or self.scenario_demand.shape[1] != j:
            raise ValueError("scenario_demand must have one column per market")
        s = self.scenario_demand.shape[0]
        if s == 0 or self.scenario_probability.shape != (s,):
            raise ValueError("scenario_probability must match the number of scenarios")
        if np.any(self.scenario_demand < 0):
            raise ValueError("scenario demand must be nonnegative")
        if np.any(self.scenario_probability <= 0):
            raise ValueError("scenario probabilities must be strictly positive")
        if not np.isclose(np.sum(self.scenario_probability), 1.0, atol=1e-10):
            raise ValueError("scenario probabilities must sum to one")

    @property
    def n_facilities(self) -> int:
        return int(self.fixed_cost.size)

    @property
    def n_markets(self) -> int:
        return int(self.shipping_cost.shape[1])

    @property
    def n_scenarios(self) -> int:
        return int(self.scenario_demand.shape[0])


def demo_instance() -> CapacityPlanningInstance:
    """Return a synthetic instance with nontrivial facility-opening decisions."""
    return CapacityPlanningInstance(
        fixed_cost=np.array([150.0, 132.0, 165.0, 118.0]),
        capacity_cost=np.array([1.65, 1.80, 1.55, 1.95]),
        max_capacity=np.array([120.0, 105.0, 135.0, 95.0]),
        shipping_cost=np.array(
            [
                [2.2, 3.1, 5.4, 6.2, 4.8],
                [4.0, 2.1, 3.0, 5.3, 4.2],
                [5.2, 4.1, 2.0, 2.6, 3.1],
                [3.8, 5.0, 4.2, 2.4, 2.0],
            ]
        ),
        shortage_penalty=np.array([18.0, 18.0, 20.0, 20.0, 19.0]),
        scenario_demand=np.array(
            [
                [34.0, 30.0, 31.0, 28.0, 27.0],
                [41.0, 36.0, 37.0, 31.0, 30.0],
                [30.0, 33.0, 40.0, 36.0, 34.0],
                [38.0, 29.0, 34.0, 42.0, 37.0],
                [45.0, 39.0, 42.0, 39.0, 35.0],
            ]
        ),
        scenario_probability=np.array([0.18, 0.22, 0.20, 0.20, 0.20]),
    )
