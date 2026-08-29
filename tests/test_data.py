import numpy as np
import pytest

from benders_capacity.data import CapacityPlanningInstance, demo_instance


def test_demo_probabilities_sum_to_one() -> None:
    instance = demo_instance()
    assert np.isclose(instance.scenario_probability.sum(), 1.0)
    assert instance.n_facilities == 4
    assert instance.n_markets == 5
    assert instance.n_scenarios == 5


def test_invalid_probability_vector_is_rejected() -> None:
    base = demo_instance()
    with pytest.raises(ValueError, match="sum to one"):
        CapacityPlanningInstance(
            fixed_cost=base.fixed_cost,
            capacity_cost=base.capacity_cost,
            max_capacity=base.max_capacity,
            shipping_cost=base.shipping_cost,
            shortage_penalty=base.shortage_penalty,
            scenario_demand=base.scenario_demand,
            scenario_probability=np.full(base.n_scenarios, 0.1),
        )
