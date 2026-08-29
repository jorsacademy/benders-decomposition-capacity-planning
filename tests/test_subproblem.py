import numpy as np

from benders_capacity.data import demo_instance
from benders_capacity.subproblem import solve_scenario_subproblem


def test_zero_capacity_serves_all_demand_as_shortage() -> None:
    instance = demo_instance()
    capacity = np.zeros(instance.n_facilities)
    result = solve_scenario_subproblem(instance, capacity, 0)
    expected = float(instance.shortage_penalty @ instance.scenario_demand[0])
    assert np.isclose(result.objective, expected)
    assert np.allclose(result.flow, 0.0)
    assert np.allclose(result.shortage, instance.scenario_demand[0])
    assert np.all(result.capacity_subgradient <= 1e-9)


def test_more_capacity_does_not_increase_recourse_cost() -> None:
    instance = demo_instance()
    low = solve_scenario_subproblem(instance, np.zeros(instance.n_facilities), 2)
    high = solve_scenario_subproblem(instance, instance.max_capacity, 2)
    assert high.objective <= low.objective + 1e-8


def test_benders_subgradient_cut_is_globally_valid_for_sample_points() -> None:
    instance = demo_instance()
    x0 = np.array([55.0, 40.0, 65.0, 25.0])
    sub = solve_scenario_subproblem(instance, x0, 1)
    alpha = sub.cut_intercept(x0)

    samples = [
        np.zeros(instance.n_facilities),
        instance.max_capacity,
        np.array([20.0, 70.0, 30.0, 45.0]),
    ]
    for x in samples:
        actual = solve_scenario_subproblem(instance, x, 1).objective
        lower_bound = alpha + sub.capacity_subgradient @ x
        assert lower_bound <= actual + 1e-6
