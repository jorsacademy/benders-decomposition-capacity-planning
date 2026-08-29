import numpy as np

from benders_capacity.benders import solve_benders
from benders_capacity.data import demo_instance
from benders_capacity.deterministic_equivalent import solve_deterministic_equivalent
from benders_capacity.subproblem import solve_scenario_subproblem


def test_benders_matches_extensive_form() -> None:
    instance = demo_instance()
    benders = solve_benders(instance, tolerance=1e-7, max_iterations=80)
    extensive = solve_deterministic_equivalent(instance)

    assert benders.converged
    assert abs(benders.objective - extensive.objective) <= 1e-4
    assert np.all(benders.capacity >= -1e-8)
    for i in range(instance.n_facilities):
        if i not in benders.open_facilities:
            assert benders.capacity[i] <= 1e-7
        assert benders.capacity[i] <= instance.max_capacity[i] + 1e-7


def test_benders_bounds_are_monotone() -> None:
    instance = demo_instance()
    result = solve_benders(instance, tolerance=1e-6, max_iterations=80)
    lower = np.array([record.lower_bound for record in result.iterations])
    upper = np.array([record.incumbent_upper_bound for record in result.iterations])

    assert np.all(np.diff(lower) >= -1e-6)
    assert np.all(np.diff(upper) <= 1e-6)


def test_reported_recourse_matches_fresh_subproblem_solves() -> None:
    instance = demo_instance()
    result = solve_benders(instance, tolerance=1e-6, max_iterations=80)
    recomputed = np.array(
        [
            solve_scenario_subproblem(instance, result.capacity, sc).objective
            for sc in range(instance.n_scenarios)
        ]
    )
    assert np.allclose(recomputed, result.scenario_recourse, atol=1e-6)
    expected = instance.scenario_probability @ recomputed
    assert np.isclose(result.expected_recourse, expected, atol=1e-6)
