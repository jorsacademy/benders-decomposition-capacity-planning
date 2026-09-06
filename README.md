# Benders Decomposition Capacity Planning

A reproducible Operations Research case study of **multi-cut Benders decomposition** for a two-stage stochastic capacity-planning problem. The project separates mixed-integer first-stage facility/capacity decisions from linear scenario recourse problems, generates optimality cuts from LP sensitivity information, and verifies the result against the full deterministic-equivalent MILP.

The implementation uses `scipy.optimize.milp` for the master/extensive-form MILPs and `scipy.optimize.linprog(method="highs")` for scenario subproblems.

## Problem

Before uncertain demand is observed, the planner decides:

- `y_i`: whether facility `i` is opened,
- `x_i`: installed capacity at facility `i`.

After demand scenario `s` is observed, the recourse model decides:

- `q_sij`: quantity shipped from facility `i` to market `j`,
- `u_sj`: unmet demand at market `j`.

A compact two-stage model is

```text
min  fixed_cost(y) + capacity_cost(x) + E_s[Q_s(x)]

s.t. 0 <= x_i <= M_i y_i
     y_i in {0,1}
```

where each scenario recourse problem is

```text
Q_s(x) = min  sum_ij c_ij q_sij + sum_j p_j u_sj

s.t. sum_j q_sij <= x_i                       for every facility i
     sum_i q_sij + u_sj = demand_sj           for every market j
     q_sij, u_sj >= 0
```

Shortage variables make every recourse LP feasible. Therefore this teaching instance requires **Benders optimality cuts but no feasibility cuts**. In general, classical Benders decomposition may require both.

## Decomposition

The master contains only `y`, `x`, and one recourse approximation variable `theta_s` per scenario. At a candidate capacity vector `x_bar`, every scenario LP returns its optimal value `Q_s(x_bar)` and the marginal sensitivity of that value with respect to the capacity right-hand sides.

SciPy documents `ineqlin.marginals` as the partial derivative of the LP objective with respect to the inequality right-hand side. Because the subproblem capacity rows are

```text
sum_j q_sij <= x_i,
```

the returned capacity marginals form a subgradient `g_s` of the convex recourse value function. The generated multi-cut is

```text
theta_s >= Q_s(x_bar) + g_s^T (x - x_bar).
```

The algorithm repeatedly:

1. solves the mixed-integer master,
2. solves all independent scenario LPs,
3. updates a feasible upper bound from the true recourse values,
4. adds violated scenario optimality cuts,
5. stops when the incumbent upper bound and master lower bound agree within tolerance.

For a two-stage stochastic linear recourse model, this is the classical **L-shaped method**, i.e. a stochastic-programming specialization of Benders decomposition.

## Why compare with the deterministic equivalent?

Benders decomposition is not a different optimization model. It is an algorithm for exploiting model structure. The repository therefore also builds the full extensive-form MILP containing all scenario flow and shortage variables.

The automated tests require the Benders objective to match the extensive-form objective within numerical tolerance. This prevents a plausible-looking but mathematically incorrect cut implementation from passing unnoticed.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_comparison.py
├── src/benders_capacity/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benders.py
│   ├── data.py
│   ├── deterministic_equivalent.py
│   ├── experiment.py
│   ├── master.py
│   └── subproblem.py
├── tests/
│   ├── test_cli.py
│   ├── test_data.py
│   ├── test_solvers.py
│   └── test_subproblem.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.10+ is supported.

## Run

```bash
python -m benders_capacity --tolerance 1e-6 --max-iterations 100
```

or

```bash
python examples/run_comparison.py
```

The JSON output contains:

- Benders objective,
- extensive-form objective,
- objective difference,
- open facilities,
- installed capacity,
- first-stage and expected recourse cost,
- scenario recourse values,
- lower/upper bound history,
- relative optimality gap,
- number of generated cuts.

## Tests

```bash
python -m pytest
```

The test suite verifies:

- input-data validation,
- zero-capacity recourse behavior,
- nonincreasing recourse cost as capacity increases,
- global validity of sampled Benders cuts,
- Benders/extensive-form objective agreement,
- monotone lower and incumbent upper bounds,
- first-stage linking constraints,
- fresh recomputation of expected recourse,
- CLI output validity.

GitHub Actions executes package installation, bytecode compilation, and the full test suite on Python 3.10 and 3.12.

## Methodological notes

- This is **multi-cut** Benders because a separate `theta_s` and cut family are maintained for every scenario. A single-cut implementation would aggregate scenario dual information into one expected-recourse cut.
- Complete recourse is created deliberately through shortage variables. Removing that modeling choice can make scenario LPs infeasible and would require feasibility cuts obtained from an infeasibility certificate/Farkas ray.
- Dual sign conventions are solver-interface specific. The implementation uses SciPy's documented RHS sensitivity convention directly and tests cut validity numerically.
- The synthetic instance is small enough that the extensive form is easy to solve. The pedagogical purpose is to expose the decomposition mechanics; Benders becomes computationally interesting when the number/size of recourse blocks is much larger.
- Parallel scenario solves, cut strengthening, Pareto-optimal cuts, stabilization, and branch-and-Benders-cut are natural extensions but intentionally outside this minimal implementation.

## References

- J. F. Benders, *Partitioning procedures for solving mixed-variables programming problems*, Numerische Mathematik 4, 238-252, 1962. DOI: https://doi.org/10.1007/BF01386316
- SciPy documentation, `scipy.optimize.linprog`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
- SciPy documentation, `scipy.optimize.milp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
- HiGHS optimization software: https://highs.dev/

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
