# Research Series — Decomposition and Advanced Mathematical Programming

This repository is one node in a broader set of decomposition and exact-optimization studies. The repositories remain separate because they target different mathematical structures, decomposition principles, and solver mechanisms.

## Core decomposition sequence

1. **benders-decomposition-capacity-planning** — classical multi-cut Benders / L-shaped decomposition for two-stage stochastic capacity planning; explicit master/subproblem structure and extensive-form verification.
2. **logic-based-benders-production-scheduling-python** — logic-based Benders for scheduling, where the subproblem is not simply represented through LP dual cuts.
3. **reinforcement-learning-benders-decomposition** — learning-guided control inside the Benders process; kept separate because the research question is solver intelligence rather than classical decomposition mechanics.
4. **column-generation-cutting-stock** — Dantzig–Wolfe/column-generation perspective with restricted master and pricing subproblem.
5. **airline-crew-scheduling-column-generation** — application of column generation to crew scheduling, with a structurally different pricing problem from cutting stock.
6. **learning-to-price-column-generation-cvrptw** — learned guidance for pricing in a routing/column-generation setting.
7. **lagrangian-relaxation-unit-commitment** — Lagrangian relaxation and unit-wise decomposition after dualizing coupling constraints.
8. **sddp-multistage-energy-storage** — multistage stochastic decomposition through value-function cuts.
9. **mpi-sppy-multistage-stochastic-planning** — distributed multistage stochastic programming / scenario decomposition tooling.
10. **matrix-free-pdhg-large-scale-linear-programming-python** — first-order large-scale LP optimization; not a decomposition method in the same sense, but part of the advanced large-scale optimization line.

## Why these stay separate

- **Benders** exploits complicating variables and recourse structure.
- **Logic-based Benders** replaces dual-based cuts with inference from a combinatorial subproblem.
- **Column generation** decomposes by variables/columns and solves a pricing problem.
- **Lagrangian relaxation** dualizes coupling constraints to obtain separable subproblems.
- **SDDP** approximates downstream value functions over multiple stochastic stages.
- **PDHG** targets large sparse optimization through first-order primal-dual iterations rather than structural decomposition.

Combining these repositories would obscure the differences between the algorithms and their proof/validation logic. They should instead be treated as a comparative advanced-OR series.

## Related solver-intelligence series

The repositories `learning-augmented-mip-solver`, `learning-to-branch-milp`, `learning-to-cut-milp`, `learning-to-presolve-mip`, `learning-to-prune-bnb-node-selection`, `neural-diving-mip-solution-prediction`, and `ml-warm-start-constraint-generation` form a separate learning-augmented solver series. They are related, but their central question is how ML controls an optimizer rather than how classical decomposition works.
