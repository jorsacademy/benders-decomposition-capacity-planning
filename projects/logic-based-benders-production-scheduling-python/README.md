# Logic-Based Benders for Production Assignment + Scheduling

A decomposition project separating aggregate production assignment from detailed sequence-dependent scheduling.

```text
MASTER MILP
job -> production line assignment
+ makespan lower bound
        ↓
SUBPROBLEM
exact sequence-dependent setup scheduling on each line
        ↓
logic feasibility / optimality cuts
        ↓
repeat until lower bound meets evaluated schedule
```

## Master problem

Binary `x[j,l]` assigns each job to exactly one production line.

The master contains two valid scheduling relaxations:

- line processing workload must not exceed the line horizon;
- `theta` is at least every line's processing workload.

Sequence-dependent setup times are deliberately omitted from the master and handled by the detailed scheduling subproblem.

The master objective is:

```text
assignment cost + theta
```

and is solved with SciPy `milp` / HiGHS.

## Exact scheduling subproblem

For each line, every permutation of its assigned jobs is evaluated. Completion time includes line-specific processing times and family-to-family sequence-dependent setups.

This is exact for the small declared line assignment. If a line's best possible sequence exceeds its horizon, the assignment is infeasible.

## Logic-based cuts

Infeasible assignment:

```text
sum_j x[j, assigned_line_j] <= J - 1
```

forbids that exact assignment.

Feasible assignment with scheduling value `Q(a)` receives a conditional optimality cut:

```text
theta >= Q(a) - M * HammingDistance(x,a)
```

with a safe makespan upper bound `M`.

At the visited assignment the cut enforces the exact subproblem value; away from it, the big-M term makes the cut nonrestrictive.

## Exactness

The assignment set is finite. The master objective is a valid lower bound because workload bounds and all generated logic cuts are valid. When the current master assignment has `theta >= Q(a)`, its evaluated objective matches the master lower bound and global optimality is proved for the declared finite model.

Regression tests independently enumerate **all line assignments** and require the Logic-Based Benders result to match the global brute-force optimum.

## Development run

Six random problems, 7 jobs × 2 lines:

```text
instance    iterations    cuts    brute-force gap
0               17        16          0
1                8         7          0
2                4         3          0
3                8         7          0
4               11        10          0
5                8         7          0

mean iterations = 9.33
mean cuts       = 8.33
```

Before the valid workload relaxation was added, the point cuts alone could force enumeration of nearly every assignment. The stronger master reduced the same development problems to the iteration counts above.

## GitHub Actions validation

A GitHub-hosted Ubuntu 24.04 runner validated the repository on:

```text
Python  3.12.14
NumPy   2.5.2
SciPy   1.18.1
```

The remote regression suite passed all **4/4 tests**.

The CI smoke configuration used three independently generated problems with:

```text
jobs       6
lines      2
families   3
```

Runner-observed result:

```text
instance   status    iterations   cuts   objective   brute-force gap
0          OPTIMAL        3         2      30.911       0.000e+00
1          OPTIMAL        5         4      23.835       0.000e+00
2          OPTIMAL       12        11      33.058       0.000e+00

mean iterations = 6.67
mean cuts       = 5.67
```

Every runner-side Benders objective matched the complete assignment-enumeration oracle exactly for these smoke instances. These values validate the implementation on the declared small model; they are not a scalability or industrial-performance claim.

## Tests

- exact line sequencing vs explicit permutation enumeration;
- Logic-Based Benders vs complete assignment enumeration on multiple fixtures;
- infeasible concentrated assignment detection;
- master assignment validity.

## Run

```bash
pip install -r requirements.txt
python run_logic_benders.py --self-test
python -m unittest discover -s tests -v
python run_logic_benders.py --bruteforce
```

## Scope

This is a small exact decomposition demonstrator. It is not presented as a generic industrial branch-and-check engine. Exact permutation subproblems and complete assignment verification are intentionally limited to small fixtures.
