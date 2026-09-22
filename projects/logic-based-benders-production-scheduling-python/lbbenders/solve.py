
from __future__ import annotations
from dataclasses import dataclass
import itertools, numpy as np
from .master import BendersCut,solve_master
from .subproblem import solve_subproblem

@dataclass(frozen=True)
class LogicBendersResult:
    assignment: tuple
    objective: float
    makespan: float
    assignment_cost: float
    iterations: int
    cuts: tuple
    status: str

def assignment_cost(instance,a):
    return float(sum(instance.assignment_cost[j,l] for j,l in enumerate(a)))

def solve_logic_benders(instance,max_iterations=500):
    # Safe conditional-cut M: upper bound on any feasible makespan.
    big_m=float(instance.processing.sum()+instance.setup.max()*instance.n_jobs+1)
    cuts=[]; incumbent=None
    for it in range(1,max_iterations+1):
        master=solve_master(instance,cuts,big_m)
        sub=solve_subproblem(instance,master.assignment)
        if not sub.feasible:
            cuts.append(BendersCut("nogood",master.assignment))
            continue
        ac=assignment_cost(instance,master.assignment)
        value=ac+sub.makespan
        if incumbent is None or value<incumbent[0]:
            incumbent=(value,master.assignment,sub.makespan,ac)
        if master.theta >= sub.makespan-1e-7:
            # master lower bound for this finite cut model meets evaluated subproblem;
            # with all previous conditional cuts, current master objective is global LB.
            return LogicBendersResult(incumbent[1],incumbent[0],incumbent[2],incumbent[3],
                                      it,tuple(cuts),"OPTIMAL")
        cuts.append(BendersCut("optimality",master.assignment,sub.makespan))
    raise RuntimeError("iteration limit")

def brute_force_global(instance):
    best=None
    for a in itertools.product(range(instance.n_lines),repeat=instance.n_jobs):
        sub=solve_subproblem(instance,a)
        if not sub.feasible: continue
        val=assignment_cost(instance,a)+sub.makespan
        if best is None or val<best[0]: best=(val,a,sub.makespan)
    if best is None: raise RuntimeError("no feasible assignment")
    return best
