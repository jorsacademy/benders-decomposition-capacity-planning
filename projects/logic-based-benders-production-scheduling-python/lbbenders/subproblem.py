
from __future__ import annotations
from dataclasses import dataclass
import itertools, numpy as np

@dataclass(frozen=True)
class SubproblemResult:
    feasible: bool
    makespan: float
    line_completion: tuple
    sequences: tuple

def exact_line_sequence(instance,line,jobs):
    jobs=tuple(jobs)
    if not jobs:return 0.0,()
    best=float("inf"); bestseq=None
    for perm in itertools.permutations(jobs):
        total=0
        prev=None
        for j in perm:
            if prev is not None:
                total += int(instance.setup[line,instance.family[prev],instance.family[j]])
            total += int(instance.processing[j,line])
            prev=j
        if total<best:
            best=total; bestseq=perm
    return float(best),tuple(bestseq)

def solve_subproblem(instance,assignment):
    a=tuple(map(int,assignment))
    if len(a)!=instance.n_jobs or any(l<0 or l>=instance.n_lines for l in a):
        raise ValueError("assignment")
    comps=[]; seqs=[]
    for l in range(instance.n_lines):
        jobs=[j for j,x in enumerate(a) if x==l]
        c,s=exact_line_sequence(instance,l,jobs)
        comps.append(c); seqs.append(s)
        if c>instance.horizon[l]+1e-12:
            return SubproblemResult(False,float("inf"),tuple(comps),tuple(seqs))
    return SubproblemResult(True,max(comps,default=0.0),tuple(comps),tuple(seqs))
