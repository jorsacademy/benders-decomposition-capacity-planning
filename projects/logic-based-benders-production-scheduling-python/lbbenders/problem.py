
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ProductionSchedulingInstance:
    processing: np.ndarray       # [J,L]
    family: np.ndarray           # [J]
    setup: np.ndarray            # [L,F,F]
    assignment_cost: np.ndarray  # [J,L]
    horizon: np.ndarray          # [L]

    def __post_init__(self):
        p=np.asarray(self.processing,int); c=np.asarray(self.assignment_cost,float)
        fam=np.asarray(self.family,int); s=np.asarray(self.setup,int); h=np.asarray(self.horizon,int)
        if p.ndim!=2 or c.shape!=p.shape or fam.shape!=(p.shape[0],): raise ValueError("shape")
        if s.shape[0]!=p.shape[1] or h.shape!=(p.shape[1],): raise ValueError("line shape")
        if np.any(p<=0) or np.any(s<0) or np.any(h<=0): raise ValueError("invalid")

    @property
    def n_jobs(self): return self.processing.shape[0]
    @property
    def n_lines(self): return self.processing.shape[1]

def generate_instance(seed=42,n_jobs=7,n_lines=2,n_families=3):
    rng=np.random.default_rng(seed)
    base=rng.integers(4,12,size=n_jobs)
    efficiency=rng.uniform(.8,1.25,size=n_lines)
    p=np.maximum(np.rint(base[:,None]*efficiency[None,:]),1).astype(int)
    fam=rng.integers(0,n_families,size=n_jobs)
    setup=np.zeros((n_lines,n_families,n_families),int)
    for l in range(n_lines):
        setup[l]=rng.integers(1,6,size=(n_families,n_families))
        np.fill_diagonal(setup[l],0)
    assign=rng.uniform(0,3,size=(n_jobs,n_lines))
    # Tight enough that poor concentration can be infeasible.
    horizon=np.full(n_lines,int(np.ceil(p.min(axis=1).sum()/n_lines*1.35)))
    return ProductionSchedulingInstance(p,fam,setup,assign,horizon)
