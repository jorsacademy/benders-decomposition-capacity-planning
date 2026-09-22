
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse

@dataclass(frozen=True)
class BendersCut:
    kind: str  # "nogood" or "optimality"
    assignment: tuple
    value: float = 0.0

@dataclass(frozen=True)
class MasterResult:
    assignment: tuple
    theta: float
    objective: float

def hamming_coefficients(assignment,n_lines):
    # d(x,a) = sum_j (1 - x[j,a_j]) because exact-one constraints hold.
    # conditional theta >= q - M*d => theta - M*sum x[j,a_j] >= q-M*J
    J=len(assignment)
    idx=[j*n_lines+assignment[j] for j in range(J)]
    return idx

def solve_master(instance,cuts,big_m):
    J,L=instance.n_jobs,instance.n_lines
    n=J*L+1; theta_idx=n-1
    c=np.zeros(n)
    c[:J*L]=instance.assignment_cost.reshape(-1)
    c[theta_idx]=1.0
    integrality=np.zeros(n,int); integrality[:J*L]=1
    lb=np.zeros(n); ub=np.ones(n); ub[theta_idx]=np.inf
    constraints=[]
    # each job exactly one line
    Aeq=np.zeros((J,n))
    for j in range(J): Aeq[j,j*L:(j+1)*L]=1
    constraints.append(LinearConstraint(sparse.csr_matrix(Aeq),np.ones(J),np.ones(J)))

    # Valid master relaxation from processing times only. Sequence-dependent
    # setups remain in the scheduling subproblem.
    base_rows=[]; base_lo=[]; base_hi=[]
    for l in range(L):
        # Necessary horizon feasibility: sum processing on a line <= horizon.
        row=np.zeros(n)
        for j in range(J):
            row[j*L+l]=instance.processing[j,l]
        base_rows.append(row); base_lo.append(-np.inf); base_hi.append(instance.horizon[l])

        # Makespan lower bound: theta >= line processing workload.
        row2=np.zeros(n)
        for j in range(J):
            row2[j*L+l]=-instance.processing[j,l]
        row2[theta_idx]=1.0
        base_rows.append(row2); base_lo.append(0.0); base_hi.append(np.inf)
    constraints.append(LinearConstraint(
        sparse.csr_matrix(np.asarray(base_rows)),
        np.asarray(base_lo),np.asarray(base_hi)
    ))

    rows=[]; lows=[]; highs=[]
    for cut in cuts:
        idx=hamming_coefficients(cut.assignment,L)
        row=np.zeros(n)
        if cut.kind=="nogood":
            # sum_j x[j,a_j] <= J-1
            row[idx]=1; rows.append(row); lows.append(-np.inf); highs.append(J-1)
        else:
            # theta - M*sum_j x[j,a_j] >= q - M*J
            row[theta_idx]=1; row[idx]=-big_m
            rows.append(row); lows.append(cut.value-big_m*J); highs.append(np.inf)
    if rows:
        constraints.append(LinearConstraint(sparse.csr_matrix(np.asarray(rows)),np.asarray(lows),np.asarray(highs)))

    r=milp(c,integrality=integrality,bounds=Bounds(lb,ub),constraints=constraints,
           options={"disp":False})
    if not r.success: raise RuntimeError(r.message)
    x=r.x[:J*L].reshape(J,L)
    a=tuple(np.argmax(x,axis=1).tolist())
    return MasterResult(a,float(r.x[theta_idx]),float(r.fun))
