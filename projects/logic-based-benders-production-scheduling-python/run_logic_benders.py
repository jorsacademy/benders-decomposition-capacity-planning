
from __future__ import annotations
import argparse, numpy as np
from lbbenders import generate_instance,solve_logic_benders,brute_force_global

def self_test():
    inst=generate_instance(seed=1,n_jobs=5,n_lines=2,n_families=2)
    b=solve_logic_benders(inst)
    e=brute_force_global(inst)
    assert abs(b.objective-e[0])<1e-7
    print("Logic-based Benders self-test: OK")

def main(a):
    vals=[]
    for i in range(a.instances):
        inst=generate_instance(a.seed+1009*i,a.jobs,a.lines,a.families)
        b=solve_logic_benders(inst)
        exact=brute_force_global(inst) if a.bruteforce else None
        gap=0 if exact is None else b.objective-exact[0]
        vals.append((b.iterations,len(b.cuts),b.objective,gap))
        print(f"instance={i} status={b.status} iter={b.iterations} cuts={len(b.cuts)} objective={b.objective:.3f} brute_gap={gap:.3e}")
    arr=np.asarray(vals,float)
    print(f"mean iterations={arr[:,0].mean():.2f} mean cuts={arr[:,1].mean():.2f}")

def parse():
    p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--jobs",type=int,default=7); p.add_argument("--lines",type=int,default=2)
    p.add_argument("--families",type=int,default=3); p.add_argument("--instances",type=int,default=8); p.add_argument("--bruteforce",action="store_true")
    return p.parse_args()
if __name__=="__main__":
    a=parse(); self_test() if a.self_test else main(a)
