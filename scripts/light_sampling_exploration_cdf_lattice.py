"""Exhaust the ideal 23-bit root-selector lattice without enumerating 2^23 samples."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import cohorts
from light_sampling_clustering_repair import EVIDENCE, scores
from light_sampling_exploration_coarse_upper import anchored_tree
from light_sampling_exploration_outer_mix import outer_mix
from light_sampling_point_moments import sha
from light_sampling_point_screening import Tree


N=1<<23
f=np.float32
TOP=f(1.0-2.**-24)


def inverse_replay(k, transitions):
    u=f((k+.5)/N)
    for norm,upper,previous,weight in transitions:
        t=min(f(u*norm),upper)
        u=min(TOP,max(f(0),f(f(t-previous)/weight)))
    return u


def lower_bound(lo,hi,cdf,norm,upper,transitions):
    while lo<hi:
        k=(lo+hi)//2
        t=min(f(inverse_replay(k,transitions)*norm),upper)
        if t<cdf:
            lo=k+1
        else:
            hi=k
    return lo


def exhaust(tree,q,return_intervals=False):
    endpoints=[]
    guarded=0
    def walk(i,lo,hi,history,interval_mass):
        nonlocal guarded
        children=tree.children[i]
        if len(children)==0 or q[children].sum()==0:
            endpoints.append((i,lo,hi,"leaf" if len(children)==0 else "null"))
            return
        weights=np.zeros(4,dtype=np.float32)
        weights[:len(children)]=q[children]/q[i]
        cdf=np.cumsum(weights,dtype=np.float32)
        total=cdf[-1]
        positive=weights[weights>0]
        if interval_mass < f(2.**-8) or f(interval_mass*min(positive)) < f(total*f(2.**-20)):
            endpoints.append((i,lo,hi,"fresh"))
            return
        guarded+=1
        upper=np.nextafter(total,f(0))
        left=lo
        for k,c in enumerate(children):
            right=lower_bound(left,hi,cdf[k],total,upper,history)
            if weights[k]>0:
                assert right>left,(int(c),float(q[c]),int(left),int(right))
                previous=f(0) if k==0 else cdf[k-1]
                walk(c,left,right,(*history,(total,upper,previous,weights[k])),
                     f(interval_mass*f(weights[k]/total)))
            else:
                assert right==left
            left=right
        assert left==hi
    walk(0,0,N,(),f(1))
    probs=np.asarray([q[i] for i,*_ in endpoints])
    counts=np.asarray([hi-lo for _,lo,hi,_ in endpoints])
    assert counts.sum()==N and (counts>0).all()
    np.testing.assert_allclose(probs.sum(),1,atol=2e-12,rtol=0)
    actual=counts/N
    if return_intervals:
        return endpoints
    return dict(guarded_nodes=guarded,terminal_intervals=len(counts),min_terminal_bins=int(counts.min()),
                tv_distance=float(abs(actual-probs).sum()/2),
                max_relative_terminal_error=float(abs(actual/probs-1).max()),
                fresh_mass=sum(float(q[i]) for i,_,_,kind in endpoints if kind=="fresh"))


def main(output):
    refs=[EVIDENCE/f"20261108-{tag}-reference.npz" for tag in
          ("clustering-repair","cluster-summaries","nearfar-combination")]
    combo=EVIDENCE/"20261108-nearfar-combination.json"
    cohort=json.loads(combo.read_text(encoding="utf8"))["interior_m2"]["cohort"]
    rows=[]
    with np.load(refs[0]) as fresh,np.load(refs[1]) as old,np.load(refs[2]) as mix:
        for group,_,keys,root,windows in cohorts(fresh,old,mix,cohort):
            tree=Tree(anchored_tree(root)[0])
            cache={}
            for point,window,label,origin,normal,_ in windows:
                key=tuple(np.r_[origin,normal])
                if key not in cache:
                    q=outer_mix(tree,scores(tree,origin,normal),.75)
                    cache[key]=exhaust(tree,q)
                rows.append(dict(cohort=group,point=point,window=window,**cache[key]))
    result=dict(inputs={p.name:sha(p) for p in [*refs,combo]},rows=rows,
                scope="Exact midpoint partition for the given ideal 23-bit selector and this float32 traversal arithmetic. Weights are replayed CPU proposals, not bit-equivalent GPU weights. Fresh subtrees have the ordinary independent sampler's precision. No claim about actual PCG frequency or all geometries.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(dict(point_windows=len(rows),minimum_bins=min(r["min_terminal_bins"] for r in rows),
        max_tv=max(r["tv_distance"] for r in rows),max_relative=max(r["max_relative_terminal_error"] for r in rows),
        largest_fresh_mass=max(r["fresh_mass"] for r in rows))))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    main(p.parse_args().output)
