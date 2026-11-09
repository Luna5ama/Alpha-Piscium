"""Estimate root-stratification headroom from fixed native conditional moments."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import cohorts
from light_sampling_clustering_repair import EVIDENCE, key_order, scores
from light_sampling_exploration_coarse_upper import anchored_tree
from light_sampling_exploration_outer_mix import outer_mix
from light_sampling_point_moments import sha
from light_sampling_point_screening import Tree


def analyze(tree, order, q, moments, count):
    p=q[order]
    m2=moments[:,3]
    assert not ((m2>0)&(p<=0)).any()
    value=np.zeros((len(q),3))
    value[order]=moments[:,:3]
    energy=np.zeros(len(q))
    energy[order]=np.divide(m2,p,out=np.zeros_like(m2),where=p>0)
    for i in reversed(range(len(q))):
        children=tree.children[i]
        if len(children):
            value[i]=value[children].sum(axis=0)
            energy[i]=energy[children].sum()
    total=value[0]
    iid=energy[0]-total@total
    children=tree.children[0]
    if len(children)==0 or q[children].sum()==0:
        return dict(iid_variance=float(iid),variance_reduction=0.,root_probabilities=[])
    root_prob=q[children]
    mu_given_root=np.divide(value[children],root_prob[:,None],out=np.zeros_like(value[children]),
                            where=root_prob[:,None]>0)
    np.testing.assert_allclose(root_prob.sum(),1.,rtol=0,atol=2e-12)
    end=np.cumsum(root_prob)
    begin=np.r_[0.,end[:-1]]
    floor=np.arange(count)/count
    ceil=floor+1./count
    overlap=np.maximum(0.,np.minimum(ceil[:,None],end)-np.maximum(floor[:,None],begin))
    np.testing.assert_allclose(overlap.sum(1)*count,1.,rtol=0,atol=2e-12)
    mu_given_stratum=(count*overlap)@mu_given_root
    np.testing.assert_allclose(mu_given_stratum.mean(0),total,rtol=0,atol=1e-9)
    removed=np.square(mu_given_stratum-total).sum(axis=1).mean()
    assert iid>0 and 0<=removed<=iid
    return dict(iid_variance=float(iid),iid_mean32_variance=float(iid/count),
                stratified_mean32_variance=float((iid-removed)/count),
                variance_reduction=float(removed/iid),root_probabilities=root_prob.tolist(),
                root_mean_rgb=mu_given_root.tolist())


def main(output,count):
    refs=[EVIDENCE/f"20261108-{tag}-reference.npz" for tag in
          ("clustering-repair","cluster-summaries","nearfar-combination")]
    combo=EVIDENCE/"20261108-nearfar-combination.json"
    cohort=json.loads(combo.read_text(encoding="utf8"))["interior_m2"]["cohort"]
    rows=[]
    with np.load(refs[0]) as fresh,np.load(refs[1]) as old,np.load(refs[2]) as mix:
        for group,_,keys,root,windows in cohorts(fresh,old,mix,cohort):
            trees={"prefix":Tree(root),"anchor":Tree(anchored_tree(root)[0])}
            order={k:key_order(t,list(map(tuple,keys))) for k,t in trees.items()}
            for point,window,label,origin,normal,moments in windows:
                methods={}
                for name,tree in trees.items():
                    native=scores(tree,origin,normal)
                    q=outer_mix(tree,native,.75) if name=="anchor" else native[0]
                    methods[name]=analyze(tree,order[name],q,moments,count)
                rows.append(dict(cohort=group,point=point,window=window,label=label,methods=methods))
    result=dict(inputs={p.name:sha(p) for p in [*refs,combo]},strata=count,rows=rows,
                scope="Analytic variance removed from independent fixed-input means by exactly one root draw per equal-width stratum. Leaf and within-light draws remain independent in the model. Archive point means have sampling error; no moving, spatially pooled or GPU performance claim.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({m:[float(v) for v in np.quantile([r["methods"][m]["variance_reduction"] for r in rows],[0,.5,.95,1])]
                     for m in ("prefix","anchor")}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("--strata",type=int,default=32)
    args=p.parse_args()
    main(args.output,args.strata)
