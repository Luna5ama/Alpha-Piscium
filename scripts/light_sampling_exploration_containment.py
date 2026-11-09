"""Screen broad-box containment regularization on fixed, measured light moments."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import cohorts
from light_sampling_clustering_repair import EVIDENCE, key_order, scores
from light_sampling_exploration_capture import RUN
from light_sampling_exploration_coarse_upper import anchored_tree
from light_sampling_exploration_outer_mix import outer_mix
from light_sampling_moving_analysis import native_node
from light_sampling_point_moments import read, sha
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import load_tree


MODES=("near-floor","near-floor+front","near-floor+front-near","near-floor-normal-span","front-only")


def regularized(tree,origin,normal,mode,near_mix=.75):
    b=tree.nodes
    lo,hi=b[:,:3],b[:,4:7]
    half=(hi-lo)*.5
    delta=lo+half-origin
    axial=delta@normal
    span=half@abs(normal)
    front=axial+span>=-1e-6*((abs(delta*normal)).sum(1)+span+1)
    cos=np.clip((axial+np.linalg.norm(half,axis=1))/np.sqrt(np.maximum((delta**2).sum(1),.25)),.001,1.)
    amplitude=b[:,3]*front*cos
    near2=(np.maximum(abs(delta)-half,0.)**2).sum(1)
    far2=((abs(delta)+half)**2).sum(1)
    wide=((lo//16)!=((hi-1)//16)).any(1)
    volume=np.clip(np.divide(axial+span,2.*span,out=np.ones_like(span),where=span>0),1./16.,1.)
    if mode=="near-floor-normal-span":
        near2=np.where(wide,np.maximum(near2,span**2/3.),near2)
    elif mode!="front-only":
        near2=np.where(wide,np.maximum(near2,(half**2).sum(1)/3.),near2)
    near_flux=np.where(wide,amplitude*volume,amplitude) if "+front" in mode or mode=="front-only" else amplitude
    far_flux=np.where(wide,amplitude*volume,amplitude) if mode in ("near-floor+front","front-only") else amplitude
    near=np.divide(near_flux,near2,out=np.zeros_like(near2),where=near2>0)
    far=np.divide(far_flux,far2,out=np.zeros_like(far2),where=far2>0)
    zero=np.where(near2==0,near_flux,0.)
    q=np.zeros(len(b))
    q[0]=1.
    null=0.
    for i,k in enumerate(tree.children):
        if not len(k) or q[i]==0:
            continue
        f,n,z=far[k],near[k],zero[k]
        if f.sum()<=0.:
            null+=q[i]
            continue
        a=near_mix if wide[i] else .5
        if z.sum()>0:
            n=z
        assert n.sum()>0.
        q[k]=q[i]*((1.-a)*f/f.sum()+a*n/n.sum())
    np.testing.assert_allclose(q[tree.leaves].sum()+null,1.,rtol=0,atol=3e-12)
    return q


def selective_near(tree, origin, normal, threshold=0.125):
    """Recompute just the near endpoint of broad, mostly-behind containing boxes."""
    b = tree.nodes
    lo,hi = b[:,:3],b[:,4:7]
    half = (hi-lo)*.5
    delta = lo+half-origin
    axial,span = delta@normal,half@abs(normal)
    front = axial+span>=-1e-6*((abs(delta*normal)).sum(1)+span+1)
    cos = np.clip((axial+np.linalg.norm(half,axis=1)) /
                  np.sqrt(np.maximum((delta**2).sum(1),.25)),.001,1.)
    amplitude = b[:,3]*front*cos
    near2 = (np.maximum(abs(delta)-half,0.)**2).sum(1)
    far2 = ((abs(delta)+half)**2).sum(1)
    near = np.divide(amplitude,near2,out=np.zeros_like(amplitude),where=near2>0)
    far = np.divide(amplitude,far2,out=np.zeros_like(amplitude),where=far2>0)
    zero = np.where(near2==0.,amplitude,0.)
    wide = ((lo//16)!=((hi-1)//16)).any(1)
    gate = (near2==0.) & wide & (amplitude>0.) & ((axial+span)<2.*threshold*span)
    q = np.zeros(len(b))
    q[0] = 1.
    null = 0.
    for i,s in enumerate(tree.children):
        if not len(s) or q[i]==0.:
            continue
        f,n,z = far[s],near[s].copy(),zero[s].copy()
        for j,k in enumerate(s):
            children = tree.children[k]
            if len(children) and gate[k]:
                n[j],z[j] = near[children].sum(),zero[children].sum()
        if not f.sum():
            null += q[i]
            continue
        p = f/f.sum()
        if z.sum():
            n = z
        a = .75 if wide[i] else .5
        q[s] = q[i]*((1.-a)*p+a*n/n.sum() if n.sum() else p)
    np.testing.assert_allclose(q[tree.leaves].sum()+null,1.,rtol=0,atol=3e-12)
    return q


def evaluate(tree,order,origin,normal,moments):
    base=scores(tree,origin,normal)
    q0=outer_mix(tree,base,.75)
    choices={"anchor":q0,"selective-near":selective_near(tree,origin,normal),**{m:regularized(tree,origin,normal,m) for m in MODES}}
    m2=moments[:,3]
    mu=moments[:,:3].sum(0)
    result={}
    for name,p0 in choices.items():
        p=p0[order]
        assert not ((p==0)&(m2>0)).any(),name
        value=np.divide(m2,p,out=np.zeros_like(m2),where=p>0).sum()-mu@mu
        result[name]=dict(variance=float(value),observed_positive_mass=float(p[m2>0].sum()))
    for v in result.values():
        v["ratio_over_anchor"]=v["variance"]/result["anchor"]["variance"]
    return result


def main(output):
    refs=[EVIDENCE/f"20261108-{tag}-reference.npz" for tag in
          ("clustering-repair","cluster-summaries","nearfar-combination")]
    combo=EVIDENCE/"20261108-nearfar-combination.json"
    group_id=read(combo)["interior_m2"]["cohort"]
    rows=[]
    with np.load(refs[0]) as fresh,np.load(refs[1]) as old,np.load(refs[2]) as mix:
        for group,_,keys,root,windows in cohorts(fresh,old,mix,group_id):
            tree=Tree(anchored_tree(root)[0])
            order=key_order(tree,list(map(tuple,keys)))
            for point,window,label,origin,normal,moments in windows:
                rows.append(dict(cohort=group,point=point,window=window,
                                 methods=evaluate(tree,order,origin,normal,moments)))
    ls5=RUN/"cases/light-sampling-5"
    saved=ls5/"ls5-wall-mode3"
    ref=read(RUN/"ls5-wall-mode3-variance-replay.json")
    cam=ls5/"camera-paired-anchor"
    _,v,l=load_tree(cam/"before_tree.bin",6)
    native=Tree(native_node(v,l,1))
    keys=sorted(native.keys)
    _,v,l=load_tree(ls5/"dynamic-coarse-anchor/before_tree.bin",6)
    tree=Tree(native_node(v,l,1))
    order=key_order(tree,keys)
    for row in ref["rows"]:
        file=saved/f"moments/left-sandstone-wall-window{row['window']}.npz"
        with np.load(file) as raw:
            moments=np.c_[raw["mean_rgb"],raw["m2"]]
        methods=evaluate(tree,order,np.array(row["origin"]),np.array(row["normal"]),moments)
        np.testing.assert_allclose(methods["anchor"]["variance"],row["methods"]["anchor-75"]["variance"])
        for v in methods.values():
            v["ratio_over_camera"]=v["variance"]/row["methods"]["camera"]["variance"]
        rows.append(dict(cohort="wall/5",point=row["point"],window=row["window"],methods=methods))
    summary={}
    for m in (*MODES,"selective-near"):
        ratios=[x["methods"][m]["ratio_over_anchor"] for x in rows if x["cohort"]!="wall/5"]
        summary[m]=dict(outside_wall_min_median_p95_max=np.quantile(ratios,[0,.5,.95,1]).tolist(),
                        wall_over_camera=[x["methods"][m]["ratio_over_camera"] for x in rows if x["cohort"]=="wall/5"])
    result=dict(inputs={p.name:sha(p) for p in [*refs,combo,RUN/"ls5-wall-mode3-variance-replay.json"]},
                summary=summary,rows=rows,scope="CPU reweight only. Preserve complete native leaf support and exact fixed-input moments; width floor and projected front fraction are proposal heuristics, not visibility or observed attenuation.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(summary))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    main(p.parse_args().output)
