"""Score a smooth near-distance proposal on the archived per-lamp moments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import cohorts
from light_sampling_clustering_repair import EVIDENCE, key_order, scores
from light_sampling_point_screening import Tree


def smooth_pmf(tree, origin, normal, soften_all):
    nodes = tree.nodes
    lo,hi = nodes[:,:3],nodes[:,4:7]
    half = (hi-lo)*.5
    delta = lo+half-origin
    h = delta @ normal
    extent = half @ np.abs(normal)
    front = h+extent >= -1e-6*(np.abs(delta*normal).sum(1)+extent+1)
    radius2 = (half*half).sum(1)
    cosine = np.clip((h+np.sqrt(radius2))/np.sqrt(np.maximum((delta*delta).sum(1),.25)),.001,1)
    amplitude = nodes[:,3]*cosine*front
    nearest = np.maximum(np.abs(delta)-half,0)
    near2 = (nearest*nearest).sum(1)
    far2 = ((np.abs(delta)+half)**2).sum(1)
    soften = ((lo//16) != ((hi-1)//16)).any(1) if not soften_all else np.ones(len(lo),dtype=bool)
    # One-axis variance of a uniform isotropic proxy for the box.
    den = near2 + np.where(soften,radius2/3,0)
    near = np.divide(amplitude,den,out=np.zeros_like(amplitude),where=den>0)
    far = amplitude/far2
    zero = np.where(den==0,amplitude,0)
    q = np.zeros(len(nodes))
    q[0] = 1.
    null = 0.
    for i, children in enumerate(tree.children):
        if not len(children) or q[i]==0:
            continue
        f = far[children]
        if f.sum()==0:
            null += q[i]
            continue
        n = zero[children] if np.any(zero[children]) else near[children]
        q[children] = q[i]*(.5*(n/n.sum()+f/f.sum()))
    np.testing.assert_allclose(q[tree.leaves].sum()+null,1,rtol=0,atol=2e-12)
    return q


def main(output):
    refs = [EVIDENCE/f'20261108-{tag}-reference.npz' for tag in
            ('clustering-repair','cluster-summaries','nearfar-combination')]
    combo_path = EVIDENCE/'20261108-nearfar-combination.json'
    combo = json.loads(combo_path.read_text(encoding='utf8'))['interior_m2']['cohort']
    rows = []
    with np.load(refs[0]) as fresh,np.load(refs[1]) as old,np.load(refs[2]) as mix:
        for cohort,_,keys,root,windows in cohorts(fresh,old,mix,combo):
            tree = Tree(root)
            order = key_order(tree,list(map(tuple,keys)))
            cached = {}
            for point,window,label,origin,normal,moments in windows:
                origin,normal = origin.astype(float),normal.astype(float)
                key = tuple(np.r_[origin,normal])
                if key not in cached:
                    cached[key] = {
                        'prefix':scores(tree,origin,normal)[0][order],
                        'soft-cross-brick':smooth_pmf(tree,origin,normal,False)[order],
                        'soft-all':smooth_pmf(tree,origin,normal,True)[order],
                    }
                m2 = moments[:,3]
                mu = moments[:,:3].sum(0)
                result = {}
                for name,p in cached[key].items():
                    missing = int(((m2>0)&(p==0)).sum())
                    v = np.divide(m2,p,out=np.zeros_like(m2),where=p>0).sum()-mu@mu
                    result[name] = dict(variance=float(v) if not missing else None,
                                        missing_positive_support=missing)
                base = result['prefix']['variance']
                for value in result.values():
                    v = value['variance']
                    value['ratio_vs_prefix'] = v/base if v is not None else None
                rows.append(dict(cohort=cohort,point=point,window=window,label=label,methods=result))
    assert len(rows)==28
    payload = dict(input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*refs,combo_path]},
                   rule='Near distance squared adds one axis variance of a sphere-shaped box proxy, r^2/3. Far half unchanged.',
                   rows=rows)
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
