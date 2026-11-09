"""Compare one endpoint lookahead with ordinary local Near/Far."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import cohorts
from light_sampling_clustering_repair import EVIDENCE, key_order, scores
from light_sampling_point_screening import Tree


def endpoints(tree, origin, normal, base):
    bounds = tree.nodes
    half = (bounds[:,4:7]-bounds[:,:3])*.5
    delta = bounds[:,:3]+half-origin
    height = delta @ normal
    extent = half @ np.abs(normal)
    front = height+extent >= -1e-6*(np.abs(delta*normal).sum(1)+extent+1.)
    amplitude = bounds[:,3]*front*np.clip((height+np.linalg.norm(half,axis=1)) /
                                         np.sqrt(np.maximum((delta*delta).sum(1),.25)),.001,1.)
    d_near = (np.maximum(np.abs(delta)-half,0)**2).sum(1)
    d_far = ((np.abs(delta)+half)**2).sum(1)
    native = np.column_stack((np.divide(amplitude,d_near,out=np.zeros_like(amplitude),where=d_near>0),
                              amplitude/d_far,np.where(d_near==0,amplitude,0.)))
    cross = ((bounds[:,:3]//16) != ((bounds[:,4:7]-1)//16)).any(1)
    expanded = native.copy()
    for i,child in enumerate(tree.children):
        if len(child) and cross[i]:
            expanded[i] = native[child].sum(axis=0)
    output = {'prefix':base}
    for name,values in {
        'near-endpoint':np.column_stack((expanded[:,0],native[:,1],expanded[:,2])),
        'far-endpoint':np.column_stack((native[:,0],expanded[:,1],native[:,2])),
        'both-endpoints':expanded,
    }.items():
        reach = np.zeros(len(bounds))
        reach[0] = 1.
        null = 0.
        for i,child in enumerate(tree.children):
            if not len(child) or reach[i]==0:
                continue
            sample = values[child]
            far,zero = sample[:,1],sample[:,2]
            near = zero if zero.sum()>0 else sample[:,0]
            # Preserve support in mismatched groups when one endpoint is null.
            # This is the normalization for this experimental proposal.
            if far.sum()==0. and near.sum()==0.:
                null += reach[i]
                continue
            if far.sum()==0.:
                weights = near/near.sum()
            elif near.sum()==0.:
                weights = far/far.sum()
            else:
                weights = .5*(near/near.sum()+far/far.sum())
            reach[child] = reach[i]*weights
        np.testing.assert_allclose(reach[tree.leaves].sum()+null,1.,rtol=0,atol=2e-12)
        output[name] = reach
    return output


def main(output):
    refs = [EVIDENCE/f'20261108-{tag}-reference.npz' for tag in
            ('clustering-repair','cluster-summaries','nearfar-combination')]
    combo = json.loads((EVIDENCE/'20261108-nearfar-combination.json').read_text(encoding='utf8'))['interior_m2']['cohort']
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
                    base = scores(tree,origin,normal)[0]
                    cached[key] = {n:q[order] for n,q in endpoints(tree,origin,normal,base).items()}
                m2 = moments[:,3]
                mu = moments[:,:3].sum(0)
                result = {}
                for name,p in cached[key].items():
                    missing = int(((m2>0)&(p==0)).sum())
                    v = np.divide(m2,p,out=np.zeros_like(m2),where=p>0).sum()-mu@mu
                    result[name] = dict(variance=float(v) if not missing else None,missing_positive_support=missing)
                base = result['prefix']['variance']
                for v in result.values():
                    v['ratio_vs_prefix'] = v['variance']/base if v['variance'] is not None else None
                rows.append(dict(cohort=cohort,point=point,window=window,label=label,methods=result))
    assert len(rows)==28
    output.write_text(json.dumps(dict(scope='Experimental one endpoint lookahead on cross-brick children.',
                                      rows=rows),indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
