"""Bounded outer-layer Near/Far blend screen on archived conditional moments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import cohorts
from light_sampling_clustering_repair import EVIDENCE, key_order, scores
from light_sampling_point_screening import Tree


def outer_mix(tree, baseline, near_fraction):
    _, _, halves = baseline
    bounds = tree.nodes
    cross = ((bounds[:,:3]//16) != ((bounds[:,4:7]-1)//16)).any(axis=1)
    q = np.zeros(len(bounds))
    q[0] = 1.
    null = 0.
    for i, children in enumerate(tree.children):
        if not len(children) or q[i]==0:
            continue
        mix = near_fraction if cross[i] else .5
        conditional = 2.*(halves[children] @ [mix,1.-mix])
        if conditional.sum()==0:
            null += q[i]
        q[children] = q[i]*conditional
    np.testing.assert_allclose(q[tree.leaves].sum()+null,1.,rtol=0,atol=2e-12)
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
                    baseline = scores(tree,origin,normal)
                    cached[key] = {'prefix':baseline[0][order],
                        'outer-near-quarter':outer_mix(tree,baseline,.25)[order],
                        'outer-near-three-quarters':outer_mix(tree,baseline,.75)[order]}
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
                   rule='Only at cross-brick parents, replace 50/50 by 25/75 or 75/25 near/far. Keep sibling normalization and containing preference.',
                   rows=rows)
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
