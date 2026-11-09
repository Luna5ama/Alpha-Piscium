"""Test one fixed world lattice phase without camera-dependent grouping."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import brick_roots, cohorts
from light_sampling_clustering_repair import EVIDENCE, score
from light_sampling_moving_analysis import combine_world_children, leaves, ordered_signature
from light_sampling_point_screening import Tree
from light_sampling_world_layout import WorldLayout, oracle


def main(output):
    refs = [EVIDENCE/f'20261108-{tag}-reference.npz' for tag in
            ('clustering-repair','cluster-summaries','nearfar-combination')]
    combo_path = EVIDENCE/'20261108-nearfar-combination.json'
    combo = json.loads(combo_path.read_text(encoding='utf8'))['interior_m2']['cohort']
    rows, checked = [], {}
    with np.load(refs[0]) as fresh,np.load(refs[1]) as old,np.load(refs[2]) as mix:
        for cohort,_,keys,root,windows in cohorts(fresh,old,mix,combo):
            bricks = brick_roots(root)
            assert ordered_signature(oracle(bricks,combine_world_children,4))==ordered_signature(root)
            shifted = {tuple(v+1 for v in key):node for key,node in bricks.items()}
            candidate = oracle(shifted,combine_world_children,4)
            signature = ordered_signature(candidate)
            assert sorted(leaves(candidate))==sorted(leaves(root))
            bounds = np.array(list(shifted))
            starts = [range(int(hi)-15,int(lo)+1) for lo,hi in zip(bounds.min(0),bounds.max(0))]
            placements = 0
            for start in itertools.product(*starts):
                layout = WorldLayout(16,4,[x+8 for x in start])
                reconstructed,_ = layout.build(shifted,combine_world_children)
                assert ordered_signature(reconstructed)==signature
                placements += 1
            assert placements
            checked[cohort] = dict(valid_camera_brick_placements=placements,ordered_sha256=signature)
            methods = {'prefix':Tree(root),'one-brick-phase':Tree(candidate)}
            world_keys = list(map(tuple,keys))
            for point,window,label,origin,normal,moments in windows:
                values = {name:score(tree,world_keys,origin.astype(float),normal.astype(float),moments)[0]
                          for name,tree in methods.items()}
                v = values['one-brick-phase']['variance']
                ratio = v/values['prefix']['variance'] if v is not None else None
                rows.append(dict(cohort=cohort,point=point,window=window,label=label,methods=values,
                                 variance_ratio=ratio))
    assert len(rows)==28
    report = dict(scope='One deterministic +1 brick shift on each world key axis, no phase sweep. Actual leaf bounds are retained.',
                  cohort_layout_checks=checked,input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [*refs,combo_path]},rows=rows)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
