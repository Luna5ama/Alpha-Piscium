"""Quantify removal of cube faces hidden by an adjacent solid voxel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_repair import EVIDENCE, control_root, score
from light_sampling_exploration_escape import RUN, neighbor_solids
from light_sampling_geometry import cube_faces
from light_sampling_point_screening import Tree


def retained_face_probability(keys, solids, origin, normal):
    retain = np.ones(len(keys))
    per_lamp = np.zeros((len(keys),2),dtype=int)
    for i, key in enumerate(keys):
        low = key[:3]
        # If every facing cube face has an open neighbor there is nothing to remove.
        side = np.where(origin > low, np.arange(3), np.arange(3)+3)
        eligible = (origin<low)|(origin>low+1.)
        blocked = eligible & solids[i,side]
        if not blocked.any():
            continue
        faces = cube_faces(low.copy(),origin,normal)
        total = sum(f['weight'] for f in faces)
        if not total:
            continue
        kept = sum(f['weight'] for f in faces if not blocked[f['axis']])
        retain[i] = kept/total
        per_lamp[i] = [len(faces),sum(blocked[f['axis']] for f in faces)]
    assert np.isfinite(retain).all() and np.all((retain>=0)&(retain<=1))
    return retain,per_lamp


def main(output):
    reference = EVIDENCE/'20261108-clustering-repair-reference.npz'
    report = dict(input_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  scope='Analytic face-selection proposal and archived voxel neighbors. M2 reduction assumes each removed face contributes zero.',
                  cases={})
    with np.load(reference) as ref:
        metadata = json.loads(str(ref['metadata']))
        for slot in (1,5):
            name = str(slot)
            keys = ref[f'{slot}/keys']
            labels = metadata[name]
            base = Tree(control_root(ref,f'{slot}/prefix/'))
            original = next((RUN/'cases'/f'light-sampling-{slot}').glob('m2-*'))
            blocked,identity = neighbor_solids(keys,original)
            assert identity==labels['identity']
            cached = {}
            rows = []
            for i,window in enumerate(labels['windows']):
                point = window['point']
                origin,normal = np.asarray(window['receiver_audit'])[:2,:3]
                if point not in cached:
                    cached[point] = retained_face_probability(keys,blocked,origin,normal)
                keep,faces = cached[point]
                moments = ref[f'{slot}/{i}/moments']
                result,p,*_ = score(base,list(map(tuple,keys)),origin,normal,moments)
                m2 = moments[:,3]
                impossible = np.flatnonzero((keep==0)&(m2>0))
                delta = np.divide((1.-keep)*m2,p,out=np.zeros_like(m2),where=p>0)
                top = np.argsort(-delta)[:6]
                rows.append(dict(point=point,window=window['window'],base_variance=result['variance'],
                                 removed_face_mass_count=int((keep<1).sum()),
                                 lamps_completely_removed=int((keep==0).sum()),
                                 positive_m2_fully_removed=[keys[i].tolist() for i in impossible],
                                 variance_ratio=float(1.-delta.sum()/result['variance']),
                                 largest=[dict(world_key=keys[i].tolist(),kept=float(keep[i]),
                                               total_and_removed_faces=faces[i].tolist(),
                                               variance_reduction=float(delta[i]))
                                          for i in top if delta[i]>0]))
            report['cases'][name] = rows
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
