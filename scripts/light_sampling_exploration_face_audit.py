"""Audit the implemented face mixture against voxel readback and area PDF."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_exploration_escape import RUN, neighbor_words
from light_sampling_geometry import cube_faces, cube_pdf
from light_sampling_moving_analysis import leaves, native_node
from light_sampling_point_moments import read
from light_sampling_weight_reference import load_tree


def main(folder, quality):
    case = read(quality/'accepted.json')
    audit = read(folder/'accepted.json')
    assert case['identity']['world_voxel_sha256']==audit['identity']['world_voxel_sha256']
    assert case['identity']['leaf_sha256']==audit['identity']['leaf_sha256']
    assert case['identity']['ordered_topology_sha256']==audit['identity']['ordered_topology_sha256']
    scene = case['scene']
    fixture = next((RUN/'cases'/scene).glob('m2-*'))
    brick = read(fixture/'accepted.json')['pose']['camera_brick']
    header,nodes,links = load_tree(quality/'before_tree.bin',1)
    keys = np.asarray([k[:4] for k in sorted(leaves(native_node(nodes,links,1)))],dtype=int)
    assert len(keys)==int(header[1]) and header[2]==header[3]==0
    words,_ = neighbor_words(keys,quality,brick)
    materials = words>>1
    solids = ((words&1)!=0) | np.isin(materials,np.unique(keys[:,3]))
    solids &= materials!=keys[:,3,None]
    indices = {tuple(key):i for i,key in enumerate(keys)}
    records = np.fromfile(folder/'audit.bin','<f4')[:1024].reshape(64,4,4)
    errors, changed = [], 0
    for origin,normal,ray,low in records:
        if ray[3] <= 0.:
            continue
        material = int(origin[3].view('u4')) & 0x7fffffff
        key = (*map(int,low[:3]),material)
        i = indices[key]
        origin,normal,ray,low = (a[:3].astype(float) for a in (origin,normal,ray,low))
        faces = cube_faces(low,origin,normal)
        old_total = sum(f['weight'] for f in faces)
        selected = int(np.argmax(np.minimum((low-origin)/ray,(low+1.-origin)/ray)))
        def covered(face):
            axis=face['axis']
            return bool(solids[i,axis+(0 if origin[axis]>low[axis] else 3)])
        keep_total = sum(f['weight'] for f in faces if not covered(f))
        ratio = 1.
        if 0. < keep_total < old_total:
            chosen = next(f for f in faces if f['axis']==selected)
            ratio = .25 + (0. if covered(chosen) else .75*old_total/keep_total)
            changed += 1
        expected = cube_pdf(low,origin,normal,ray)*ratio
        errors.append(expected)
    actual = records[:,2,3]
    observed = actual[actual>0].astype(float)
    assert len(observed)==len(errors) and len(errors)>0
    rel = np.abs(observed/np.asarray(errors)-1)
    assert np.isfinite(rel).all() and rel.max()<.005,rel
    print(json.dumps(dict(samples=len(errors),adjusted_samples=changed,maximum_relative_error=float(rel.max()))))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audit_case',type=Path)
    parser.add_argument('quality_case',type=Path)
    opts = parser.parse_args()
    main(opts.audit_case,opts.quality_case)
