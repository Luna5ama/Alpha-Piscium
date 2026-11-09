"""Offline score of a local full-cube escape proxy with fixed leaf support."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_repair import EVIDENCE, control_root, score
from light_sampling_moving_analysis import Node, morton_encode
from light_sampling_point_moments import read
from light_sampling_point_screening import Tree

RUN = EVIDENCE.parents[1]/'builds/light-sampling/20261108-clustering-repair'


def escape_proxy(keys, original):
    accepted = read(original/'accepted.json')
    meta = np.fromfile(original/'before_voxel-meta.bin','<u4')
    materials = np.memmap(original/'before_voxel-materials.bin','<u4','r')
    origin = (np.asarray(accepted['pose']['camera_brick'])-8)*16
    allocation = np.full(4096,-1,dtype=int)
    bricks = meta[9217:9217+int(meta[0])]
    allocation[bricks] = np.arange(len(bricks))
    np.testing.assert_array_equal(meta[4097+bricks],np.arange(len(bricks)))
    emitters = np.unique(keys[:,3])
    def lookup(cells):
        local = cells-origin
        inside = ((local>=0)&(local<256)).all(axis=1)
        values = np.zeros(len(cells),dtype='<u4')
        ids = np.flatnonzero(inside)
        slot = allocation[morton_encode(local[ids]//16)]
        loaded = slot>=0
        selected = ids[loaded]
        values[selected] = materials[slot[loaded]*4096+morton_encode(local[selected]%16)]
        return values
    np.testing.assert_array_equal(lookup(keys[:,:3]) >> 1,keys[:,3])
    open_faces = np.zeros(len(keys),dtype=int)
    for shift in np.concatenate([np.eye(3,dtype=int),-np.eye(3,dtype=int)]):
        material = lookup(keys[:,:3]+shift)
        solid = ((material&1)!=0) | np.isin(material>>1,emitters)
        open_faces += ~solid
    return (1.+open_faces)/7.,open_faces,accepted['identity']


def reweight(node, factors):
    data = node.data.copy()
    if node.children:
        children = tuple(reweight(c,factors) for c in node.children)
        data[3] = sum(c.data[3] for c in children)
        return Node(data,node.material,children)
    key = (*map(int,data[:3]),node.material)
    data[3] *= factors[key]
    return Node(data,node.material,())


def main(output):
    source = EVIDENCE/'20261108-clustering-repair-reference.npz'
    results = {}
    with np.load(source) as ref:
        metadata = json.loads(str(ref['metadata']))
        for slot in (1,5):
            labels = metadata[str(slot)]
            root = control_root(ref,f'{slot}/prefix/')
            keys = ref[f'{slot}/keys']
            original = next((RUN/'cases'/f'light-sampling-{slot}').glob('m2-*'))
            factors, faces, identity = escape_proxy(keys,original)
            assert identity == labels['identity']
            key_list = list(map(tuple,keys))
            trees = {'prefix':Tree(root),'local-escape':Tree(reweight(root,dict(zip(key_list,factors,strict=True))))}
            rows = []
            for i,w in enumerate(labels['windows']):
                origin,normal = np.asarray(w['receiver_audit'])[:2,:3]
                moments = ref[f'{slot}/{i}/moments']
                measures = {n:score(t,key_list,origin,normal,moments)[0] for n,t in trees.items()}
                rows.append(dict(point=w['point'],window=w['window'],methods=measures,
                                 variance_ratio=measures['local-escape']['variance']/measures['prefix']['variance']))
            results[str(slot)] = dict(open_face_histogram=np.bincount(faces,minlength=7).tolist(),rows=rows)
    result = dict(scope='Full-cube neighbor occupancy only. The proposal retains at least one seventh of each old mass. Physical emission and all world keys are unchanged.',
                  input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scenes=results)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
