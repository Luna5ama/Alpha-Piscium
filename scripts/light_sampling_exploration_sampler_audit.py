"""Reconstruct the GPU's selected leaf PMF and conditional cube PDF."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_repair import scores
from light_sampling_exploration_endpoints import endpoints
from light_sampling_exploration_outer_mix import outer_mix
from light_sampling_moving_analysis import native_node
from light_sampling_point_moments import read, sha
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import cube_pdf, load_tree


def main(folder, output, proposal):
    receipt = read(folder/'accepted.json')
    assert receipt['settings']['SETTING_DEBUG_LIGHT_AUDIT']=='true'
    model = int(receipt['settings']['SETTING_DEBUG_LIGHT_WEIGHT_MODEL'])
    assert model == (11 if proposal=='lookahead' else 6)
    header,nodes,links = load_tree(folder/'before_tree.bin',model)
    tree = Tree(native_node(nodes,links,1))
    assert len(tree.leaves)==int(header[1]) and header[2]==header[3]==0
    indices = dict(zip(tree.keys,tree.leaves,strict=True))
    data = np.fromfile(folder/'audit.bin','<f4',count=1024).reshape(64,4,4)
    leaves, cubes, nulls = [], [], 0
    for row in data:
        status = row[3,3]
        if status==0.:
            continue
        origin, normal = row[0,:3].astype(float),row[1,:3].astype(float)
        base = scores(tree,origin,normal)
        if proposal=='lookahead':
            pmf = endpoints(tree,origin,normal,base[0])['both-endpoints']
        elif proposal=='outer-near':
            pmf = outer_mix(tree,base,.75)
        else:
            pmf = base[0]
        if status<0.:
            assert 1.-pmf[tree.leaves].sum()>0.
            nulls += 1
            continue
        key = (*map(int,row[3,:3]),int(row[0,3].view('<u4'))&0x7fffffff)
        p = pmf[indices[key]]
        assert p>0.
        leaves.append(abs(float(row[1,3])/p-1.))
        if row[2,3]>0.:
            c = cube_pdf(row[3,:3],origin,normal,row[2,:3])
            cubes.append(abs(float(row[2,3])/c-1.))
    assert leaves and cubes and max(leaves)<.01 and max(cubes)<.005, (leaves,cubes)
    result = dict(proposal=proposal,job=receipt['job'],source=receipt['source'],emitters=int(header[1]),
                  selected_leaf_samples=len(leaves),cube_samples=len(cubes),null_selections=nulls,
                  leaf_pdf_max_relative_error=max(leaves),cube_pdf_max_relative_error=max(cubes),
                  audit_sha256=sha(folder/'audit.bin'),tree_sha256=sha(folder/'before_tree.bin'))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--proposal',choices=('prefix','outer-near','lookahead'),required=True)
    a=parser.parse_args()
    main(a.folder,a.output,a.proposal)
