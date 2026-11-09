"""Verify each observed GPU Cut2 selection against a direct flat-cut probability."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_geometry import cube_pdf
from light_sampling_moving_analysis import native_node, ordered_signature
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import load_tree, probabilities


def path_probability(tree, leaf, origin, normal):
    """Expand eligible child clusters explicitly, avoiding the GPU grouping logic."""
    bounds = tree.nodes
    crosses = ((bounds[:,:3]//16) != ((bounds[:,4:7]-1)//16)).any(axis=1)
    path = [leaf]
    while tree.parents[path[-1]] >= 0:
        path.append(int(tree.parents[path[-1]]))
    path.reverse()
    cursor = 0
    p = 1.
    while cursor < len(path)-1:
        parent, child = path[cursor:cursor+2]
        expands = bool(crosses[child] and len(tree.children[child]))
        cut = [int(entry)
               for candidate in tree.children[parent]
               for entry in (tree.children[candidate] if crosses[candidate] and len(tree.children[candidate])
                             else (candidate,))]
        cursor += 2 if expands else 1
        distribution = probabilities(bounds[cut],6,origin,normal)
        np.testing.assert_allclose(distribution.sum(),1.,rtol=0,atol=2e-12)
        p *= distribution[cut.index(path[cursor])]
    return float(p)


def main(tree_path, audit_path):
    header, nodes, links = load_tree(tree_path,1)
    assert header[2]==header[3]==0
    root = native_node(nodes,links,1)
    tree = Tree(root)
    positions = {key:int(i) for key,i in zip(tree.keys,tree.leaves,strict=True)}
    raw = np.fromfile(audit_path,'<f4')
    records = raw[:64*16].reshape(64,4,4)
    reasons = {'receiver_slots':0,'selected_slots':0,'sampled_cube_pdf_slots':0}
    errors = []
    for row in records:
        if row[3,3]==0:
            continue
        reasons['receiver_slots'] += 1
        observed = float(row[1,3])
        assert np.isfinite(observed)
        if observed<=0:
            continue
        reasons['selected_slots'] += 1
        reasons['sampled_cube_pdf_slots'] += int(row[2,3]>0)
        material = int(row[0,3].view('u4')) & 0x7fffffff
        leaf = positions[(*map(int,row[3,:3]),material)]
        expected = path_probability(tree,leaf,row[0,:3].astype(float),row[1,:3].astype(float))
        assert expected>0
        errors.append(abs(observed/expected-1))
    assert errors
    assert max(errors)<.005, errors
    result = dict(**reasons, leaf_pdf_max_relative_error=max(errors),
                  tree_ordered_sha256=ordered_signature(root))
    if reasons['sampled_cube_pdf_slots']:
        cube_errors = []
        for origin,normal,direction,low in records:
            if direction[3] <= 0:
                continue
            expected = cube_pdf(low[:3].astype(float),origin[:3].astype(float),
                                normal[:3].astype(float),direction[:3].astype(float))
            cube_errors.append(abs(float(direction[3])/expected-1.))
        assert len(cube_errors)==reasons['sampled_cube_pdf_slots']
        assert max(cube_errors)<.005, cube_errors
        result['cube_pdf'] = dict(samples=len(cube_errors),max_relative_error=max(cube_errors))
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tree',type=Path)
    parser.add_argument('audit',type=Path)
    main(**{key+'_path':path for key,path in vars(parser.parse_args()).items()})
