"""Check a live coarse build against the recorded native Prefix subtrees."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_exploration_coarse_upper import anchored_tree, cell_roots, coarse_tree
from light_sampling_clustering_repair import build_primitives
from light_sampling_moving_analysis import leaves, native_node, ordered_signature, read
from light_sampling_point_moments import sha
from light_sampling_weight_reference import load_tree


def equal_tree(left, right):
    pending = [(left, right)]
    while pending:
        a, b = pending.pop()
        assert a.material == b.material and len(a.children) == len(b.children)
        np.testing.assert_array_equal(a.data, b.data)
        pending.extend(zip(a.children, b.children, strict=True))


def envelopes(node):
    if not node.children:
        np.testing.assert_array_equal(node.data[4:7], node.data[:3] + 1.)
        return 1, 0.
    boxes = np.array([c.data for c in node.children])
    np.testing.assert_array_equal(node.data[:3], boxes[:,:3].min(axis=0))
    np.testing.assert_array_equal(node.data[4:7], boxes[:,4:7].max(axis=0))
    relative_error = abs(node.data[3]/boxes[:,3].sum()-1.)
    assert relative_error < 1e-5, relative_error
    descendants = [envelopes(c) for c in node.children]
    return 1+sum(r[0] for r in descendants), max(relative_error, *(r[1] for r in descendants))


def float32_split(centers, lo, hi, mass, ids, fuse=None):
    """Mirror GLSL's accumulation order while deriving buckets from integer bounds."""
    p = centers[ids]
    minimum, span = p.min(axis=0), np.ptp(p,axis=0)
    best = None
    for axis in range(3):
        if not span[axis]:
            continue
        bucket = np.minimum((p[:,axis]-minimum[axis])*12//span[axis],11)
        weights = np.zeros(12,np.float32)
        for i,k in zip(ids,bucket,strict=True):
            weights[k] = np.float32(weights[k]+np.float32(mass[i]))
        prefix = np.cumsum(weights,dtype=np.float32)
        suffix = np.cumsum(weights[::-1],dtype=np.float32)[::-1]
        for slot in range(11):
            left = bucket <= slot
            if not left.any() or left.all():
                continue
            ld = hi[ids[left]].max(axis=0)-lo[ids[left]].min(axis=0)
            rd = hi[ids[~left]].max(axis=0)-lo[ids[~left]].min(axis=0)
            dl,dr = np.float32(ld@ld),np.float32(rd@rd)
            l,r = prefix[slot],suffix[slot+1]
            if fuse=='left':
                cost = np.float32(float(l)*float(dl)+float(r*dr))
            elif fuse=='right':
                cost = np.float32(float(r)*float(dr)+float(l*dl))
            else:
                cost = np.float32(l*dl+r*dr)
            if best is None or cost < best[0]:
                best = cost,left
    if best is None:
        return ids[:len(ids)//2],ids[len(ids)//2:]
    return ids[best[1]],ids[~best[1]]


def first_difference(a, b, size, path=()):
    if ordered_signature(a)==ordered_signature(b):
        return None
    left = [sorted(cell_roots(c,size)) for c in a.children]
    right = [sorted(cell_roots(c,size)) for c in b.children]
    if left!=right:
        return dict(path=path, expected_cells_per_child=left, observed_cells_per_child=right)
    for i,(x,y) in enumerate(zip(a.children,b.children,strict=True)):
        diff = first_difference(x,y,size,(*path,i))
        if diff:
            return diff
    raise AssertionError('Different signature with identical child splits')


def main(prefix, candidate, output, rule):
    controls = {n:read(p/'accepted.json') for n,p in (('prefix',prefix),('live',candidate))}
    baseline, live = controls.values()
    for k in ('camera_sha256','leaf_sha256','world_voxel_sha256','grid'):
        assert baseline['identity'][k] == live['identity'][k], k
    assert live['identity']['grid']==16
    cell = (64,64,64)
    report = {'scope':'Exact leaf list and retained subtree bounds, mass, material and nonempty child order; physical links and empty slots may differ.',
              'rule':rule,'scene':live['scene'], 'sources':{n:p['source'] for n,p in controls.items()}, 'phases':{}}
    for phase in ('before','after'):
        paths = {n:p/f'{phase}_tree.bin' for n,p in (('prefix',prefix),('live',candidate))}
        def root(path):
            header, nodes, links = load_tree(path,6)
            assert header[2]==header[3]==0
            tree = native_node(nodes,links,1)
            rows = sorted(leaves(tree))
            assert len(rows)==len(set(rows))==int(header[1])
            return header, tree, rows
        _, prefix_tree, baseline_rows = root(paths['prefix'])
        header, candidate_tree, rows = root(paths['live'])
        assert rows == baseline_rows
        expected, count = anchored_tree(prefix_tree) if rule=='anchor' else coarse_tree(prefix_tree,cell)
        assert int(header[7])==count, (header[7],count,'live coarse path did not run')
        observed = cell_roots(candidate_tree,cell)
        control = cell_roots(prefix_tree,cell)
        assert observed.keys()==control.keys()
        for key in observed:
            equal_tree(observed[key],control[key])
        nodes, error = envelopes(candidate_tree)
        expected_sha, observed_sha = ordered_signature(expected), ordered_signature(candidate_tree)
        replays = {}
        for variant in ((None,'left','right') if rule=='sah' else ()):
            replay = build_primitives(np.asarray(list(control)),list(control.values()),True,
                      partition=lambda *args: float32_split(*args,fuse=variant))
            replays[variant or 'unfused'] = ordered_signature(replay)
        report['phases'][phase] = dict(emitters=len(rows), coarse_roots=count, active_nodes=nodes,
            internal_relative_mass_error_max=error, retained_subtrees_bit_exact=True,
            expected_ordered_topology_sha256=expected_sha, observed_ordered_topology_sha256=observed_sha,
            same_topology_as_offline=expected_sha==observed_sha,
            float32_topology_sha256=replays, matching_float32_replays=[m for m,h in replays.items() if h==observed_sha],
            first_reference_difference=first_difference(expected,candidate_tree,cell),
            tree_sha256={n:sha(p) for n,p in paths.items()})
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--rule',choices=('anchor','sah'),default='sah')
    a = parser.parse_args()
    main(a.prefix,a.candidate,a.output,a.rule)
