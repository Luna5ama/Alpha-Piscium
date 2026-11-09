"""Screen deterministic global light clustering with the captured conditional moments."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from light_sampling_moving_analysis import Node, merged, ordered_signature
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import probabilities

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'experiments/light-sampling'
BUCKETS = 12


def area(lo, hi):
    x, y, z = (hi - lo).T
    return 2 * (x*y + y*z + z*x)


def split(positions, flux, ids, light_sah):
    p = positions[ids]
    lo, hi = p.min(axis=0), p.max(axis=0)
    diag = hi - lo + 1
    best = None
    for axis in range(3):
        extent = hi[axis] - lo[axis]
        if not extent:
            continue
        bucket = np.minimum((p[:, axis] - lo[axis]) * BUCKETS // extent, BUCKETS - 1)
        count = np.bincount(bucket, minlength=BUCKETS)
        flux_bins = np.bincount(bucket, weights=flux[ids], minlength=BUCKETS) if light_sah else count
        aabb_lo = np.full((BUCKETS, 3), np.inf)
        aabb_hi = np.full((BUCKETS, 3), -np.inf)
        np.minimum.at(aabb_lo, bucket, p)
        np.maximum.at(aabb_hi, bucket, p+1)
        nonempty = (np.cumsum(count)[:-1] != 0) & (np.cumsum(count[::-1])[::-1][1:] != 0)
        slots = np.flatnonzero(nonempty)
        l = slots
        r = slots + 1
        left_area = area(np.minimum.accumulate(aabb_lo)[l], np.maximum.accumulate(aabb_hi)[l])
        right_area = area(np.minimum.accumulate(aabb_lo[::-1])[::-1][r], np.maximum.accumulate(aabb_hi[::-1])[::-1][r])
        left = np.cumsum(flux_bins)[l]
        right = np.cumsum(flux_bins[::-1])[::-1][r]
        factor = diag.max() / diag[axis] if light_sah else 1
        cost = factor * (left * left_area + right * right_area)
        i = int(np.argmin(cost))
        value = (float(cost[i]), axis, int(slots[i]))
        if best is None or value < best[0]:
            best = value, bucket <= slots[i]
    # The ordered input makes this fallback deterministic even for coincident centers.
    if best is None:
        mid = len(ids)//2
        return ids[:mid], ids[mid:]
    mask = best[1]
    return ids[mask], ids[~mask]


def build(keys, flux, light_sah):
    order = np.lexsort(keys.T[::-1])
    keys, flux = keys[order], flux[order]
    assert len(keys) and np.all(np.isfinite(flux) & (flux > 0))
    assert len({tuple(k) for k in keys}) == len(keys)
    positions = keys[:, :3].astype(np.int64)
    leaves = [Node(np.array([*p, m, *(p + 1)], float), int(key[3]), ())
              for p, key, m in zip(positions, keys, flux)]
    # Build depth-first with an explicit worklist so unbalanced inputs cannot
    # consume the Python recursion limit. Children retain low/high split order.
    binary = [None]
    stack = [(0, np.arange(len(keys)))]
    while stack:
        i, indices = stack.pop()
        if len(indices) == 1:
            binary[i] = leaves[int(indices[0])]
            continue
        left, right = split(positions, flux, indices, light_sah)
        j = len(binary)
        binary.extend([None, None])
        binary[i] = (j, j+1)
        stack.extend([(j+1, right), (j, left)])
    wide = {0: None}
    order = [0]
    for i in order:
        pair = binary[i]
        if isinstance(pair, tuple):
            children = tuple(c for j in pair for c in (binary[j] if isinstance(binary[j], tuple) else (j,)))
            wide[i] = children
            order.extend(children)
        else:
            wide[i] = pair
    for i in reversed(order):
        if isinstance(wide[i], tuple):
            wide[i] = merged(wide[j] for j in wide[i])
    return wide[0]


def control(data, prefix):
    nodes, materials, children = (data[prefix+k] for k in ('nodes','materials','children'))
    out = [None] * len(nodes)
    for i in reversed(range(len(nodes))):
        child = tuple(out[j] for j in children[i] if j >= 0)
        out[i] = Node(nodes[i, :7], int(materials[i]) if materials[i] >= 0 else None, child)
    return Tree(out[0])


def scores(tree, origin, normal):
    n = tree.nodes
    half = (n[:, 4:7] - n[:, :3]) * .5
    delta = n[:, :3] + half - origin
    h = delta @ normal
    extent = half @ np.abs(normal)
    front = h + extent >= -1e-6 * ((np.abs(delta*normal)).sum(1) + extent + 1)
    amplitude = n[:, 3] * front * np.clip((h + np.linalg.norm(half, axis=1)) /
                     np.sqrt(np.maximum((delta*delta).sum(1), .25)), .001, 1)
    near2 = (np.maximum(np.abs(delta)-half, 0)**2).sum(1)
    far2 = ((np.abs(delta)+half)**2).sum(1)
    near = np.divide(amplitude, near2, out=np.zeros_like(amplitude), where=near2 > 0)
    far = np.divide(amplitude, far2, out=np.zeros_like(amplitude), where=far2 > 0)
    zero = np.where(near2 == 0, amplitude, 0)
    cond = np.zeros((len(n), 2))
    q = np.zeros(len(n))
    q[0] = 1
    null = 0.
    for i, children in enumerate(tree.children):
        if not len(children):
            continue
        f = far[children]
        if not f.sum():
            null += q[i]
            continue
        z = zero[children]
        a = z if z.sum() else near[children]
        cond[children, 0] = .5*a/a.sum()
        cond[children, 1] = .5*f/f.sum()
        q[children] = q[i] * cond[children].sum(1)
        # Sample deterministic sibling sets against the maintained formula.
        if i % 127 == 0:
            np.testing.assert_allclose(cond[children].sum(1), probabilities(n[children], 6, origin, normal), atol=1e-12)
    np.testing.assert_allclose(q[tree.leaves].sum() + null, 1, atol=2e-12)
    return q, null, cond


def key_order(tree, keys):
    positions = {k: i for k, i in zip(tree.keys, tree.leaves)}
    assert set(positions) == set(keys)
    return np.array([positions[k] for k in keys])


def score(tree, keys, origin, normal, moments):
    q, null, cond = scores(tree, origin, normal)
    order = key_order(tree, keys)
    p = q[order]
    m2 = moments[:, 3]
    mu = moments[:, :3].sum(0)
    assert np.isfinite(moments).all() and np.all(m2 >= 0)
    missing = (m2 > 0) & (p == 0)
    penalty = np.divide(m2, p, out=np.zeros_like(m2), where=p > 0)
    result = dict(variance=None if missing.any() else float(penalty.sum()-mu@mu),
                  missing_positive_support=int(missing.sum()), null_probability=null,
                  zero_probability_leaves=int((p==0).sum()), measured_positive_mass=float(p[m2>0].sum()))
    return result, p, penalty, q, cond, order


def trace(tree, score_result, m2, leaf):
    _, _, _, q, cond, order = score_result
    weights = np.zeros(len(tree.nodes))
    weights[order] = np.sqrt(m2)
    for i in reversed(range(len(weights))):
        if len(tree.children[i]):
            weights[i] = weights[tree.children[i]].sum()
    out = []
    node = int(order[leaf])
    while tree.parents[node] >= 0:
        parent = int(tree.parents[node])
        near, far = map(float, cond[node])
        ideal = float(weights[node]/weights[parent])
        out.append(dict(**tree.node_description(node), parent=tree.node_description(parent), path_probability=float(q[node]),
                        near_half=near, far_half=far, moment_optimal_conditional=ideal,
                        below_moment_optimal=near+far < ideal-1e-12))
        node = parent
    out.reverse()
    return dict(first_underallocated=next((x for x in out if x['below_moment_optimal']), None), path=out)


def stats(tree):
    n = tree.nodes
    internal = tree.materials < 0
    cross = ((n[:, :3]//16) != ((n[:, 4:7]-1)//16)).any(axis=1)
    return dict(leaves=len(tree.leaves), nodes=len(n), depth=int(tree.depth.max()),
                empty_slots=0, bytes_if_compacted_root_1=32*(len(n)+1),
                cross_brick_internal=int((internal & cross).sum()),
                area_weighted_mass=float((area(n[internal, :3], n[internal, 4:7])*n[internal, 3]).sum()))


def trees(data, group):
    prefix = control(data, group+'/world-prefix/')
    camera = control(data, group+'/camera-relative/')
    keys = list(map(tuple, data[group+'/keys']))
    order = key_order(prefix, keys)
    masses = prefix.nodes[order, 3]
    np.testing.assert_array_equal(masses, camera.nodes[key_order(camera,keys),3])
    variants = dict(camera=camera, prefix=prefix)
    build_stats = {}
    raw_keys = np.asarray(keys)
    shuffled = np.random.default_rng(711).permutation(len(keys))
    for name, flux_cost in [('light-sah',True),('geometry-sah',False)]:
        t = time.perf_counter()
        a = build(raw_keys, masses, flux_cost)
        b = build(raw_keys[shuffled], masses[shuffled], flux_cost)
        elapsed = time.perf_counter()-t
        sig = ordered_signature(a)
        assert sig == ordered_signature(b)
        variants[name] = Tree(a)
        np.testing.assert_array_equal(masses, variants[name].nodes[key_order(variants[name],keys),3])
        build_stats[name] = dict(**stats(variants[name]), topology_sha256=sig,
                                 permuted_input_same_topology=True, temporary_binary_nodes=2*len(keys)-1,
                                 two_cpu_builds_wall_seconds=elapsed)
    return keys, variants, build_stats


def main(output):
    ref = EVIDENCE/'20261108-cluster-summaries-reference.npz'
    new = EVIDENCE/'20261108-nearfar-combination-reference.npz'
    result = dict(scope='Conditional RGB M2 on captured receivers and inputs, with no brightness cutoff. Candidate uses no receiver or moments.',
                  source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ref,new)}, builds={}, rows=[])
    with np.load(ref) as data, np.load(new) as combo:
        meta = json.loads(str(data['metadata']))
        for group in ('ls1','sides','side-top'):
            keys, variants, build_stats = trees(data,group)
            identity = meta[group]['identity']
            result['builds'][group] = dict(identity=identity, **build_stats)
            windows = []
            for w in meta[group]['windows']:
                path = f'{group}/{w["point"]}/{w["window"]}'
                origin, normal = np.asarray(w['receiver_audit'])[:2,:3]
                moments = np.c_[data[path+'/mean_rgb'], data[path+'/m2']]
                windows.append((path, origin,normal,moments,None))
            if group != 'ls1':
                np.testing.assert_array_equal(np.asarray(keys),combo['keys'])
                np.testing.assert_array_equal(data[group+'/world-prefix/nodes'], combo['tree_nodes'][:,:7])
                np.testing.assert_array_equal(data[group+'/world-prefix/children'],combo['tree_children'])
                for i, label in enumerate(combo['moment_labels']):
                    if label.split('/')[0] != group:
                        continue
                    point = list(combo['point_ids']).index(label.split('/')[1])
                    windows.append((str(label), combo['origins'][point], combo['normals'][point],
                                    combo['conditional_moments'][i], combo['pmfs'][point,:,5]))
            for label,origin,normal,moments,prior in windows:
                origin,normal = origin.astype(float), normal.astype(float)
                evaluated = {name:score(tree,keys,origin,normal,moments) for name,tree in variants.items()}
                if prior is not None:
                    np.testing.assert_allclose(evaluated['prefix'][1],prior,atol=1e-10,rtol=3e-6)
                methods = {}
                for name,row in evaluated.items():
                    value = dict(row[0])
                    for baseline in ('camera','prefix'):
                        denominator = evaluated[baseline][0]['variance']
                        value['variance_over_'+baseline] = value['variance']/denominator if value['variance'] is not None and denominator else None
                    if name.endswith('sah'):
                        delta = row[2]-evaluated['prefix'][2]
                        positive = moments[:,3] > 0
                        assert positive.any()
                        leaf = int(np.argmax(np.where(positive, delta, -np.inf)))
                        if row[0]['missing_positive_support']:
                            leaf = int(np.flatnonzero((moments[:,3]>0)&(row[1]==0))[0])
                        value['worst_leaf'] = dict(key=list(map(int,keys[leaf])), m2=float(moments[leaf,3]),
                            candidate_pmf=float(row[1][leaf]), prefix_pmf=float(evaluated['prefix'][1][leaf]),
                            candidate=trace(variants[name], row, moments[:,3], leaf),
                            prefix=trace(variants['prefix'], evaluated['prefix'], moments[:,3],leaf))
                    methods[name] = value
                result['rows'].append(dict(point_window=label, group=group, origin=origin.tolist(), normal=normal.tolist(), methods=methods))
                print(label, *(round(methods[n]['variance_over_prefix'],4) if methods[n]['variance_over_prefix'] is not None else None for n in ('camera','light-sah','geometry-sah')), flush=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
