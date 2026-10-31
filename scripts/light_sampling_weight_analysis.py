"""Inspect current four-way light-tree captures without modifying their originals."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_geometry import cube_integral
from light_sampling_weight_reference import audit_capture, distribution, load_tree


def baseline_review(directory, cube_points):
    header, nodes, links = load_tree(directory / 'tree.bin', 0)
    children, depths, leaves = {}, {}, []
    pending = [(1, 0)]
    while pending:
        index, depth = pending.pop()
        depths[index] = depth
        if nodes[index, 3] == 0:
            continue
        link = int(links[index])
        if link & 0x80000000:
            leaves.append(index)
        else:
            base, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
            children[index] = list(range(base, base + count))
            pending.extend((child, depth + 1) for child in children[index])
    leaf_ids = np.array(leaves)
    records = np.fromfile(directory / 'audit.bin', '<f4')[:1024].reshape(64, 4, 4)
    rows = []
    for slot, record in enumerate(records):
        if record[3, 3] == 0:
            continue
        origin, normal = record[0, :3].astype(float), record[1, :3].astype(float)
        q_map, zero = distribution(nodes, links, origin, normal, 0)
        q = np.array([q_map.get(i, 0) for i in leaves])
        half = (nodes[leaf_ids, 4:7] - nodes[leaf_ids, :3]) * .5
        delta = nodes[leaf_ids, :3] + half - origin
        d2 = np.sum(delta * delta, axis=1)
        r2 = np.sum(half * half, axis=1)
        cosine = np.clip((delta @ normal + np.sqrt(r2)) / np.sqrt(np.maximum(d2, .25)), .001, 1)
        scores = nodes[leaf_ids, 3] * cosine / np.maximum.reduce([d2, r2, np.full_like(d2, .25)])
        p = scores / scores.sum()
        subtree = {i: float(s) for i, s in zip(leaves, scores)}
        proposal_subtree = dict(q_map)
        for i in sorted(children, key=depths.get, reverse=True):
            subtree[i] = sum(subtree.get(j, 0) for j in children[i])
            proposal_subtree[i] = sum(proposal_subtree.get(j, 0) for j in children[i])
        by_depth = {}
        branches = []
        for i, kids in children.items():
            ideal = np.array([subtree.get(j, 0) for j in kids]) / subtree[i]
            actual_mass = np.array([proposal_subtree.get(j, 0) for j in kids])
            actual = actual_mass / actual_mass.sum()
            positive = ideal > 0
            contribution = subtree[i] / subtree[1] * np.sum(ideal[positive] * np.log(ideal[positive] / actual[positive]))
            depth = depths[i]
            by_depth[depth] = by_depth.get(depth, 0) + float(contribution)
            branches.append(dict(node=i, depth=depth, kl=float(contribution), ideal=ideal.tolist(), actual=actual.tolist()))
        kl = float(np.sum(p * np.log(p / q)))
        assert np.isclose(sum(by_depth.values()), kl, atol=1e-9)
        behind = delta @ normal + np.sum(np.abs(normal) * half, axis=1) <= 0
        row = dict(slot=slot, origin=origin.tolist(), normal=normal.tolist(), zero_probability=zero,
                   below_horizon_probability=float(q[behind].sum()), total_variation=float(np.abs(p-q).sum()*.5),
                   kl=kl, kl_by_depth=by_depth, largest_branches=sorted(branches, key=lambda x: x['kl'], reverse=True)[:10])
        if len(rows) < cube_points:
            weights = np.array([cube_integral(nodes[i, :3], origin, normal) * nodes[i, 3] for i in leaves])
            if weights.sum() > 0:
                target = weights / weights.sum()
                row['unoccluded_cube_proxy'] = dict(tree_second_moment=float(np.sum(target**2 / q)),
                                                    flat_second_moment=float(np.sum(target**2 / p)))
        rows.append(row)
    return dict(audit=audit_capture(directory, 0), receiver_points=rows,
                cube_scope='Fixed mass, unoccluded projected cube integral; excludes texture and visibility.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--cube-points', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(baseline_review(args.directory, args.cube_points), indent=2) + '\n', encoding='utf-8')
