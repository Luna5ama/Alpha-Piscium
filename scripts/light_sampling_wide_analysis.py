"""Independent audit of wide light-tree captures."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def audit(directory, arity):
    words = np.fromfile(directory / 'tree.bin', '<u4')
    header = words[:8]
    assert header[1] and header[2] == header[3] == 0
    words = words[8:].reshape(-1, 8)
    nodes = words.view('<f4').astype(float)
    parents, leaves, seen, depths, counts = {}, {}, set(), [], []
    pending = [(1, 0)]
    while pending:
        i, depth = pending.pop()
        assert i not in seen and 0 < i < len(nodes)
        seen.add(i)
        node, link = nodes[i], int(words[i, 7])
        if node[3] == 0:
            continue
        assert np.isfinite(node[:7]).all() and node[3] > 0
        if link & 0x80000000:
            assert np.all(node[4:7] - node[:3] == 1)
            key = tuple(node[:3])
            assert key not in leaves
            leaves[key] = (i, link & 0x7fffffff, node[3])
            depths.append(depth)
            continue
        base, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
        assert 1 <= count <= arity and 0 < base and base + count <= len(nodes)
        counts.append(count)
        children = nodes[base:base + count]
        assert np.isclose(node[3], children[:, 3].sum(), rtol=2e-6)
        for j, child in enumerate(children, base):
            if child[3] > 0:
                assert np.all(node[:3] <= child[:3]) and np.all(node[4:7] >= child[4:7])
            assert j not in parents
            parents[j] = (i, base, count)
            pending.append((j, depth + 1))
    assert len(leaves) == header[1]
    assert np.isclose(sum(x[2] for x in leaves.values()), nodes[1, 3], rtol=2e-6)
    records = np.fromfile(directory / 'audit.bin', '<f4').reshape(64, 4, 4).astype(float)
    errors = []
    for record in records:
        if record[3, 3] == 0:
            continue
        origin, normal = record[0, :3], record[1, :3]
        index = leaves[tuple(record[3, :3])][0]
        pdf = 1.0
        while index != 1:
            parent, base, count = parents[index]
            child = nodes[base:base + count]
            half = (child[:, 4:7] - child[:, :3]) * .5
            delta = child[:, :3] + half - origin
            r2 = np.sum(half * half, axis=1)
            d2 = np.sum(delta * delta, axis=1)
            cosine = np.clip((delta @ normal + np.sqrt(r2)) / np.sqrt(np.maximum(d2, .25)), .001, 1)
            weight = child[:, 3] * cosine / np.maximum(np.maximum(d2, r2), .25)
            assert weight.sum() > 0 and np.isclose((weight / weight.sum()).sum(), 1)
            pdf *= weight[index - base] / weight.sum()
            index = parent
        errors.append(abs(record[1, 3] / pdf - 1))
    assert errors and max(errors) < .005, errors
    return dict(emitters=int(header[1]), allocated=int(header[0]), overflow=int(header[2]), exhausted=int(header[3]),
                leaf_set_sha256=hashlib.sha256(repr(sorted((p, m, w) for p, (_, m, w) in leaves.items())).encode()).hexdigest(),
                root_mass=nodes[1, 3], reachable_nodes=len(seen), max_depth=max(depths), mean_depth=float(np.mean(depths)),
                child_counts={str(n): counts.count(n) for n in range(1, arity + 1)}, samples=len(errors), max_pdf_relative_error=max(errors))


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1]), int(sys.argv[2])), indent=2))
