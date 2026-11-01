"""Audit synthetic outer-build captures against an independent full-key trie."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_moving_analysis import (
    Node, camera_from_audit, combine_world_children, leaf_key, native_node, ordered_signature,
)
from light_sampling_weight_reference import load_tree
from light_sampling_world_layout import WorldLayout, oracle


def audit_tree(path, grid, arity, model, camera_brick, fixture_id):
    fixture_id = next(name for name in ('old-seam', 'cross-zero', 'cycle') if fixture_id.startswith(name))
    header, nodes, links = load_tree(path, model)
    layout = WorldLayout(grid, arity, camera_brick)
    stride = 48 if model in (4, 5) else 32
    assert path.stat().st_size == 32 + (layout.capacity + 262144) * stride
    assert header[0] == header[2] == header[3] == 0
    count = int(header[1])
    if fixture_id == 'cycle':
        assert count in (0, 1, 2)
    else:
        assert count == (6 if fixture_id == 'old-seam' else 2)
    if fixture_id == 'old-seam':
        first = camera_brick[0] - grid // 2
        xs = [1, 2, 16, first, first + 2, first + grid - 1]
    else:
        xs = [-1, 0][:count]
    expected = {}
    for index, x in enumerate(xs):
        brick = (x, camera_brick[1], camera_brick[2])
        position = np.asarray(brick) * 16
        expected[brick] = Node(np.r_[position, index + 1, position + 1], 2, ())
    reference = oracle(expected, combine_world_children, arity)
    root = native_node(nodes, links, 1) if count else None
    assert ordered_signature(root) == ordered_signature(reference)
    visited, leaf_keys = set(), []

    def visit(index):
        assert 0 < index < layout.capacity and index not in visited
        visited.add(index)
        node, link = nodes[index], int(links[index])
        assert node[3] > 0 and np.isfinite(node[:7]).all()
        if link & 0x80000000:
            key = leaf_key(native_node(nodes, links, index))
            leaf_keys.append(key)
            mean, variance = node[:3] + .5, 5 / 12
        else:
            base, child_count = link & 0x0fffffff, ((link >> 28) & 7) + 1
            assert 2 <= child_count <= arity
            assert base > 0 and base + child_count <= layout.capacity
            children = [child for child in range(base, base + child_count) if nodes[child, 3] > 0]
            assert len(children) >= 2, 'Unary chains must be compressed'
            moments = [visit(child) for child in children]
            active = nodes[children]
            assert np.array_equal(node[:3], active[:, :3].min(axis=0))
            assert np.array_equal(node[4:7], active[:, 4:7].max(axis=0))
            assert np.isclose(node[3], active[:, 3].sum(), rtol=1e-6)
            mean = np.average([m[0] for m in moments], axis=0, weights=active[:, 3])
            variance = np.average([m[1] + np.sum((m[0] - mean) ** 2) for m in moments], weights=active[:, 3])
        if model in (4, 5):
            assert np.allclose(node[8:11], mean, rtol=0, atol=.003)
            assert np.isclose(node[11], variance, rtol=.002, atol=.003)
        return mean, variance

    if count:
        visit(1)
        assert len(leaf_keys) == len(set(leaf_keys)) == count
    else:
        assert np.count_nonzero(nodes[:layout.capacity]) == 0, 'Stale outer node after empty build'
    if count == 1:
        assert int(links[1]) & 0x80000000, 'Single emitter must be the root leaf'
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                emitters=count, reachable_nodes=len(visited), ordered_oracle_matches=True,
                outer_capacity=layout.capacity, stride=stride, moment_fields_verified=model in (4, 5),
                empty_outer_fully_zero=count == 0, singleton_root_leaf=count == 1)


def analyze(run):
    cases = json.loads((run / 'fixtures/cases.json').read_text(encoding='utf-8-sig'))
    definitions = json.loads((run / 'gpu-fixtures/fixtures.json').read_text())
    quality = json.loads((run / 'cases.json').read_text())
    output = []
    for case in cases:
        definition = next(row for row in definitions if case['id'].startswith(row['id']))
        job = json.loads((run / case['path']).read_text(encoding='utf-8-sig'))
        provenance = job['result']['provenance']
        assert job['success'] and not provenance['stale'] and not provenance['head_changed']
        assert provenance['resolved_revision'] == definition['revision']
        assert provenance['patch_sha256'] == definition['sha256']
        assert all(row['status'] == 'RECEIPT_STATUS_OK' for row in job['result']['action_receipts'])
        settings = {row['name']: row['value'] for row in provenance['effective_settings']['settings']}
        for setting, field in (('SETTING_VOXEL_GRID_SIZE', 'grid'), ('SETTING_DEBUG_LIGHT_ARITY', 'arity'),
                               ('SETTING_DEBUG_LIGHT_WEIGHT_MODEL', 'model')):
            assert int(settings[setting]) == int(case[field])
        reference = next(row for row in quality if row['method'] == 'world-a' and row['grid'] == 16
                         and row['preset_id'] == provenance['preset_id'])
        folder = (run / reference['path']).parent
        pose = camera_from_audit(folder)
        assert list(case['camera_brick']) == pose['camera_brick']
        camera = np.fromfile(folder / 'iris_ssbo_0.bin', '<u4')
        frame_ids = []
        for path in case['tree_paths']:
            tree_path = run / path
            camera_path = tree_path.with_name(tree_path.name.replace('_tree', '_camera'))
            if not camera_path.exists():
                assert case['id'].startswith('cycle')
                camera_path = tree_path.parent / 'camera.bin'
            capture_camera = np.fromfile(camera_path, '<u4')
            assert np.array_equal(np.r_[camera[16:32], camera[80:112]],
                                  np.r_[capture_camera[16:32], capture_camera[80:112]])
            frame_ids.append(json.loads(tree_path.with_suffix('.json').read_text())['frame_id'])
        trees = [audit_tree(run / path, int(case['grid']), int(case['arity']), int(case['model']),
                            case['camera_brick'], case['id']) for path in case['tree_paths']]
        if case['id'].startswith('cycle'):
            counts = [tree['emitters'] for tree in trees]
            assert len(trees) == 16 and set(counts) == {0, 1, 2}
            assert any(a == 2 and b == 0 for a, b in zip(counts, counts[1:])), counts
            assert all(a < b for a, b in zip(frame_ids, frame_ids[1:]))
        output.append(dict(**case, checks=trees, source_and_initial_camera_verified=True,
                           job_id=job.get('job_id'), patch_sha256=provenance['patch_sha256'], frame_ids=frame_ids))
    assert len(output) == 4
    (run / 'gpu-fixture-validation.json').write_text(json.dumps(dict(cases=output, passed=True), indent=2) + '\n')
    return dict(passed=True, cases=len(output), tree_dumps=sum(len(row['checks']) for row in output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run)))
