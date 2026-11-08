"""Bake frozen Ward summaries from native pilots and audit live per-pixel selections."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_cluster_summaries import KS, OUTER_CAPACITY, candidate_weights, fit, native_slots, verify_moments
from light_sampling_moving_analysis import native_node
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import load_tree

SCENES = range(1, 6)
ROWS_PER_SCENE = 11


def load_native(path):
    header, nodes, links = load_tree(path, 1)
    assert header[1] > 0 and header[2] == 0
    tree = Tree(native_node(nodes, links, 1))
    assert len(tree.leaves) == header[1]
    slots, addresses = native_slots(tree, path)
    cross = np.any(np.floor(tree.nodes[:, :3] / 16) != np.floor((tree.nodes[:, 4:7] - 1) / 16), axis=1)
    eligible = cross & (tree.materials < 0)
    eligible[0] = False
    assert np.all(addresses[eligible] < OUTER_CAPACITY)
    return tree, slots, addresses, eligible


def bake(pilots, output):
    atlas = np.zeros((ROWS_PER_SCENE * len(SCENES), OUTER_CAPACITY, 4), dtype='<f4')
    records = {}
    for scene in SCENES:
        name = f'light-sampling-{scene}'
        path = pilots / name / 'tree.bin'
        tree, _, addresses, eligible = load_native(path)
        scene_rows = atlas[(scene - 1) * ROWS_PER_SCENE:scene * ROWS_PER_SCENE]
        columns = addresses[eligible]
        assert len(columns) == len(set(columns.tolist()))
        scene_rows[0, columns] = tree.nodes[eligible, :4]
        scene_rows[1, columns, :3] = tree.nodes[eligible, 4:7]
        scene_rows[1, 0, 3] = len(tree.leaves)
        for k, first in zip(KS, (2, 3, 5)):
            values, _, _, _ = fit(tree, k, np.float32)
            verify_moments(tree, values, np.float32)
            scene_rows[first:first + k, columns] = values[eligible, :, 1:5].transpose(1, 0, 2)
            if k > 1:
                scene_rows[9 if k == 2 else 10, columns, :k] = values[eligible, :, 0]
        records[name] = dict(emitters=len(tree.leaves), cross_brick_candidates=len(columns),
                             pilot_tree_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    output.write_bytes(atlas.tobytes())
    return dict(scenes=records, atlas_sha256=hashlib.sha256(atlas.tobytes()).hexdigest())


def verify(pilot, live, atlas_path, scene, model, audit):
    tree, slots, addresses, eligible = load_native(pilot)
    actual, actual_slots, actual_addresses, actual_eligible = load_native(live)
    assert actual.keys == tree.keys
    np.testing.assert_array_equal(actual.materials, tree.materials)
    np.testing.assert_array_equal(actual_slots, slots)
    np.testing.assert_array_equal(actual_eligible, eligible)
    np.testing.assert_array_equal(actual_addresses[eligible], addresses[eligible])
    np.testing.assert_array_equal(actual.nodes[:, [0, 1, 2, 4, 5, 6]], tree.nodes[:, [0, 1, 2, 4, 5, 6]])
    np.testing.assert_allclose(actual.nodes[:, 3], tree.nodes[:, 3], rtol=2e-6, atol=0)
    for a, b in zip(actual.children, tree.children):
        np.testing.assert_array_equal(a, b)
    with atlas_path.open('rb') as stream:
        atlas = np.fromfile(stream, '<f4').reshape(-1, OUTER_CAPACITY, 4)
    scene_rows = atlas[(scene - 1) * ROWS_PER_SCENE:scene * ROWS_PER_SCENE]
    columns = actual_addresses[eligible]
    assert int(scene_rows[1, 0, 3]) == len(actual.leaves)
    np.testing.assert_array_equal(scene_rows[0, columns, :3], actual.nodes[eligible, :3])
    np.testing.assert_allclose(scene_rows[0, columns, 3], actual.nodes[eligible, 3], rtol=2e-6, atol=0)
    np.testing.assert_array_equal(scene_rows[1, columns, :3], actual.nodes[eligible, 4:7])
    if audit is None:
        return dict(emitters=len(actual.leaves), cross_brick_candidates=len(columns))

    fits = []
    for k, first in zip(KS, (2, 3, 5)):
        values = np.zeros((len(actual.nodes), k, 5), dtype=np.float32)
        values[eligible, :, 1:5] = scene_rows[first:first + k, columns].transpose(1, 0, 2)
        if k == 1:
            values[eligible, 0, 0] = actual.nodes[eligible, 3]
        else:
            values[eligible, :, 0] = scene_rows[9 if k == 2 else 10, columns, :k]
        fits.append(values)
    columns_by_model = {1: 0, 7: 1, 8: 2, 9: 3, 10: 4}
    record = np.fromfile(audit, '<f4')[:1024].reshape(64, 4, 4)
    lookup = {key: index for index, key in enumerate(actual.keys)}
    errors = []
    changed = 0
    selected = record[record[:, 3, 3] == 1]
    representatives = np.linspace(0, len(selected) - 1, min(12, len(selected)), dtype=int)
    for row in selected[representatives]:
        material = int(row[0, 3].view('<u4')) & 0x7fffffff
        light = (*map(int, row[3, :3]), material)
        observed_pdf = float(row[1, 3])
        assert light in lookup and observed_pdf > 0
        weights, _ = candidate_weights(actual, fits, row[0, :3], row[1, :3], eligible, np.float32)
        q, _ = actual.pmf(weights)
        expected = q[actual.leaves[lookup[light]], columns_by_model[model]]
        assert expected > 0
        errors.append(abs(observed_pdf / expected - 1))
        baseline = q[actual.leaves[lookup[light]], 0]
        changed += int(abs(expected - baseline) > .001 * max(expected, baseline))
    assert errors and max(errors) < .002, (len(errors), max(errors, default=None))
    return dict(emitters=len(actual.leaves), cross_brick_candidates=len(columns),
                audited_pdf_count=len(errors), max_relative_pdf_error=max(errors),
                audited_pdf_different_from_prefix_A=changed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='operation', required=True)
    make = subparsers.add_parser('bake')
    make.add_argument('--pilots', type=Path, required=True)
    make.add_argument('--atlas', type=Path, required=True)
    check = subparsers.add_parser('verify')
    for name in ('pilot', 'live', 'atlas'):
        check.add_argument('--' + name, type=Path, required=True)
    check.add_argument('--scene', type=int, choices=SCENES, required=True)
    check.add_argument('--model', type=int, choices=(1, 7, 8, 9, 10), default=1)
    check.add_argument('--audit', type=Path)
    args = parser.parse_args()
    result = bake(args.pilots, args.atlas) if args.operation == 'bake' else verify(
        args.pilot, args.live, args.atlas, args.scene, args.model, args.audit)
    print(json.dumps(result))
