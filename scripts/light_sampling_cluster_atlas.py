"""Bake frozen Ward summaries from native pilots and audit live per-pixel selections."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_cluster_summaries import KS, OUTER_CAPACITY, candidate_weights, fit, geometry, native_slots, verify_moments
from light_sampling_moving_analysis import native_node
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import distribution, load_tree

SCENES = range(1, 7)
ROWS_PER_SCENE = 23
MODELS = (1, 7, 8, 9, 10, 6, 11, 12, 13, 14)


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


def bake(pilots, output, previous=None, moving=None):
    atlas = np.zeros((ROWS_PER_SCENE * len(SCENES), OUTER_CAPACITY, 4), dtype='<f4')
    old = np.fromfile(previous, '<f4').reshape(5, 11, OUTER_CAPACITY, 4) if previous else None
    old_moving = np.fromfile(moving, '<f4').reshape(5, 11, OUTER_CAPACITY, 4)[0] if moving else None
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
            values, _, counts, stats = fit(tree, k, np.float32, include_bounds=True)
            verify_moments(tree, values, np.float32)
            bounds = stats['bounds']
            active = np.arange(k) < counts[:, None]
            np.testing.assert_array_equal(np.where(active[..., None], bounds[:, :, 0], np.inf).min(axis=1),
                                          tree.nodes[:, :3])
            np.testing.assert_array_equal(np.where(active[..., None], bounds[:, :, 1], -np.inf).max(axis=1),
                                          tree.nodes[:, 4:7])
            scene_rows[first:first + k, columns] = values[eligible, :, 1:5].transpose(1, 0, 2)
            if k > 1:
                scene_rows[9 if k == 2 else 10, columns, :k] = values[eligible, :, 0]
                start = 11 + 2 * (first - 3)
                scene_rows[start:start + 2 * k, columns, :3] = (
                    bounds[eligible].reshape(len(columns), 2 * k, 3).transpose(1, 0, 2))
        previous_rows = old[scene - 1] if old is not None and scene < 6 else old_moving if scene == 6 else None
        if previous_rows is not None:
            np.testing.assert_array_equal(scene_rows[:11], previous_rows, err_msg=name)
        records[name] = dict(emitters=len(tree.leaves), cross_brick_candidates=len(columns),
                             pilot_tree_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    output.write_bytes(atlas.tobytes())
    return dict(scenes=records, atlas_sha256=hashlib.sha256(atlas.tobytes()).hexdigest())


def decode_fits(tree, rows, addresses, eligible):
    fits = []
    bounds = []
    columns = addresses[eligible]
    for k, first in zip(KS, (2, 3, 5)):
        values = np.zeros((len(tree.nodes), k, 5), dtype=np.float32)
        values[eligible, :, 1:5] = rows[first:first + k, columns].transpose(1, 0, 2)
        if k == 1:
            values[eligible, 0, 0] = tree.nodes[eligible, 3]
            support = tree.nodes[:, [0, 1, 2, 4, 5, 6]].reshape(-1, 1, 2, 3)
        else:
            values[eligible, :, 0] = rows[9 if k == 2 else 10, columns, :k]
            start = 11 + 2 * (first - 3)
            support = np.zeros((len(tree.nodes), k, 2, 3), dtype=np.float32)
            support[eligible] = rows[start:start + 2 * k, columns, :3].transpose(1, 0, 2).reshape(-1, k, 2, 3)
        fits.append(values)
        bounds.append(support)
    return fits, bounds


def distance_scores(amplitude, bounds, origin):
    lo, hi = bounds[..., 0, :], bounds[..., 1, :]
    half = (hi - lo) * .5
    delta = np.abs(lo + half - origin)
    near2 = (np.maximum(delta - half, 0) ** 2).sum(axis=-1)
    far2 = ((delta + half) ** 2).sum(axis=-1)
    near = np.divide(amplitude, near2, out=np.zeros_like(amplitude), where=near2 > 0)
    far = np.divide(amplitude, far2, out=np.zeros_like(amplitude), where=amplitude > 0)
    return np.stack((near, far, np.where(near2 == 0, amplitude, 0)), axis=-1)


def scores_for_model(tree, fits, bounds, eligible, origin, normal, model):
    """Unnormalized N/F/Z per candidate, with the original angular model for each representative."""
    origin, normal, front = geometry(tree, origin, normal, np.float64)
    boxes = tree.nodes[:, [0, 1, 2, 4, 5, 6]].reshape(-1, 2, 3)
    half = (boxes[:, 1] - boxes[:, 0]) * .5
    delta = boxes[:, 0] + half - origin
    radius = np.linalg.norm(half, axis=1)
    cos = np.clip((delta @ normal + radius) / np.sqrt(np.maximum((delta ** 2).sum(axis=1), .25)), .001, 1)
    native = distance_scores(tree.nodes[:, 3] * cos * front, boxes, origin)
    scores = native.copy()
    if model == 11:
        for i in np.flatnonzero(eligible):
            scores[i] = native[tree.children[i]].sum(axis=0)
    elif model >= 12:
        scores[eligible] = 0
        use = eligible & front
        components = fits[model - 12][use].astype(np.float64)
        delta = components[..., 1:4] - origin
        cosine = np.clip((delta @ normal + np.sqrt(2 * components[..., 4])) /
                         np.sqrt(np.maximum((delta ** 2).sum(axis=-1), .25)), .001, 1)
        mass = tree.nodes[use, 3:4] if model == 12 else components[..., 0]
        scores[use] = distance_scores(mass * cosine, bounds[model - 12][use], origin).sum(axis=1)
    return scores


def mix(scores):
    near, far, zero = scores.T
    far_total = far.sum()
    if far_total == 0:
        return np.zeros_like(far)
    first = zero if zero.sum() > 0 else near
    return .5 * (first / first.sum() + far / far_total)


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

    fits, bounds = decode_fits(actual, scene_rows, actual_addresses, eligible)
    _, native_nodes, native_links = load_tree(live, 1)
    record = np.fromfile(audit, '<f4')[:1024].reshape(64, 4, 4)
    lookup = {key: index for index, key in enumerate(actual.keys)}
    errors = []
    changed = 0
    selected = record[record[:, 3, 3] == 1]
    for row in selected:
        material = int(row[0, 3].view('<u4')) & 0x7fffffff
        light = (*map(int, row[3, :3]), material)
        observed_pdf = float(row[1, 3])
        assert light in lookup and observed_pdf > 0
        origin, normal = row[0, :3], row[1, :3]
        weights, _ = candidate_weights(actual, fits, origin, normal, eligible, np.float32)
        node = leaf = actual.leaves[lookup[light]]
        scores = scores_for_model(actual, fits, bounds, eligible, origin, normal, model) if model in MODELS[5:] else None
        expected = baseline = 1.0
        while node:
            parent = actual.parents[node]
            siblings = actual.children[parent]
            offset = node - siblings[0]
            off = weights[siblings, 0]
            baseline *= off[offset] / off.sum()
            if scores is not None:
                sibling_weights = mix(scores[siblings])
            else:
                sibling_weights = weights[siblings, MODELS.index(model)]
            expected *= sibling_weights[offset] / sibling_weights.sum()
            node = parent
        if model == 6:
            native_q, _ = distribution(native_nodes, native_links, origin, normal, 6)
            np.testing.assert_allclose(expected, native_q[actual_addresses[leaf]], rtol=1e-6, atol=0)
        assert expected > 0
        errors.append(abs(observed_pdf / expected - 1))
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
    make.add_argument('--previous', type=Path)
    make.add_argument('--moving', type=Path)
    check = subparsers.add_parser('verify')
    for name in ('pilot', 'live', 'atlas'):
        check.add_argument('--' + name, type=Path, required=True)
    check.add_argument('--scene', type=int, choices=SCENES, required=True)
    check.add_argument('--model', type=int, choices=MODELS, default=1)
    check.add_argument('--audit', type=Path)
    args = parser.parse_args()
    result = bake(args.pilots, args.atlas, args.previous, args.moving) if args.operation == 'bake' else verify(
        args.pilot, args.live, args.atlas, args.scene, args.model, args.audit)
    print(json.dumps(result))
