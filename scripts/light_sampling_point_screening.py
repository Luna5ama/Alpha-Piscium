"""Six fixed proposal hypotheses scored against saved conditional RGB M2."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_moving_analysis import (
    captured_brick_roots, native_node, ordered_signature, rebuild_grouping, morton_decode,
)
from light_sampling_point_moments import ROOT, read, receiver_camera
from light_sampling_weight_analysis import leaf_identity
from light_sampling_weight_reference import load_tree, probabilities

MODES = ['A', 'nearfar-A', 'bounded-nearfar', 'cosine-B', 'outer-lookahead-1', 'outer-lookahead-2']


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class Tree:
    def __init__(self, root):
        queue = [root]
        self.children = []
        for node in queue:
            start = len(queue)
            queue.extend(node.children)
            self.children.append(np.arange(start, len(queue), dtype=int))
        self.nodes = np.stack([node.data for node in queue])
        self.materials = np.array([-1 if node.material is None else node.material for node in queue])
        self.leaves = np.flatnonzero(self.materials >= 0)
        self.keys = [(*map(int, self.nodes[i, :3]), int(self.materials[i])) for i in self.leaves]
        assert len(self.keys) == len(set(self.keys))
        self.parents = np.full(len(queue), -1, dtype=int)
        self.depth = np.zeros(len(queue), dtype=int)
        for i, children in enumerate(self.children):
            self.parents[children] = i
            self.depth[children] = self.depth[i] + 1

    def aggregate(self, leaf_values):
        output = np.zeros((len(self.nodes),) + leaf_values.shape[1:])
        output[self.leaves] = leaf_values
        for i in reversed(range(len(output))):
            if len(self.children[i]):
                output[i] = output[self.children[i]].sum(axis=0)
        return output

    def weights(self, origin, normal):
        nodes = self.nodes
        half = (nodes[:, 4:7] - nodes[:, :3]) * .5
        delta = nodes[:, :3] + half - origin
        radius2 = (half * half).sum(axis=1)
        distance2 = (delta * delta).sum(axis=1)
        height = delta @ normal
        extent = half @ np.abs(normal)
        front = height + extent >= -1e-6 * (np.abs(delta * normal).sum(axis=1) + extent + 1)
        cosine = np.clip((height + np.sqrt(radius2)) / np.sqrt(np.maximum(distance2, .25)), .001, 1)
        amplitude = nodes[:, 3] * cosine * front
        denominator = np.maximum.reduce([distance2, radius2, np.full(len(nodes), .25)])
        a = amplitude / denominator
        tight_height = np.where(height >= extent, height, np.maximum(height + extent, 0) ** 2 / (4 * np.maximum(extent, 1e-30)))
        b = nodes[:, 3] * np.clip(tight_height / np.sqrt(np.maximum(distance2, .25)), .001, 1) / denominator * front
        near2 = (np.maximum(np.abs(delta) - half, 0) ** 2).sum(axis=1)
        far2 = ((np.abs(delta) + half) ** 2).sum(axis=1)
        bounded = amplitude / np.maximum.reduce([near2, radius2, np.full(len(nodes), .25)])
        far = np.divide(amplitude, far2, out=np.zeros_like(amplitude), where=far2 > 0)
        weights = np.stack([a, a, a, b, a, a], axis=1)
        crosses = np.any(np.floor(nodes[:, :3] / 16) != np.floor((nodes[:, 4:7] - 1) / 16), axis=1)
        for i, children in enumerate(self.children):
            if not len(children):
                continue
            weights[children, 1] = probabilities(nodes[children], 6, origin, normal)
            if far[children].sum() > 0:
                weights[children, 2] = .5 * (bounded[children] / bounded[children].sum() + far[children] / far[children].sum())
            else:
                weights[children, 2] = 0
            if crosses[i]:
                weights[i, 4] = a[children].sum()
        for i, children in enumerate(self.children):
            if len(children) and crosses[i]:
                weights[i, 5] = weights[children, 4].sum()
        assert np.isfinite(weights).all() and (weights >= 0).all()
        return weights, crosses

    def pmf(self, weights):
        q = np.zeros_like(weights)
        q[0] = 1
        null = np.zeros(weights.shape[1])
        for i, children in enumerate(self.children):
            if not len(children):
                continue
            total = weights[children].sum(axis=0)
            q[children] = q[i] * np.divide(weights[children], total, out=np.zeros_like(weights[children]), where=total > 0)
            null += np.where(total == 0, q[i], 0)
        np.testing.assert_allclose(q[self.leaves].sum(axis=0) + null, 1, atol=2e-12)
        return q, null

    def node_description(self, index):
        node = self.nodes[index]
        return dict(compact_node=int(index), depth=int(self.depth[index]),
                    bounds=node[[0, 1, 2, 4, 5, 6]].tolist(), mass=float(node[3]))

    def splits(self, q, sqrt_m2, column, limit=4):
        sums = self.aggregate(sqrt_m2[:, None])[:, 0]
        result = []
        for i, children in enumerate(self.children):
            if not len(children) or q[i, column] == 0 or sums[i] == 0:
                continue
            p = q[children, column] / q[i, column]
            if np.any((sums[children] > 0) & (p == 0)):
                penalty = None
            else:
                penalty = float((np.divide(sums[children] ** 2, p, out=np.zeros(len(children)), where=p > 0).sum() - sums[i] ** 2) / q[i, column])
            result.append(dict(**self.node_description(i), penalty=penalty, path_probability=float(q[i, column]),
                               children=[dict(**self.node_description(int(j)), probability=float(q[j, column] / q[i, column]),
                                              fixed_conditional_optimal_share=float(sums[j] / sums[i])) for j in children]))
        return sorted(result, key=lambda row: float('inf') if row['penalty'] is None else row['penalty'], reverse=True)[:limit]

    def work(self, q, crosses):
        reads = np.zeros((len(self.nodes), 2), dtype=int)
        for i, children in enumerate(self.children):
            for child in children:
                if len(self.children[child]) and crosses[child]:
                    reads[i, 0] += len(self.children[child])
                    reads[i, 1] += len(self.children[child])
                    for grandchild in self.children[child]:
                        if len(self.children[grandchild]) and crosses[grandchild]:
                            reads[i, 1] += len(self.children[grandchild])
        return dict(scope='Static child-record reads implied by the offline definition, no GPU timing or reuse credit.',
                    eligible_candidates=int(sum(crosses[i] and len(self.children[i]) > 0 for i in range(1, len(self.nodes)))),
                    whole_tree_additional_reads=reads.sum(axis=0).tolist(),
                    expected_additional_reads_per_sample=[float(q[:, 4 + k] @ reads[:, k]) for k in range(2)])


def tree_variants(folder, fixture, layout, grid=16):
    header, nodes, links = load_tree(folder / 'tree.bin', 1)
    native = native_node(nodes, links, 1)
    camera = receiver_camera(folder, fixture)
    bricks = captured_brick_roots(folder, nodes, links, grid, camera['camera_brick'], layout, 4)
    replay = rebuild_grouping(bricks, grid, camera['camera_brick'], layout, 4)
    assert ordered_signature(native) == ordered_signature(replay)
    prefix = rebuild_grouping(bricks, grid, camera['camera_brick'], 'world-prefix', 4)
    relative = rebuild_grouping(bricks, grid, camera['camera_brick'], 'camera-relative', 4)
    result = {'world-prefix': Tree(prefix), 'camera-relative': Tree(relative)}
    if layout == 'camera-relative':
        result['legacy-native'] = Tree(native)
    else:
        assert ordered_signature(native) == ordered_signature(prefix)
    return result, camera


def score(tree, records, moments, origin, normal, output):
    by_key = {tuple(map(int, record)): i for i, record in enumerate(records)}
    assert set(by_key) == set(tree.keys)
    order = np.array([by_key[key] for key in tree.keys])
    mu = moments['mean_rgb'][order]
    m2 = moments['m2'][order]
    unavailable = np.flatnonzero(moments['complete_samples'][order] == 0)
    weights, crosses = tree.weights(origin, normal)
    q, null = tree.pmf(weights)
    leaf_q = q[tree.leaves]
    np.savez_compressed(output, keys=np.asarray(tree.keys, '<i4'), mean_rgb=mu, m2=m2, probabilities=leaf_q,
                        null_probability=null, modes=np.asarray(MODES), masses=tree.nodes[tree.leaves, 3])
    if len(unavailable):
        return dict(unavailable_lights=[tree.keys[i] for i in unavailable], methods=None, work=tree.work(q, crosses))
    assert np.isfinite(mu).all() and np.isfinite(m2).all() and (m2 >= 0).all()
    total_mu = mu.sum(axis=0)
    mu_squared = float(total_mu @ total_mu)
    optimal_second = float(np.sqrt(m2).sum() ** 2)
    methods = {}
    for column, mode in enumerate(MODES):
        p = leaf_q[:, column]
        failures = np.flatnonzero((m2 > 0) & (p == 0))
        terms = np.divide(m2, p, out=np.zeros_like(m2), where=p > 0)
        terms[failures] = np.inf
        prediction = None if len(failures) or not np.isfinite(terms).all() else float(terms.sum() - mu_squared)
        top = np.argsort(terms)[-8:][::-1]
        lights = []
        for i in top:
            path = []
            index = int(tree.leaves[i])
            while tree.parents[index] >= 0:
                parent = int(tree.parents[index])
                path.append(dict(**tree.node_description(parent), child=tree.node_description(index),
                                 probability=float(q[index, column] / q[parent, column]) if q[parent, column] > 0 else 0))
                index = parent
            lights.append(dict(key=tree.keys[i], mass=float(tree.nodes[tree.leaves[i], 3]), probability=float(p[i]),
                               mean_rgb=mu[i].tolist(), m2=float(m2[i]), penalty_term=None if not np.isfinite(terms[i]) else float(terms[i]), path=path[::-1]))
        methods[mode] = dict(variance=prediction, null_probability=float(null[column]),
                             support_failures=[tree.keys[i] for i in failures], top_penalty_lights=lights,
                             conditional_noise_at_this_q=None if prediction is None else float(np.divide(
                                 m2 - (mu * mu).sum(axis=1), p, out=np.zeros_like(m2), where=p > 0).sum()),
                             mean_light_selection_variance=None if prediction is None else float(np.divide(
                                 (mu * mu).sum(axis=1), p, out=np.zeros_like(m2), where=p > 0).sum() - mu_squared),
                             tree_selection_excess=None if prediction is None else float(prediction + mu_squared - optimal_second),
                             worst_splits=tree.splits(q, np.sqrt(m2), column))
    # Highest disjoint clusters entirely below/equal the world y=0 plane.
    low_clusters = []
    aggregate_mu = tree.aggregate(mu)
    aggregate_m2 = tree.aggregate(m2[:, None])[:, 0]
    aggregate_sqrt = tree.aggregate(np.sqrt(m2)[:, None])[:, 0]
    for i in range(1, len(tree.nodes)):
        parent = tree.parents[i]
        if tree.nodes[i, 5] <= 0 and tree.nodes[parent, 5] > 0:
            low_clusters.append(dict(**tree.node_description(i), probability=dict(zip(MODES, q[i].tolist())),
                                     mean_rgb=aggregate_mu[i].tolist(), m2_sum=float(aggregate_m2[i]),
                                     m2_share=float(aggregate_m2[i] / m2.sum()) if m2.sum() > 0 else None,
                                     mean_rgb_sum_share=float(aggregate_mu[i].sum() / total_mu.sum()) if total_mu.sum() > 0 else None,
                                     sqrt_m2_share=float(aggregate_sqrt[i] / aggregate_sqrt[0]) if aggregate_sqrt[0] > 0 else None))
    return dict(methods=methods, mean_rgb_total=total_mu.tolist(), fixed_conditional_optimal_variance=optimal_second - mu_squared,
                optimal_leaf_diagnostic='q proportional to sqrt(M2) for these finite conditional samples; not a cheap production proposal.',
                low_y_clusters=low_clusters, work=tree.work(q, crosses))


def analyze(run):
    datasets = read(run / 'point-cases.json')
    results = []
    output = run / 'screening'
    output.mkdir(exist_ok=True)
    for dataset in datasets:
        folder, fixture = resolve(dataset['folder']), read(resolve(dataset['fixture']))
        records = np.fromfile(resolve(fixture['texture']), '<i4').reshape(-1, 4)[4:]
        assert len(records) == fixture['light_count']
        trees, camera = tree_variants(folder, fixture, dataset['layout'])
        native = trees['world-prefix']
        by_key = dict(zip(native.keys, native.nodes[native.leaves, 3]))
        masses = np.asarray([by_key[tuple(map(int, r))] for r in records], '<f4')
        if 'leaf_identity_sha256' in fixture:
            assert hashlib.sha256(records.tobytes() + masses.tobytes()).hexdigest() == fixture['leaf_identity_sha256']
        else:
            identity = leaf_identity(folder / 'tree.bin', 1)
            assert hashlib.sha256(repr(identity).encode()).hexdigest() == fixture['leaf_mass_sha256']
        for receiver in fixture['receivers']:
            origin = np.asarray(receiver.get('origin_actual', receiver.get('origin')), float)
            normal = np.asarray(receiver['normal'], float)
            for window in (1, 2):
                name = f"{receiver['id']}-window{window}"
                moments = np.load(resolve(dataset['moments']) / f'{name}.npz')
                row = dict(dataset=dataset['id'], point=receiver['id'], window=window, camera=camera, trees={})
                for variant, tree in trees.items():
                    row['trees'][variant] = score(tree, records, moments, origin, normal, output / f"{dataset['id']}-{name}-{variant}.npz")
                def base(variant):
                    methods = row['trees'][variant]['methods']
                    return methods['A']['variance'] if methods is not None else None
                a, old_a = base('world-prefix'), base('camera-relative')
                for result in row['trees'].values():
                    if result['methods'] is None:
                        continue
                    for method in result['methods'].values():
                        value = method['variance']
                        method['relative_world_prefix_A'] = value / a if value is not None and a not in (None, 0) else None
                        method['relative_camera_A'] = value / old_a if value is not None and old_a not in (None, 0) else None
                results.append(row)
    (run / 'screening.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return dict(points=sorted({row['point'] for row in results}), conditional_windows=len(results), output=str(run / 'screening.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run)))
