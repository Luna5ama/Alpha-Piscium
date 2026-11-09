"""Pack and replay the newly measured fixed receivers for the clustering screen."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_repair import build, control, key_order, score, trace
from light_sampling_moving_analysis import (
    captured_brick_roots, leaves, native_node, ordered_signature, rebuild_grouping,
)
from light_sampling_point_moments import read, sha
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import load_tree


def pack_reference(run, output):
    arrays = {}
    metadata = {}
    for scene in (1, 5):
        case = run / 'cases' / f'light-sampling-{scene}'
        (accepted,) = case.glob('m2-*/accepted.json')
        info = read(accepted)
        folder = accepted.parent
        header, nodes, links = load_tree(folder / 'tree.bin', 1)
        assert header[2] == 0 and header[1] == info['emitters']
        prefix = native_node(nodes, links, 1)
        brick = info['pose']['camera_brick']
        roots = captured_brick_roots(folder, nodes, links, 16, brick, 'world-prefix', prefix='before_')
        assert ordered_signature(prefix) == ordered_signature(rebuild_grouping(roots, 16, brick, 'world-prefix'))
        camera = rebuild_grouping(roots, 16, brick, 'camera-relative')
        records = sorted(leaves(prefix))
        np.testing.assert_array_equal(records, sorted(leaves(camera)))
        for name, root in [('prefix', prefix), ('camera', camera)]:
            tree = Tree(root)
            children = np.full((len(tree.nodes), 4), -1, dtype='<i4')
            for i, child in enumerate(tree.children):
                children[i, :len(child)] = child
            for key, value in [('nodes', tree.nodes.astype('<f4')), ('materials', tree.materials.astype('<i4')),
                               ('children', children)]:
                arrays[f'{scene}/{name}/{key}'] = value
        fixture = run / 'm2-source/shaders' / info['fixture']['texture']
        assert sha(fixture) == info['fixture']['texture_sha256']
        keys = np.fromfile(fixture, '<i4').reshape(-1, 4)[4:]
        np.testing.assert_array_equal(keys, np.array([x[:4] for x in records], dtype='<i4'))
        arrays[f'{scene}/keys'] = keys
        measurements = []
        for index, window in enumerate(info['windows']):
            path = Path(window['path'])
            with np.load(path) as m:
                arrays[f'{scene}/{index}/moments'] = np.c_[m['mean_rgb'], m['m2']]
                arrays[f'{scene}/{index}/complete_samples'] = m['complete_samples']
            measurements.append({key: window[key] for key in ('point','window','receiver_audit','missing_pixel_groups',
                                  'nonfinite_rgb_samples','second_moment_identity_max_relative_error')}
                                  | {'moment_source_sha256': sha(path)})
        metadata[str(scene)] = dict(source=info['path'], identity=info['identity'], windows=measurements)
    arrays['metadata'] = np.asarray(json.dumps(metadata, separators=(',', ':')))
    np.savez_compressed(output, **arrays)


def screen(reference):
    result = []
    with np.load(reference) as data:
        for scene, info in json.loads(str(data['metadata'])).items():
            raw_keys = data[f'{scene}/keys']
            keys = list(map(tuple, raw_keys))
            prefix = control(data, f'{scene}/prefix/')
            camera = control(data, f'{scene}/camera/')
            flux = prefix.nodes[key_order(prefix, keys), 3]
            trees = dict(prefix=prefix, camera=camera)
            for name, power, kr in [('light-sah', True, True), ('geometry-sah', False, False),
                                    ('flux-no-Kr', True, False), ('count-with-Kr', False, True)]:
                trees[name] = Tree(build(raw_keys, flux, power, aspect_correction=kr))
            for index, window in enumerate(info['windows']):
                origin, normal = np.asarray(window['receiver_audit'])[:2, :3]
                moments = data[f'{scene}/{index}/moments']
                evaluated = {n: score(tree, keys, origin, normal, moments) for n, tree in trees.items()}
                methods = {n: dict(row[0]) for n, row in evaluated.items()}
                m2 = moments[:, 3]
                if window['point'] == 'left-red-pillar':
                    for candidate in ('light-sah', 'geometry-sah'):
                        a, b = evaluated[candidate], evaluated['prefix']
                        delta = np.where(m2 > 0, a[2] - b[2], -np.inf)
                        leaf = int(np.argmax(delta))
                        methods[candidate]['dominant_excess_second_moment_leaf'] = dict(
                            key=list(map(int, keys[leaf])), m2=float(m2[leaf]), excess_second_moment=float(delta[leaf]),
                            candidate_pmf=float(a[1][leaf]), prefix_pmf=float(b[1][leaf]),
                            candidate_path=trace(trees[candidate], a, m2, leaf),
                            prefix_path=trace(prefix, b, m2, leaf))
                weights = prefix.weights(origin, normal)[0]
                q, null = prefix.pmf(weights[:, :1])
                p = q[key_order(prefix, keys), 0]
                mu = moments[:, :3].sum(0)
                unsupported = (p == 0) & (m2 > 0)
                methods['prefix-A'] = dict(
                    variance=None if unsupported.any() else float(np.divide(m2, p, out=np.zeros_like(m2), where=p > 0).sum() - mu @ mu),
                    missing_positive_support=int(unsupported.sum()), null_probability=float(null[0]))
                for row in methods.values():
                    row['variance_over_prefix'] = (row['variance'] / methods['prefix']['variance']
                                                  if row['variance'] is not None else None)
                result.append(dict(scene=int(scene), identity=info['identity'], source=info['source'],
                                   **window, methods=methods))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pack-from', type=Path)
    args = parser.parse_args()
    if args.pack_from:
        pack_reference(args.pack_from.resolve(), args.reference)
    rows = screen(args.reference)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf8')
    for row in rows:
        print(row['scene'], row['point'], row['window'],
              {name: round(x['variance_over_prefix'], 5) if x['variance_over_prefix'] is not None else None
               for name, x in row['methods'].items()})
