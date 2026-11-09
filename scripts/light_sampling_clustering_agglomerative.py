"""Compare exact bottom-up and binned top-down flux-diagonal clustering offline."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import brick_roots, cohorts
from light_sampling_clustering_repair import EVIDENCE, build_primitives, stats, widen_binary
from light_sampling_clustering_rule_screen import (
    CONTROLS, GATE, RULES, array_sha, score_window, sha, summarize, validate_root, validate_shuffle,
)
from light_sampling_moving_analysis import Node, leaves, ordered_signature

PAPER = 'https://www.graphics.cornell.edu/~bjw/IRT08Agglomerative.pdf'
BOTTOM = {scope: f'{scope}/bottom-up-flux-diagonal' for scope in ('global', 'upper')}
TOP = {scope: f'{scope}/flux-diagonal' for scope in BOTTOM}


def agglomerate(keys, primitives):
    """Exact mutual nearest neighbors with canonical ordering of equal-cost pairs."""
    assert len(keys) == len(primitives) > 0
    assert len({tuple(k) for k in keys}) == len(keys)
    order = np.lexsort(keys.T[::-1])
    binary = [primitives[i] for i in order]
    data = np.stack([node.data for node in binary])
    lower, upper, flux = data[:, :3], data[:, 4:7], data[:, 3]
    assert np.all(upper > lower) and np.all(np.isfinite(flux) & (flux > 0))
    active = np.ones(len(keys), dtype=bool)
    roots = np.arange(len(keys))
    chain = []

    # Each active slot is the rank of its minimum descendant world key. For
    # fixed A, scanning B in this order breaks ties by the symmetric pair key.
    while len(binary) < 2 * len(keys) - 1:
        live = np.flatnonzero(active)
        if not chain:
            chain.append(int(live[0]))
        a = chain[-1]
        others = live[live != a]
        size = np.maximum(upper[a], upper[others]) - np.minimum(lower[a], lower[others])
        costs = (flux[a] + flux[others]) * (size * size).sum(axis=1)
        b = int(others[costs.argmin()])
        if len(chain) < 2 or b != chain[-2]:
            assert b not in chain, 'a symmetric total edge order cannot cycle'
            chain.append(b)
            continue

        del chain[-2:]
        a, b = sorted((a, b))
        # Union bounds and summed positive flux make distance to any third
        # active cluster nondecreasing. Earlier chain links remain valid.
        lower[a] = np.minimum(lower[a], lower[b])
        upper[a] = np.maximum(upper[a], upper[b])
        flux[a] += flux[b]
        binary.append((int(roots[a]), int(roots[b])))
        roots[a] = len(binary) - 1
        active[b] = False
    return binary


def greedy_oracle(keys, primitives):
    """Independent scalar global pair minimization for small validation fixtures."""
    active = {tuple(map(int, key)): (node.data.tolist(), tuple(map(int, key)))
              for key, node in zip(keys, primitives, strict=True)}

    def measure(pair):
        a, b = pair
        x, y = active[a][0], active[b][0]
        low = [min(x[i], y[i]) for i in range(3)]
        high = [max(x[i], y[i]) for i in range(4, 7)]
        flux = x[3] + y[3]
        diagonal2 = sum((hi - lo)**2 for lo, hi in zip(low, high, strict=True))
        return (flux * diagonal2, a, b), [*low, flux, *high]

    tied_steps = 0
    while len(active) > 1:
        pairs = [measure(pair) for pair in combinations(sorted(active), 2)]
        (cost, a, b), data = min(pairs)
        tied_steps += sum(row[0][0] == cost for row in pairs) > 1
        active[a] = data, (active[a][1], active.pop(b)[1])
    return next(iter(active.values()))[1], tied_steps


def check_oracle(keys, primitives):
    rank = np.lexsort(keys.T[::-1])
    fixtures = {
        'first-world-keys': rank[:24],
        'fixed-shuffle': np.random.default_rng(501).permutation(len(keys))[:24],
    }
    result = {}
    for label, ids in fixtures.items():
        sample = [primitives[i] for i in ids]
        canonical = {id(node): tuple(map(int, keys[i])) for i, node in zip(ids, sample, strict=True)}
        binary = agglomerate(keys[ids], sample)

        def shape(index):
            node = binary[index]
            if isinstance(node, tuple):
                return tuple(shape(child) for child in node)
            return canonical[id(node)]

        expected, ties = greedy_oracle(keys[ids], sample)
        assert shape(len(binary) - 1) == expected
        result[label] = dict(
            primitives=len(ids), tied_global_minimum_steps=ties,
            ordered_binary_topology_matches=True,
            world_key_sha256=hashlib.sha256(keys[ids].astype('<i8').tobytes()).hexdigest(),
        )
    return result


def make_trees(root, keys, prior):
    rows = list(leaves(root))
    by_key = {row[:4]: row[4] for row in rows}
    assert len(rows) == len(keys) == len(by_key)
    brick_map = brick_roots(root)
    lamps = [Node(np.array([*key[:3], by_key[tuple(key)], *(key[:3] + 1)], float), int(key[3]), ())
             for key in keys]
    expected = Counter(rows)
    trees = {'prefix': validate_root(root, expected, brick_map)}
    topology = {'prefix': dict(**stats(trees['prefix']), ordered_sha256=ordered_signature(root))}
    oracles = {}
    inputs = {
        'global': (keys, lamps, None),
        'upper': (np.asarray(list(brick_map)), list(brick_map.values()), brick_map),
    }
    for scope, (primitive_keys, primitives, preserved) in inputs.items():
        control = 'global_sah' if scope == 'global' else 'upper_only'
        oracles[scope] = check_oracle(primitive_keys, primitives)
        for name, partition in [(control, None), (TOP[scope], RULES['flux-diagonal'])]:
            node = build_primitives(primitive_keys, primitives, True, partition=partition)
            tree = validate_root(node, expected, preserved)
            trees[name] = tree
            topology[name] = dict(**stats(tree), ordered_sha256=ordered_signature(node))

        binary = agglomerate(primitive_keys, primitives)
        node = widen_binary(binary, len(binary) - 1)
        tree = validate_root(node, expected, preserved)
        shuffled = np.random.default_rng(711).permutation(len(primitive_keys))
        binary_shuffled = agglomerate(primitive_keys[shuffled], [primitives[i] for i in shuffled])
        permuted = widen_binary(binary_shuffled, len(binary_shuffled) - 1)
        validate_shuffle(tree, validate_root(permuted, expected, preserved))
        name = BOTTOM[scope]
        trees[name] = tree
        topology[name] = dict(
            **stats(tree), ordered_sha256=ordered_signature(node), input_shuffle_bit_exact=True,
            upper_brick_objects_preserved=len(preserved) if preserved is not None else None,
        )

    for name in (*CONTROLS, *TOP.values()):
        assert topology[name]['ordered_sha256'] == prior[name]['ordered_sha256']
    return trees, topology, oracles


def main(output):
    index = EVIDENCE / '20261108-clustering-repair.json'
    recorded = json.loads(index.read_text(encoding='utf8'))['clustering_rule_screen']
    combo_index = EVIDENCE / '20261108-nearfar-combination.json'
    combo_identity = json.loads(combo_index.read_text(encoding='utf8'))['interior_m2']['cohort']
    refs = [EVIDENCE / f'20261108-{name}-reference.npz' for name in
            ('clustering-repair', 'cluster-summaries', 'nearfar-combination')]
    input_sha = {p.name: sha(p) for p in (*refs, combo_index)}
    assert input_sha == recorded['source_sha256']
    previous = {(r['cohort'], r['point'], r['window'], r['previous_label']): r
                for r in recorded['windows']}
    columns = recorded['method_columns']
    scripts = Path(__file__).parent
    script_names = ('clustering_repair', 'clustering_rule_screen', 'clustering_hierarchy',
                    'moving_analysis', 'point_screening', 'weight_reference')
    script_sha = {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                  for p in (Path(__file__), *(scripts / f'light_sampling_{n}.py' for n in script_names))}
    result = dict(
        scope='Offline exact mutual nearest neighbors over all active clusters. Fixed model 6 and unchanged binary-to-four collapse.',
        reference=PAPER,
        cost='(flux(A) + flux(B)) * squared_diagonal(AABB_union(A, B))',
        tie_break='Symmetric lexicographic pair of the clusters’ minimum descendant world keys. Children ordered by the same keys.',
        source_sha256=input_sha, scripts_sha256_lf=script_sha,
        gate=recorded['gate'], gate_points=sorted(GATE), controls=list(CONTROLS),
        top_down=list(TOP.values()), bottom_up=list(BOTTOM.values()),
        replay='python scripts/light_sampling_clustering_agglomerative.py '
               'builds/light-sampling/20261108-clustering-repair/agglomerative/result.json',
        cohorts={}, windows=[],
    )
    with np.load(refs[0]) as fresh, np.load(refs[1]) as old, np.load(refs[2]) as combo:
        for name, identity, keys, root, windows in cohorts(fresh, old, combo, combo_identity):
            prior = recorded['cohorts'][name]
            assert identity == prior['identity']
            print(f'{name}: {len(keys)} leaves, building and cross-checking', flush=True)
            trees, topology, oracles = make_trees(root, keys, prior['topology'])
            result['cohorts'][name] = dict(identity=identity, topology=topology, oracle=oracles)
            leaf_keys = list(map(tuple, keys))
            for point, window, label, origin, normal, moments in windows:
                saved = previous.pop((name, point, window, label))
                assert saved['moments_sha256'] == array_sha(moments)
                assert saved['mean_reference_sha256'] == array_sha(moments[:, :3])
                origin, normal = origin.astype(float), normal.astype(float)
                assert saved['receiver'] == [origin.tolist(), normal.tolist()]
                controls = {method: dict(zip(columns, saved['methods'][method], strict=True))
                            for method in (*CONTROLS, *TOP.values())}
                rows, _ = score_window(trees, leaf_keys, origin, normal, moments, controls)
                for method, expected in controls.items():
                    for metric in columns:
                        np.testing.assert_allclose(rows[method][metric], expected[metric], rtol=1e-10, atol=1e-14)
                for scope in BOTTOM:
                    variance = rows[BOTTOM[scope]]['variance']
                    control = rows[TOP[scope]]['variance']
                    rows[BOTTOM[scope]]['variance_over_top_down'] = variance / control if variance is not None else None
                result['windows'].append(dict(saved, methods=rows))
            print(f'{name}: {len(windows)} windows verified', flush=True)
    assert not previous and len(result['windows']) == 28 and len(result['cohorts']) == 5
    result['summaries'] = summarize(result['windows'], trees)
    result['passed_gate'] = [name for name in BOTTOM.values() if not result['summaries'][name]['gate_failures']]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
