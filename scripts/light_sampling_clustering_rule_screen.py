"""Replay six fixed spatial split rules against captured conditional RGB moments."""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import brick_roots, cohorts
from light_sampling_clustering_repair import EVIDENCE, build, build_primitives, score, split, stats
from light_sampling_moving_analysis import leaves, ordered_signature
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import probabilities

GATE = {
    'left-red-pillar', 'yellow-candidate-ceiling', 'side-a', 'side-b', 'top',
    'reviewed-vertical-face', 'reviewed-vertical-face-rng2',
}
PATH_POINTS = GATE - {'reviewed-vertical-face', 'reviewed-vertical-face-rng2'}
CONTROLS = ('prefix', 'global_sah', 'upper_only')


def diagonal_squared(lower, upper):
    size = upper - lower
    return (size * size).sum(axis=-1)


def center_split(centers, lower, upper, flux, ids, rule):
    local = centers[ids]
    minimum = local.min(axis=0)
    maximum = local.max(axis=0)
    span = maximum - minimum
    axis = int(span.argmax())
    if not span[axis]:
        middle = len(ids) // 2
        return ids[:middle], ids[middle:]

    coordinate = local[:, axis]
    if rule == 'spatial-midpoint':
        left = 2 * coordinate <= minimum[axis] + maximum[axis]
        return ids[left], ids[~left]

    # The builder assigns canonical world-key IDs for center ties.
    order = ids[np.lexsort((ids, coordinate))]
    middle = len(ids) // 2
    if rule == 'flux-median':
        cumulative = np.cumsum(flux[order])
        middle = int(np.argmin(np.abs(2 * cumulative[:-1] - cumulative[-1]))) + 1
    # Preserve the builder's canonical order for descendants and coincident centers.
    return np.sort(order[:middle]), np.sort(order[middle:])


RULES = {
    'flux-diagonal-Kr': partial(split, light_sah=True, aspect_correction=True, size=diagonal_squared),
    'flux-diagonal': partial(split, light_sah=True, aspect_correction=False, size=diagonal_squared),
    'count-diagonal': partial(split, light_sah=False, aspect_correction=False, size=diagonal_squared),
    **{name: partial(center_split, rule=name)
       for name in ('spatial-midpoint', 'count-median', 'flux-median')},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def validate_root(root, expected, original_bricks=None):
    assert Counter(leaves(root)) == expected
    if original_bricks is not None:
        rebuilt = brick_roots(root)
        assert rebuilt.keys() == original_bricks.keys()
        assert all(rebuilt[key] is node for key, node in original_bricks.items())
    return Tree(root)


def validate_shuffle(tree, other):
    np.testing.assert_array_equal(tree.nodes.view('u8'), other.nodes.view('u8'))
    np.testing.assert_array_equal(tree.materials, other.materials)
    np.testing.assert_array_equal(tree.parents, other.parents)


def build_cohort(root, keys, previous):
    source_leaves = {row[:4]: row[4] for row in leaves(root)}
    flux = np.asarray([source_leaves[tuple(key)] for key in keys])
    expected = Counter((*key, mass) for key, mass in source_leaves.items())
    bricks = brick_roots(root)
    brick_keys = np.asarray(list(bricks))
    brick_nodes = list(bricks.values())

    trees, topology = {}, {}
    for scope, count in [('global', len(keys)), ('upper', len(bricks))]:
        shuffle = np.random.default_rng(711).permutation(count)
        partitions = {
            ('global_sah' if scope == 'global' else 'upper_only'): None,
            **{f'{scope}/{name}': rule for name, rule in RULES.items()},
        }

        def construct(order, partition):
            if scope == 'global':
                return build(keys[order], flux[order], True, partition=partition)
            return build_primitives(brick_keys[order], [brick_nodes[i] for i in order], True,
                                    partition=partition)

        for name, partition in partitions.items():
            candidate = construct(np.arange(count), partition)
            interface = bricks if scope == 'upper' else None
            tree = validate_root(candidate, expected, interface)
            validate_shuffle(tree, validate_root(construct(shuffle, partition), expected, interface))
            trees[name] = tree
            topology[name] = dict(**stats(tree), ordered_sha256=ordered_signature(candidate),
                                  input_shuffle_bit_exact=True,
                                  upper_brick_objects_preserved=len(bricks) if interface else None)
    trees['prefix'] = validate_root(root, expected, bricks)
    topology['prefix'] = dict(**stats(trees['prefix']), ordered_sha256=ordered_signature(root))
    for name in CONTROLS:
        assert topology[name]['ordered_sha256'] == previous[name]['ordered_sha256']
    return trees, topology


def score_window(trees, keys, origin, normal, moments, previous):
    evaluations = {name: score(tree, keys, origin, normal, moments) for name, tree in trees.items()}
    baseline = evaluations['prefix'][0]['variance']
    assert baseline > 0
    rows = {}
    for name, (result, leaf_pmf, _, reach, _, _) in evaluations.items():
        residual = abs(math.fsum(leaf_pmf) + result['null_probability'] - 1)
        assert residual <= 2e-12
        count = np.asarray([len(children) for children in trees[name].children])
        variance = result['variance']
        rows[name] = dict(
            **result, pmf_conservation_absolute_error=residual,
            variance_over_prefix=variance / baseline if variance is not None else None,
            expected_internal_levels=float(reach @ (count > 0)),
            expected_nonempty_child_records=float(reach @ count),
        )
    for name in CONTROLS:
        for metric in ('variance', 'null_probability', 'measured_positive_mass', 'missing_positive_support'):
            np.testing.assert_allclose(rows[name][metric], previous[name][metric], rtol=1e-10, atol=1e-14)
    return rows, evaluations


def window_id(row):
    return {key: row[key] for key in ('cohort', 'point', 'window', 'previous_label')}


def summarize(windows, names):
    summaries = {}
    for name in names:
        gate = [row for row in windows if row['point'] in GATE]
        def ratio(row):
            value = row['methods'][name]['variance_over_prefix']
            return math.inf if value is None else value

        def worst(rows):
            row = max(rows, key=ratio)
            return dict(**window_id(row), variance_over_prefix=row['methods'][name]['variance_over_prefix'])

        summaries[name] = dict(
            worst=worst(windows), gate_worst=worst(gate),
            gate_failures=[dict(**window_id(row), variance_over_prefix=row['methods'][name]['variance_over_prefix'])
                           for row in gate if ratio(row) > 1],
            windows_with_missing_positive_support=sum(
                row['methods'][name]['missing_positive_support'] > 0 for row in windows),
            max_pmf_conservation_absolute_error=max(
                row['methods'][name]['pmf_conservation_absolute_error'] for row in windows),
            max_null_probability=max(row['methods'][name]['null_probability'] for row in windows),
            mean_expected_internal_levels=float(np.mean([row['methods'][name]['expected_internal_levels']
                                                        for row in windows])),
            mean_expected_nonempty_child_records=float(np.mean([
                row['methods'][name]['expected_nonempty_child_records'] for row in windows])),
        )
    return summaries


def path(tree, evaluation, leaf, origin, normal):
    _, p, _, reach, cond, order = evaluation
    current = int(order[leaf])
    steps = []
    while tree.parents[current] >= 0:
        parent = int(tree.parents[current])
        siblings = tree.children[parent]
        reference = probabilities(tree.nodes[siblings], 6, origin, normal)
        np.testing.assert_allclose(cond[siblings].sum(axis=1), reference, rtol=2e-13, atol=1e-15)
        probability = float(reference[siblings == current].item())
        steps.append(dict(
            selected=tree.node_description(current), siblings=[tree.node_description(int(i)) for i in siblings],
            near_half=float(cond[current, 0]), far_half=float(cond[current, 1]),
            reached=float(reach[current]), reference_conditional=probability,
        ))
        current = parent
    steps.reverse()
    independent_p = math.prod(step['reference_conditional'] for step in steps)
    np.testing.assert_allclose(p[leaf], independent_p, rtol=2e-13, atol=0)
    return dict(leaf_pmf=float(p[leaf]), independent_product=independent_p, steps=steps)


def diagnose(trees, origin, normal, moments, keys, evaluations, selected):
    base = evaluations['prefix']
    out = {}
    for name in selected:
        candidate = evaluations[name]
        supported = (moments[:, 3] > 0) & (base[1] > 0) & (candidate[1] > 0)
        delta = candidate[2] - base[2]
        trace_leaves = {}
        missing = np.flatnonzero((moments[:, 3] > 0) & (candidate[1] == 0))
        if len(missing):
            trace_leaves['lost_support'] = int(missing[np.argmax(moments[missing, 3])])
        for label, mask, direction in [('largest_increase', supported & (delta > 0), 1),
                                       ('largest_reduction', supported & (delta < 0), -1)]:
            if mask.any():
                trace_leaves[label] = int(np.argmax(np.where(mask, direction * delta, -np.inf)))
        traces = {}
        for label, leaf in trace_leaves.items():
            m2 = float(moments[leaf, 3])
            records = {n: path(trees[n], evaluations[n], leaf, origin, normal) for n in ('prefix', name)}
            direct = {n: m2 / r['independent_product'] if r['independent_product'] > 0 else None
                      for n, r in records.items()}
            direct_delta = None
            if all(v is not None for v in direct.values()):
                direct_delta = direct[name] - direct['prefix']
                np.testing.assert_allclose(direct_delta, delta[leaf], rtol=2e-12,
                                           atol=3e-15 * max(direct.values()))
            traces[label] = dict(world_key=list(map(int, keys[leaf])), m2=m2, paths=records,
                                 independent_m2_over_p=direct, independent_delta=direct_delta)
        out[name] = dict(
            supported_increases=float(delta[supported & (delta > 0)].sum()),
            supported_reductions=float(delta[supported & (delta < 0)].sum()), lights=traces,
        )
    return out


def main(output):
    index = EVIDENCE / '20261108-clustering-repair.json'
    recorded = json.loads(index.read_text(encoding='utf8'))['hierarchy_isolation']
    combo_index = EVIDENCE / '20261108-nearfar-combination.json'
    combo_identity = json.loads(combo_index.read_text(encoding='utf8'))['interior_m2']['cohort']
    refs = [EVIDENCE / f'20261108-{name}-reference.npz' for name in
            ('clustering-repair', 'cluster-summaries', 'nearfar-combination')]
    inputs = {p.name: sha(p) for p in refs}
    assert inputs == recorded['source_sha256']
    previous = {(r['cohort'], r['point'], r['window'], r['previous_label']): r['methods']
                for r in recorded['windows']}
    scripts = Path(__file__).parent
    result = dict(
        scope='Offline conditional RGB variance. Original Half-space A + Near/Far (model 6), same leaves and binary-to-four collapse.',
        controls=list(CONTROLS), rules=list(RULES),
        rule_details='12 buckets and X/Y/Z then lowest bucket tie-break for diagonal costs. '
                     'Center rules use longest doubled integer center span, X/Y/Z ties, low-inclusive midpoint, '
                     'floor count median or first minimum flux imbalance, with canonical world keys for ties.',
        gate='Every named gate point-window must be at most Prefix. Rank by maximum gate variance ratio, '
             'then maximum overall variance ratio, then method name. A missing positive-M2 leaf ranks as infinite.',
        gate_points=sorted(GATE),
        work_proxies='Expected reached internal nodes and expected nonempty child records, including reached null stops.',
        source_sha256={**inputs, combo_index.name: sha(combo_index)},
        scripts_sha256_lf={p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                           for p in (Path(__file__), *(
            scripts / f'light_sampling_{name}.py' for name in
            ('clustering_repair', 'clustering_hierarchy', 'moving_analysis', 'point_screening', 'weight_reference')))},
        replay='python scripts/light_sampling_clustering_rule_screen.py '
               'builds/light-sampling/20261108-clustering-repair/rule-screen/result.json',
        cohorts={}, windows=[],
    )
    cases = []
    with np.load(refs[0]) as fresh, np.load(refs[1]) as old, np.load(refs[2]) as combo:
        for name, identity, keys, root, windows in cohorts(fresh, old, combo, combo_identity):
            prior = recorded['cohorts'][name]
            assert identity == prior['identity']
            print(f'{name}: building and verifying {len(keys)} leaves', flush=True)
            trees, topology = build_cohort(root, keys, prior['topology'])
            result['cohorts'][name] = dict(identity=identity, topology=topology)
            leaf_keys = list(map(tuple, keys))
            for point, window, label, origin, normal, moments in windows:
                origin, normal = origin.astype(float), normal.astype(float)
                rows, evaluations = score_window(trees, leaf_keys, origin, normal, moments,
                                                 previous.pop((name, point, window, label)))
                row = dict(cohort=name, point=point, window=window, previous_label=label,
                           receiver=[origin.tolist(), normal.tolist()],
                           mean_reference_sha256=array_sha(moments[:, :3]),
                           mean_rgb=moments[:, :3].sum(axis=0).tolist(),
                           moments_sha256=array_sha(moments), methods=rows)
                result['windows'].append(row)
                if point in PATH_POINTS:
                    cases.append((row, trees, origin, normal, moments, leaf_keys, evaluations))
                print(f'{name}: {point}/{window} scored', flush=True)
    assert not previous and len(result['windows']) == 28 and len(result['cohorts']) == 5
    summaries = summarize(result['windows'], trees)
    def ranking(name):
        s = summaries[name]
        gate = s['gate_worst']['variance_over_prefix']
        overall = s['worst']['variance_over_prefix']
        return (gate if gate is not None else math.inf, overall if overall is not None else math.inf, name)

    ordered = sorted((name for name in summaries if name not in CONTROLS), key=ranking)
    selected = ordered[:2]
    result.update(
        summaries=summaries, candidate_order=ordered,
        passed_gate=[name for name in ordered if not summaries[name]['gate_failures']],
        diagnostics=[dict(**window_id(row), methods=diagnose(trees, origin, normal, moments, keys,
                                                            evaluations, selected))
                     for row, trees, origin, normal, moments, keys, evaluations in cases],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
