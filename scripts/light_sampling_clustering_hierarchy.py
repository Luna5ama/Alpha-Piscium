"""Isolate upper and brick-local Light-SAH changes with the saved M2 cohorts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_repair import (
    EVIDENCE, build, build_primitives, control_root, ordered_signature, score, stats,
)
from light_sampling_moving_analysis import Node, leaves, merged
from light_sampling_point_screening import Tree


def brick_for(node):
    low = np.floor_divide(node.data[:3], 16)
    high = np.floor_divide(node.data[4:7] - 1, 16)
    return tuple(map(int, low)) if np.array_equal(low, high) else None


def brick_roots(root):
    roots = {}
    pending = [root]
    while pending:
        node = pending.pop()
        brick = brick_for(node)
        if brick is None:
            assert node.children
            pending.extend(node.children)
        else:
            assert brick not in roots, (brick, 'split across maximal single-brick subtrees')
            roots[brick] = node
    return roots


def upper_shape(node):
    brick = brick_for(node)
    return brick if brick is not None else tuple(upper_shape(c) for c in node.children)


def replace_inside(node, replacements, pin_mass, errors):
    brick = brick_for(node)
    if brick is not None:
        new = replacements[brick]
    else:
        children = tuple(replace_inside(c, replacements, pin_mass, errors) for c in node.children)
        new = merged(children)
    np.testing.assert_array_equal(new.data[[0, 1, 2, 4, 5, 6]], node.data[[0, 1, 2, 4, 5, 6]])
    difference = abs(float(new.data[3] - node.data[3]))
    errors.append((difference, difference / float(node.data[3])))
    if pin_mass:
        data = new.data.copy()
        data[3] = node.data[3]
        return Node(data, new.material, new.children)
    return new


def variants(root, keys):
    records = {k[:4]: k[4] for k in leaves(root)}
    assert len(records) == len(keys)
    flux = np.asarray([records[tuple(k)] for k in keys])
    expected = defaultdict(Counter)
    for key, mass in records.items():
        expected[tuple(p // 16 for p in key[:3])][(*key, mass)] += 1

    roots = brick_roots(root)
    assert roots.keys() == expected.keys()
    for brick, node in roots.items():
        assert Counter(leaves(node)) == expected[brick]
    upper = build_primitives(np.asarray(list(roots)), list(roots.values()), True)
    after = brick_roots(upper)
    assert after.keys() == roots.keys()
    assert all(after[brick] is node for brick, node in roots.items())

    replacements = {}
    for brick, node in roots.items():
        rows = np.asarray(leaves(node))
        replacements[brick] = build(rows[:, :4].astype(np.int64), rows[:, 4], True)
        assert Counter(leaves(replacements[brick])) == expected[brick]
    errors = []
    inner = replace_inside(root, replacements, False, errors)
    pinned = replace_inside(root, replacements, True, [])
    assert upper_shape(inner) == upper_shape(pinned) == upper_shape(root)

    global_tree = build(keys, flux, True)
    methods = dict(prefix=root, upper_only=upper, inner_only=inner, global_sah=global_tree)
    expected_records = Counter((*k, v) for k, v in records.items())
    for node in [*methods.values(), pinned]:
        assert Counter(leaves(node)) == expected_records
    # No changes in AABB or ordered structure between the inner controls.
    assert ordered_signature(inner) == ordered_signature(pinned)
    diag = dict(bricks=len(roots),
                inner_mass_max_absolute=max(x[0] for x in errors),
                inner_mass_max_relative=max(x[1] for x in errors))
    return {n: Tree(node) for n, node in methods.items()}, Tree(pinned), diag, {
        n: ordered_signature(node) for n, node in methods.items()
    }


def optimal_root(tree, evaluated, moments):
    result, _, _, q, cond, order = evaluated
    root_children = tree.children[0]
    pmf = cond.sum(axis=1)
    m2 = np.zeros(len(tree.nodes))
    m2[order] = moments[:, 3]
    child_moments = []
    for child in root_children:
        second = 0.
        unsupported = 0
        null = 0.
        leaf_probability = 0.
        pending = [(child, 1.)]
        while pending:
            i, p = pending.pop()
            children = tree.children[i]
            if not len(children):
                leaf_probability += p
                if p > 0:
                    second += m2[i] / p
                elif m2[i] > 0:
                    unsupported += 1
            elif pmf[children].sum() == 0:
                null += p
                # Keep traversing at zero probability to count positive-M2 leaves.
                pending.extend((j, 0.) for j in children)
            else:
                pending.extend((j, p * pmf[j]) for j in children)
        np.testing.assert_allclose(leaf_probability + null, 1, rtol=0, atol=1e-12)
        child_moments.append(dict(**tree.node_description(int(child)),
                                  existing_root_probability=float(q[child]),
                                  conditional_second_moment=float(second) if not unsupported else None,
                                  conditional_null=float(null), missing_positive_support=unsupported))
    if any(row['missing_positive_support'] for row in child_moments):
        return dict(variance=None, children=child_moments)
    a = np.asarray([row['conditional_second_moment'] for row in child_moments])
    old = q[root_children]
    mu = moments[:, :3].sum(0)
    mu2 = float(mu @ mu)
    assert np.all(old[a > 0] > 0)
    np.testing.assert_allclose(np.divide(a, old, out=np.zeros_like(a), where=old > 0).sum()-mu2,
                               result['variance'], rtol=1e-12, atol=1e-16)
    sqrt_a = np.sqrt(a)
    best = sqrt_a / sqrt_a.sum() if sqrt_a.sum() > 0 else old
    for row, probability in zip(child_moments, best):
        row['optimal_root_probability'] = float(probability)
    return dict(variance=float(sqrt_a.sum()**2 - mu2),
                null_probability=float(best @ np.asarray([row['conditional_null'] for row in child_moments])),
                children=child_moments)


def cohorts(fresh, old, combo, combo_identity):
    for scene, info in json.loads(str(fresh['metadata'])).items():
        windows = []
        for i, window in enumerate(info['windows']):
            origin, normal = np.asarray(window['receiver_audit'])[:2, :3]
            windows.append((window['point'], window['window'], None, origin, normal, fresh[f'{scene}/{i}/moments']))
        yield f'new/{scene}', info['identity'], fresh[f'{scene}/keys'], control_root(fresh, f'{scene}/prefix/'), windows

    for group, info in json.loads(str(old['metadata'])).items():
        windows = []
        keys = old[group+'/keys']
        prefix = group+'/world-prefix/'
        for window in info['windows']:
            point, index = window['point'], window['window']
            label = f'{group}/{point}/{index}'
            origin, normal = np.asarray(window['receiver_audit'])[:2, :3]
            windows.append((point, index, label, origin, normal, np.c_[old[label+'/mean_rgb'], old[label+'/m2']]))
        if group != 'ls1':
            assert combo_identity == info['identity']
            np.testing.assert_array_equal(keys, combo['keys'])
            np.testing.assert_array_equal(old[prefix+'nodes'], combo['tree_nodes'][:, :7])
            np.testing.assert_array_equal(old[prefix+'children'], combo['tree_children'])
            for i, label in enumerate(map(str, combo['moment_labels'])):
                prefix_group, point, window = label.split('/')
                if prefix_group == group:
                    j = list(combo['point_ids']).index(point)
                    windows.append((point, int(window[-1]), label, combo['origins'][j], combo['normals'][j],
                                    combo['conditional_moments'][i]))
        yield 'previous/'+group, info['identity'], keys, control_root(old, prefix), windows


def main(output):
    evidence = EVIDENCE/'20261108-clustering-repair.json'
    recorded = json.loads(evidence.read_text(encoding='utf8'))
    old_rows = {r['point_window']: r['methods'] for r in recorded['offline_screen']['rows']}
    new_rows = {(r['scene'], r['point'], r['window']): r['methods'] for r in recorded['new_point_windows']}
    atlas = {row['scene_slot']: row for row in recorded['frozen_prototype']['atlas']['trees']}
    previous_builds = recorded['offline_screen']['builds']
    combo_identity = json.loads((EVIDENCE/'20261108-nearfar-combination.json').read_text(encoding='utf8'))['interior_m2']['cohort']
    refs = [EVIDENCE/f'20261108-{suffix}-reference.npz' for suffix in
            ('clustering-repair', 'cluster-summaries', 'nearfar-combination')]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in refs}
    result = dict(scope='Offline hierarchy isolation with captured conditional RGB moments, 12-bin flux SAH + Kr and the original model 6.',
                  source_sha256=source_hashes, cohorts={}, windows=[],
                  residual_scope='Global - upper-only - inner-only + prefix is a nonadditive remainder. Global crosses the brick boundary and is not the composition of the two controlled substitutions.')
    with np.load(refs[0]) as fresh, np.load(refs[1]) as old, np.load(refs[2]) as combo:
        for name, identity, keys, root, windows in cohorts(fresh, old, combo, combo_identity):
            assert ordered_signature(root) == identity['ordered_topology_sha256']
            methods, pinned, diagnostic, hashes = variants(root, keys)
            stage, group = name.split('/')
            previous_hash = (atlas[int(group)]['semantic_topology_sha256'] if stage == 'new'
                             else previous_builds[group]['light-sah']['topology_sha256'])
            assert hashes['global_sah'] == previous_hash
            leaf_keys = list(map(tuple, keys))
            result['cohorts'][name] = dict(identity=identity, **diagnostic,
                topology={n: dict(**stats(t), ordered_sha256=hashes[n]) for n, t in methods.items()})
            for point, index, label, origin, normal, moments in windows:
                origin, normal = origin.astype(float), normal.astype(float)
                evaluations = {n: score(t, leaf_keys, origin, normal, moments) for n, t in methods.items()}
                pinned_score = score(pinned, leaf_keys, origin, normal, moments)
                original = old_rows[label] if label else new_rows[int(name.split('/')[-1]),point,index]
                for a, b in [('prefix', 'prefix'), ('global_sah', 'light-sah')]:
                    np.testing.assert_allclose(evaluations[a][0]['variance'], original[b]['variance'], rtol=1e-10)
                assert not pinned_score[0]['missing_positive_support']
                base = evaluations['prefix'][0]['variance']
                assert base > 0
                comparison = {n: dict(e[0], variance_over_prefix=e[0]['variance']/base if e[0]['variance'] is not None else None)
                              for n,e in evaluations.items()}
                ratio = {n: v['variance_over_prefix'] for n,v in comparison.items()}
                residual = (ratio['global_sah'] - ratio['upper_only'] - ratio['inner_only'] + 1
                            if all(v is not None for v in ratio.values()) else None)
                measured_inner = evaluations['inner_only']
                row = dict(cohort=name, point=point, window=index, previous_label=label, methods=comparison,
                           global_nonadditive_remainder_over_prefix=residual,
                           inner_pinned_mass_variance_over_prefix=pinned_score[0]['variance']/base,
                           inner_rounding_max_leaf_pmf_delta=float(np.max(np.abs(pinned_score[1]-measured_inner[1]))))
                if point == 'left-red-pillar':
                    root_trials = {n: optimal_root(t, evaluations[n], moments) for n, t in methods.items()}
                    for n, trial in root_trials.items():
                        v = trial['variance']
                        trial['variance_over_unmodified_prefix'] = v/base if v is not None else None
                        excess = comparison[n]['variance']-base
                        if v is not None and excess > 0:
                            trial['fraction_of_excess_removed'] = (comparison[n]['variance']-v)/excess
                    row['root_only_oracle'] = root_trials
                result['windows'].append(row)
                print(name, point, index, {n: round(v, 5) if v is not None else None for n,v in ratio.items()}, flush=True)
    assert len(result['windows']) == 28
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
