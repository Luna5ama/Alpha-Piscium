"""Screen fixed Ward-compressed light summaries against complete point references."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_cluster_capture import GROUPS, RUN, write
from light_sampling_moving_analysis import Node
from light_sampling_point_moments import ROOT, read, sha
from light_sampling_point_screening import Tree, tree_variants
from light_sampling_weight_reference import load_tree

KS = (1, 2, 4)
METHODS = ('prefix-A', 'lookahead-1', 'K1', 'K2', 'K4', 'camera-A', 'flat-leaf-A')
OUTER_CAPACITY = 7620  # G16, four-way, LightSamplingLayout.glsl


def tree_arrays(tree, physical=None):
    children = np.full((len(tree.nodes), 4), -1, dtype='<i4')
    for i, kids in enumerate(tree.children):
        children[i, :len(kids)] = kids
    return dict(nodes=tree.nodes.astype('<f8'), materials=tree.materials.astype('<i4'),
                children=children, slots=physical if physical is not None else (children >= 0).sum(axis=1))


def tree_from(data, prefix):
    nodes, materials, children = (data[prefix + key] for key in ('nodes', 'materials', 'children'))
    parsed = [None] * len(nodes)
    for i in reversed(range(len(nodes))):
        parsed[i] = Node(nodes[i], None if materials[i] < 0 else int(materials[i]),
                         tuple(parsed[j] for j in children[i] if j >= 0))
    return Tree(parsed[0])


def native_slots(tree, path):
    _, nodes, links = load_tree(path, 1)
    native = [1]
    slots = np.zeros(len(tree.nodes), dtype='<i4')
    for i, index in enumerate(native):
        np.testing.assert_array_equal(nodes[index, [0, 1, 2, 4, 5, 6]], tree.nodes[i, [0, 1, 2, 4, 5, 6]])
        assert np.isclose(nodes[index, 3], tree.nodes[i, 3], rtol=2e-6)
        link = int(links[index])
        if link & 0x80000000:
            assert tree.materials[i] == (link & 0x7fffffff)
            continue
        first, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
        slots[i] = count
        start = len(native)
        native.extend(j for j in range(first, first + count) if nodes[j, 3] > 0)
        np.testing.assert_array_equal(tree.children[i], np.arange(start, len(native)))
    assert len(native) == len(tree.nodes)
    return slots, np.asarray(native, dtype='<i4')


def pack_reference(destination):
    """One compressed pack; all references align to each group's prefix leaf order."""
    arrays, groups = {}, {}
    for group in GROUPS:
        folder = RUN / 'captures' / group
        if not (folder / 'accepted.json').exists():
            continue
        meta = read(folder / 'accepted.json')
        fixture = meta['fixture']
        trees, _ = tree_variants(folder, fixture, 'world-prefix')
        prefix = trees['world-prefix']
        slots, addresses = native_slots(prefix, folder / 'tree.bin')
        arrays[group + '/native_addresses'] = addresses
        for layout, tree in trees.items():
            for key, value in tree_arrays(tree, slots if layout == 'world-prefix' else None).items():
                arrays[f'{group}/{layout}/{key}'] = value
        keys = np.asarray(prefix.keys, dtype='<i4')
        arrays[group + '/keys'] = keys
        records = np.fromfile(ROOT / fixture['texture'], '<i4').reshape(-1, 4)[4:]
        index = {tuple(key): i for i, key in enumerate(records)}
        order = [index[key] for key in prefix.keys]
        summary = meta['summary']
        for row in summary:
            with np.load(row['path']) as moments:
                for key in moments.files:
                    arrays[f"{group}/{row['point']}/{row['window']}/{key}"] = moments[key][order]
        groups[group] = {key: meta[key] for key in (
            'folder', 'job_id', 'manifest_id', 'manifest_sha256', 'file_count', 'preset_id', 'revision',
            'camera', 'identity', 'phase_stats', 'context', 'native_capacity')}
        groups[group]['tree_sha256'] = sha(folder / 'tree.bin')
        groups[group]['fixture_sha256'] = fixture['texture_sha256']
        groups[group]['source_uuid'] = meta['provenance']['active_source_uuid']
        groups[group]['source_snapshot_sha256'] = meta['provenance']['source_snapshot_sha256']
        groups[group]['settings'] = {entry['name']: entry['value'] for entry in meta['provenance']['effective_settings']['settings']
                                     if entry['name'].startswith('SETTING_') and entry['changed_from_default']}
        groups[group]['windows'] = [{key: value for key, value in row.items() if key != 'path'} for row in summary]
    assert groups, 'No accepted reference groups'
    if 'sides' in groups and 'side-top' in groups:
        a, b = groups['sides'], groups['side-top']
        assert a['identity'] == b['identity'] and a['context'] == b['context']
        for one, two in zip([r for r in a['windows'] if r['point'] == 'side-27'],
                            [r for r in b['windows'] if r['point'] == 'side-27']):
            np.testing.assert_array_equal(one['receiver_audit'], two['receiver_audit'])
    arrays['metadata'] = np.asarray(json.dumps(groups, allow_nan=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return groups


def fit(tree, k, dtype=np.float64, permute=False):
    """Columns: mass, mean xyz, variance trace. Rank orders full minimum leaf keys."""
    n = len(tree.nodes)
    values = np.zeros((n, k, 5), dtype=dtype)
    ranks = np.full((n, k), -1, dtype='<i4')
    counts = np.zeros(n, dtype='<i4')
    pairs = np.zeros(n, dtype='<i4')
    merges = np.zeros_like(pairs)
    inputs = np.zeros_like(pairs)
    values[tree.leaves, 0, 0] = tree.nodes[tree.leaves, 3]
    values[tree.leaves, 0, 1:4] = tree.nodes[tree.leaves, :3] + .5
    values[tree.leaves, 0, 4] = 5 / 12
    for rank, i in enumerate(sorted(range(len(tree.keys)), key=tree.keys.__getitem__)):
        ranks[tree.leaves[i], 0] = rank
    counts[tree.leaves] = 1
    order = sorted(range(n), key=lambda i: (tree.depth[i], i if not permute else -i), reverse=True)
    triangles = [np.triu_indices(i, 1) for i in range(4 * k + 1)]
    for i in order:
        children = tree.children[i]
        if not len(children):
            continue
        if permute:
            children = children[::-1]
        r = ranks[children].ravel()
        sort = np.argsort(r)
        sort = sort[r[sort] >= 0]
        rank = r[sort]
        v = values[children].reshape(-1, 5)[sort]
        inputs[i] = len(v)
        while len(v) > k:
            a, b = triangles[len(v)]
            ma, mb = v[a, 0], v[b, 0]
            delta = v[b, 1:4] - v[a, 1:4]
            delta2 = (delta * delta).sum(axis=1)
            cost = (ma * mb / (ma + mb)) * delta2
            # r is sorted; argmin's first tie is the lexicographic pair of min leaf keys.
            choice = int(np.argmin(cost))
            left, right = int(a[choice]), int(b[choice])
            total = ma[choice] + mb[choice]
            fraction = mb[choice] / total
            v[left, 0] = total
            v[left, 1:4] += delta[choice] * fraction
            v[left, 4] = ((1 - fraction) * v[left, 4] + fraction * v[right, 4] +
                          (1 - fraction) * fraction * delta2[choice])
            v = np.delete(v, right, axis=0)
            rank = np.delete(rank, right)
            pairs[i] += len(a)
            merges[i] += 1
        counts[i] = len(v)
        values[i, :len(v)] = v
        ranks[i, :len(v)] = rank
    assert np.isfinite(values).all() and (values[:, :, [0, 4]] >= 0).all()
    return values, ranks, counts, dict(pairs=pairs, merges=merges, inputs=inputs)


def verify_moments(tree, values, dtype):
    anchor = tree.nodes[0, :3]
    leaf_center = tree.nodes[tree.leaves, :3] + .5 - anchor
    mass = tree.nodes[tree.leaves, 3:4]
    direct = tree.aggregate(np.c_[mass, mass * leaf_center, mass[:, 0] * (5 / 12 + (leaf_center ** 2).sum(axis=1))])
    ref_mean = direct[:, 1:4] / direct[:, 0:1]
    ref_var = direct[:, 4] / direct[:, 0] - (ref_mean ** 2).sum(axis=1)
    m = values[:, :, 0].astype(float)
    total = m.sum(axis=1)
    centers = values[:, :, 1:4].astype(float) - anchor
    mean = (m[:, :, None] * centers).sum(axis=1) / total[:, None]
    var = (m * (values[:, :, 4] + ((centers - mean[:, None, :]) ** 2).sum(axis=2))).sum(axis=1) / total
    mass_error = float(np.max(abs(total / direct[:, 0] - 1)))
    center_error = float(np.max(abs(mean - ref_mean)))
    variance_error = float(np.max(abs(var - ref_var) / np.maximum(ref_var, 1)))
    limits = (2e-14, 2e-10, 1e-9) if dtype == np.float64 else (3e-6, 5e-4, 1e-3)
    assert all(a < b for a, b in zip((mass_error, center_error, variance_error), limits)), (mass_error, center_error, variance_error)
    return dict(max_relative_mass_error=mass_error, max_abs_center_error_blocks=center_error,
                max_variance_error_over_max_1_variance=variance_error)


def geometry(tree, origin, normal, dtype):
    nodes, origin, normal = (np.asarray(v, dtype=dtype) for v in (tree.nodes, origin, normal))
    half = (nodes[:, 4:7] - nodes[:, :3]) * .5
    delta = nodes[:, :3] + half - origin
    height = (delta * normal).sum(axis=1)
    extent = (half * np.abs(normal)).sum(axis=1)
    front = height + extent >= -1e-6 * ((np.abs(delta * normal)).sum(axis=1) + extent + 1)
    return origin, normal, front


def candidate_weights(tree, fits, origin, normal, eligible, dtype):
    # A and lookahead use the previously reviewed scorer unchanged.
    base, _ = tree.weights(origin, normal)
    origin, normal, front = geometry(tree, origin, normal, dtype)
    weights = [base[:, 0], base[:, 4]]
    for k, values in zip(KS, fits):
        result = base[:, 0].copy()
        use = eligible & front
        result[eligible] = 0
        component = values[use]
        delta = component[:, :, 1:4] - origin
        d2 = (delta * delta).sum(axis=2)
        r2 = 2 * component[:, :, 4]
        c = np.clip(((delta * normal).sum(axis=2) + np.sqrt(r2)) / np.sqrt(np.maximum(d2, .25)), .001, 1)
        m = (tree.nodes[use, 3:4].astype(dtype) if k == 1 and dtype == np.float32 else component[:, :, 0])
        result[use] = (m * c / np.maximum(np.maximum(d2, r2), .25)).sum(axis=1)
        weights.append(result)
    return np.stack(weights, axis=1), front


def summarize_paths(tree, q, work):
    cumulative = work.copy()
    for i, children in enumerate(tree.children):
        cumulative[children] += cumulative[i]
    return {key: dict(mean_per_sample=float(q @ values), max_on_reachable_path=int(cumulative[q > 0, j].max()))
            for j, (key, values) in enumerate(zip(WORK_FIELDS, work.T))}


WORK_FIELDS = ('logical_node_records', 'physical_node_slots', 'empty_node_mass_tests', 'A_weight_calls_nonempty',
               'half_space_checks', 'summary_count_words', 'summary_component_slots', 'component_kernels',
               'sibling_weight_adds', 'sampling_draw_steps', 'normalizations', 'selection_slot_scan_upper_bound',
               'requested_node_bytes', 'requested_summary_bytes', 'selected_node_rereads', 'component_weight_adds')


def query_work(tree, slots, q, weights, eligible, front, counts, k):
    work = np.zeros((len(tree.nodes), len(WORK_FIELDS)), dtype=np.int64)
    work[0, [0, 1, 12]] = (1, 1, 32)  # load root once
    for i, children in enumerate(tree.children):
        if not len(children):
            continue
        physical, logical = int(slots[i]), len(children)
        touched = children[eligible[children]]
        expanded = touched if k == -1 else ()
        extra, extra_live = sum(slots[j] for j in expanded), sum(len(tree.children[j]) for j in expanded)
        summaries = touched[front[touched]] if k > 0 else ()
        components = sum(counts[j] for j in summaries)
        a_calls = logical - (len(touched) if k else 0) + extra_live
        total_positive = int(weights[children].sum() > 0)
        work[i] += (logical + extra_live, physical + extra, physical - logical + extra - extra_live,
                    a_calls, a_calls + (len(touched) if k > 0 else 0), len(summaries) if k > 1 else 0,
                    components, components, physical + extra, total_positive, total_positive,
                    physical * total_positive, 32 * (physical + extra + total_positive),
                    16 * components if k == 1 else (4 * len(summaries) + 20 * components if k > 1 else 0),
                    total_positive, components if k > 1 else 0)
    return summarize_paths(tree, q, work)


def build_work(tree, slots, capacity, eligible, k, counts, stats):
    # Proposed separate GPU build: breadth worklist by native index, then reverse levels,
    # with fixed-capacity scratch including a full canonical key per intermediate component.
    internal = tree.materials < 0
    scheduled = len(tree.nodes)
    depths = int(tree.depth.max()) + 1
    level_capacity = 49  # 42 world + 6 in-brick splits, plus root
    worklist = 4 * capacity + 32 * level_capacity
    assert depths <= level_capacity
    scratch_stride = 36 * k + (4 if k > 1 else 0)
    stride = 16 if k == 1 else 20 * k + 4
    base_slots = int(slots.sum())
    live = int(counts[eligible].sum())
    components = int(counts.sum())
    child_components = int(stats['inputs'].sum())
    count_reads = scheduled - 1 if k > 1 else 0
    count_writes = scheduled if k > 1 else 0
    return dict(
        allocation=dict(base_node_capacity=capacity, base_node_bytes=32 * capacity + 32,
                        outer_summary_capacity=OUTER_CAPACITY, active_summary_nodes=int(eligible.sum()),
                        active_components=live, record_stride_bytes=stride,
                        active_payload_bytes=live * (16 if k == 1 else 20),
                        active_count_bytes=4 * int(eligible.sum()) if k > 1 else 0, index_bytes=0,
                        persistent_summary_bytes=OUTER_CAPACITY * stride,
                        build_component_stride_bytes=scratch_stride, build_worklist_bytes=worklist,
                        build_component_scratch_bytes=capacity * scratch_stride,
                        peak_additional_bytes=OUTER_CAPACITY * stride + worklist + capacity * scratch_stride,
                        per_invocation_merge_array_bytes=4 * k * 36),
        full_tree=dict(scheduled_nodes=scheduled, initialized_leaves=int((~internal).sum()), internal_nodes=int(internal.sum()),
                       breadth_and_reverse_active_dispatches=2 * depths, active_level_finalizers=depths - 1,
                       fixed_submission_dispatches=3 * level_capacity - 1,
                       physical_child_slots_per_tree_walk=base_slots,
                       logical_child_links_per_tree_walk=scheduled - 1, empty_slots_per_walk=base_slots - scheduled + 1,
                       schedule_worklist_bytes_written=4 * scheduled, schedule_worklist_bytes_read_twice=8 * scheduled,
                       schedule_node_bytes=32 * (1 + base_slots), build_node_bytes=32 * (scheduled + base_slots),
                       child_components_read=child_components, components_written=components,
                       temporary_bytes_read=36 * child_components + 4 * count_reads,
                       temporary_bytes_written=36 * components + 4 * count_writes,
                       persistent_summary_bytes_written=(16 if k == 1 else 20) * live + (4 * int(eligible.sum()) if k > 1 else 0),
                       merges=int(stats['merges'].sum()), candidate_pair_comparisons=int(stats['pairs'].sum()),
                       initial_key_sort_comparison_upper_bound=int((stats['inputs'] * (stats['inputs'] - 1) // 2).sum()),
                       max_parent_pair_comparisons=int(stats['pairs'].max()), max_parent_input_components=int(stats['inputs'].max())))


def metric(tree, p, q, mu, m2, null, column=0):
    failures = (m2 > 0) & (p == 0)
    terms = np.divide(m2, p, out=np.zeros_like(m2), where=p > 0)
    total = mu.sum(axis=0)
    return dict(variance=None if failures.any() else float(terms.sum() - total @ total),
                null_probability=float(null), support_failures=[tree.keys[i] for i in np.flatnonzero(failures)],
                penalty_lights=[dict(key=tree.keys[i], q=float(p[i]), m2=float(m2[i]), term=float(terms[i]))
                                for i in np.argsort(terms)[-6:][::-1]],
                splits=[] if q is None else tree.splits(q, np.sqrt(m2), column, limit=4))


def screen(reference, output):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    builds = {}
    with np.load(reference, allow_pickle=False) as data:
        groups = json.loads(str(data['metadata']))
        for group, group_meta in groups.items():
            prefix = f'{group}/world-prefix/'
            tree = tree_from(data, prefix)
            np.testing.assert_array_equal(data[group + '/keys'], tree.keys)
            slots = data[prefix + 'slots']
            camera = tree_from(data, f'{group}/camera-relative/')
            lookup = {key: i for i, key in enumerate(camera.keys)}
            camera_order = np.array([lookup[key] for key in tree.keys])
            np.testing.assert_array_equal(camera.nodes[camera.leaves[camera_order], 3], tree.nodes[tree.leaves, 3])
            _, eligible = tree.weights(np.zeros(3), np.array([0., 1., 0.]))
            eligible &= tree.materials < 0
            eligible[0] = False  # the root is entered without a candidate weight
            assert (data[group + '/native_addresses'][eligible] < OUTER_CAPACITY).all()
            fits, float_fits, counts, saved = [], [], {}, {}
            builds[group] = {}
            for k in KS:
                for dtype, name, fit_list in ((np.float64, 'f64', fits), (np.float32, 'f32', float_fits)):
                    v, r, c, stats = fit(tree, k, dtype)
                    rerun = fit(tree, k, dtype, permute=True)
                    for a, b in zip((v, r, c), rerun):
                        np.testing.assert_array_equal(a, b)
                    check = verify_moments(tree, v, dtype)
                    fit_list.append(v)
                    for key, value in (('values', v), ('key_rank', r), ('count', c)):
                        saved[f'K{k}_{name}_{key}'] = value
                    if name == 'f64':
                        counts[k] = c
                        builds[group][f'K{k}'] = build_work(tree, slots, group_meta['native_capacity'], eligible, k, c, stats)
                    builds[group][f'K{k}'][name + '_validation'] = check
                builds[group][f'K{k}']['f32_component_min_key_ranks_changed'] = int(np.count_nonzero(saved[f'K{k}_f64_key_rank'] != saved[f'K{k}_f32_key_rank']))
            np.savez_compressed(output / f'{group}-components.npz', **saved, eligible=eligible)
            for point in group_meta['windows']:
                name, window = point['point'], point['window']
                audit = np.asarray(point['receiver_audit'])
                origin, normal = audit[0, :3], audit[1, :3]
                key = f'{group}/{name}/{window}/'
                mu, m2 = data[key + 'mean_rgb'], data[key + 'm2']
                assert np.isfinite(mu).all() and np.isfinite(m2).all() and (m2 >= 0).all()
                np.testing.assert_array_equal(data[key + 'complete_samples'], data[key + 'potential_samples'])
                weights, front = candidate_weights(tree, fits, origin, normal, eligible, np.float64)
                f32_weights, _ = candidate_weights(tree, float_fits, origin, normal, eligible, np.float32)
                q, null = tree.pmf(weights)
                q32, null32 = tree.pmf(f32_weights)
                camera_q, camera_null = camera.pmf(camera.weights(origin, normal)[0][:, :1])
                leaf_a = weights[tree.leaves, 0]
                a_sum = leaf_a.sum()
                flat = leaf_a / a_sum if a_sum > 0 else np.zeros_like(leaf_a)
                leaves = np.c_[q[tree.leaves], camera_q[camera.leaves[camera_order], 0], flat]
                all_null = np.r_[null, camera_null, float(a_sum == 0)]
                np.testing.assert_allclose(leaves.sum(axis=0) + all_null, 1, atol=2e-12)
                methods = {}
                for j, method in enumerate(METHODS):
                    if j == 5:
                        inv = np.argsort(camera_order)
                        methods[method] = metric(camera, camera_q[camera.leaves, 0], camera_q,
                                                 mu[inv], m2[inv], camera_null[0])
                    else:
                        methods[method] = metric(tree, leaves[:, j], q if j < 5 else None, mu, m2, all_null[j], j)
                    if j < 5:
                        k = (0, -1, *KS)[j]
                        methods[method]['query_work'] = query_work(tree, slots, q[:, j], weights[:, j], eligible,
                                                                  front, counts.get(k), k)
                        if k > 0:
                            p32 = q32[tree.leaves, j]
                            methods[method]['f32_summary'] = dict(
                                full_pmf_total_variation=float(.5 * (abs(p32 - leaves[:, j]).sum() + abs(null32[j] - null[j]))),
                                support_failures=int(((m2 > 0) & (p32 == 0)).sum()),
                                variance=metric(tree, p32, None, mu, m2, null32[j])['variance'])
                for row in methods.values():
                    row['ratios'] = {m: row['variance'] / methods[m]['variance'] if row['variance'] is not None
                                     and methods[m]['variance'] not in (0, None) else None
                                     for m in ('prefix-A', 'camera-A', 'flat-leaf-A')}
                flat_node = tree.aggregate(leaf_a[:, None])[:, 0]
                node_m2 = tree.aggregate(m2[:, None])[:, 0]
                max_leaf = np.zeros(len(tree.nodes))
                max_leaf[tree.leaves] = leaf_a
                for i in reversed(range(len(tree.nodes))):
                    if len(tree.children[i]):
                        max_leaf[i] = max(max_leaf[tree.children[i]])
                residual = np.divide(weights, flat_node[:, None], out=np.zeros_like(weights), where=flat_node[:, None] > 0)
                positive_clusters = eligible & (flat_node > 0)
                aggregate = {}
                for j, method in enumerate(METHODS[:5]):
                    idx = np.flatnonzero(positive_clusters)
                    strongest = idx[np.argsort(residual[idx, j])[:5]]
                    aggregate[method] = [dict(**tree.node_description(int(i)), weight_over_flat_sum=float(residual[i, j]),
                                              largest_leaf_share=float(max_leaf[i] / flat_node[i]),
                                              path_probability=float(q[i, j]), measured_m2=float(node_m2[i])) for i in strongest]
                low = [dict(**tree.node_description(int(i)), q=dict(zip(METHODS[:5], q[i].tolist())),
                            observed_M2_share=float(node_m2[i] / m2.sum()) if m2.sum() else None)
                       for i in tree.children[0] if tree.nodes[i, 5] <= 0]
                np.savez_compressed(output / f'{group}-{name}-window{window}.npz', keys=data[group + '/keys'],
                                    leaf_pmf=leaves, null_probability=all_null, methods=np.asarray(METHODS),
                                    node_arrival=q, node_weights=weights, leaf_A_sum_by_node=flat_node, node_m2=node_m2,
                                    f32_leaf_pmf=q32[tree.leaves], f32_null_probability=null32)
                results.append(dict(group=group, point=name, window=window, methods=methods,
                                    low_root_children=low, most_underestimated_clusters=aggregate))
    write(output / 'screening.json', dict(groups=groups, build=builds, results=results))
    return dict(groups=list(groups), rows=len(results), output=str(output / 'screening.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('pack', 'screen'))
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=RUN / 'screening')
    args = parser.parse_args()
    result = list(pack_reference(args.reference)) if args.operation == 'pack' else screen(args.reference, args.output)
    print(json.dumps(result))
