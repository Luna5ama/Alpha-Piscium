"""Independent audit of wide light-tree captures."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from light_sampling_geometry import audit_cube, audit_tree
from light_sampling_gilbert_analysis import check_settings
from light_sampling_runtime import LUMA, read_image, render, tree_stats
from light_sampling_spatial_new_scenes import ROIS

ROOT = Path('builds/light-sampling/20261030-wide-tree')
CASES = {'binary-off': '921caa33fadc78dda6bf9de2f6c75b6ff0332b4b',
         'binary-ahead2': '921caa33fadc78dda6bf9de2f6c75b6ff0332b4b',
         'wide4-off': 'e9a0dfd09660d28da18c5e14c61d7537875a4498',
         'wide8-off': '073f6bd1f026dfc2274e559e3bc66db4b3831e09'}


def leaf_set(path, wide):
    words = np.fromfile(path, '<u4')[8:].reshape(-1, 8)
    nodes = words.view('<f4')
    pending, leaves = [1], []
    while pending:
        i = pending.pop()
        node, link = nodes[i], int(words[i, 7])
        if node[3] == 0:
            continue
        if link & 0x80000000:
            leaves.append((tuple(map(float, node[:3])), link & 0x7fffffff, float(node[3])))
        elif wide:
            base, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
            pending.extend(range(base, base + count))
        else:
            pending.extend((link, link + 1))
    return sorted(leaves)


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
    probe = next(record for record in records if record[3, 3] != 0)
    origin, normal = probe[0, :3], probe[1, :3]
    pending = [(1, 1.0)]
    pmf_sum = 0.0
    while pending:
        i, probability = pending.pop()
        node, link = nodes[i], int(words[i, 7])
        if link & 0x80000000:
            pmf_sum += probability
            continue
        base, count = link & 0x0fffffff, ((link >> 28) & 7) + 1
        child = nodes[base:base + count]
        half = (child[:, 4:7] - child[:, :3]) * .5
        delta = child[:, :3] + half - origin
        r2 = np.sum(half * half, axis=1)
        d2 = np.sum(delta * delta, axis=1)
        cosine = np.clip((delta @ normal + np.sqrt(r2)) / np.sqrt(np.maximum(d2, .25)), .001, 1)
        weight = child[:, 3] * cosine / np.maximum(np.maximum(d2, r2), .25)
        total = weight.sum()
        assert total > 0
        pending.extend((base + j, probability * w / total) for j, w in enumerate(weight) if w > 0)
    assert abs(pmf_sum - 1) < 1e-10
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
                child_counts={str(n): counts.count(n) for n in range(1, arity + 1)}, pmf_sum=pmf_sum,
                samples=len(errors), max_pdf_relative_error=max(errors))


def analyze():
    rows, timings, images = [], [], []
    for number in range(1, 6):
        scene = f'light-sampling-{number}'
        captures, scene_hashes, leaf_sets = [], set(), []
        for name, revision in CASES.items():
            d = ROOT / 'raw' / scene / name
            manifest = json.loads((d / 'manifest.json').read_text())
            for item in manifest['files']:
                path = d / item['relative_path']
                assert path.stat().st_size == item['byte_size']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            job = json.loads((d / 'job-result.json').read_text())
            assert job['success'] and all(r['status'] == 'RECEIPT_STATUS_OK' for r in job['result']['action_receipts'])
            result = job['result']
            p = result['provenance']
            assert p['resolved_revision'] == revision and p['preset_id'] == scene
            assert not p['stale'] and not p['head_changed'] and not p['dirty_shader_delta_sha256']
            check_settings(result['action_receipts'][0]['runtime_mutation']['effective_settings'], name == 'binary-ahead2', True)
            scene_hashes.add(p['scene_sha256'])
            wide = name.startswith('wide')
            leaf_sets.append(leaf_set(d / 'tree.bin', wide))
            tree = audit(d, 4 if name == 'wide4-off' else 8) if wide else tree_stats(d / 'tree.bin')
            pdf = {'samples': tree['samples'], 'max_relative_pdf_error': tree['max_pdf_relative_error']} if wide else audit_tree(d / 'tree.bin', d / 'audit.bin', False, name == 'binary-ahead2')
            cube = audit_cube(d / 'audit.bin')
            single, sm = read_image(d / 'single.bin')
            mean, mm = read_image(d / 'mean32.mean.bin')
            variance, _ = read_image(d / 'mean32.variance.bin')
            count, _ = read_image(d / 'mean32.valid_count.bin')
            incomplete, _ = read_image(d / 'mean32.incomplete_batches.bin')
            assert (count >= 0).all() and (count <= 32).all() and (variance >= 0).all()
            complete = (count == 32).all(2) & (incomplete == 0).all(2)
            accum = mm['accumulation']
            assert accum['frames'] == 32 and accum['end_frame'] - accum['start_frame'] == 31 and sm['frame_id'] < accum['start_frame']
            assert json.loads((d / 'tree.json').read_text())['frame_id'] == json.loads((d / 'audit.json').read_text())['frame_id']
            row = dict(scene=scene, case=name, revision=revision, scene_sha256=p['scene_sha256'], source_sha256=p['source_snapshot_sha256'],
                       incomplete_pixels=int((~complete).sum()), missing_samples_by_channel=[int((32 - count[:, :, i]).sum()) for i in range(4)],
                       tree=tree, pdf=pdf, cube_pdf=cube)
            rows.append(row)
            captures.append((row, mean, variance, complete))
            for label, data in [('Single frame', single), ('32-frame mean', mean)]:
                image_name = f'{scene}-{name}-{label.replace(" ", "-")}.png'
                render(data, ROOT / image_name, 6)
                images.append(dict(path=image_name, dimensions={'Scene': scene, 'Method': name, 'Accumulation': label}))
        assert len(scene_hashes) == 1 and all(x == leaf_sets[0] for x in leaf_sets)
        receiver = captures[0][1][:, :, 3] > 0
        assert all(np.array_equal(receiver, c[1][:, :, 3] > 0) for c in captures)
        receiver &= np.logical_and.reduce([c[3] for c in captures])
        shared = gaussian_filter(np.mean([m[:, :, :3] @ LUMA for _, m, _, _ in captures], axis=0), 2)
        lit = receiver & (shared > 1e-6)
        q50, q99 = np.quantile(shared[lit], [.5, .99])
        masks = {'dark_half': lit & (shared <= q50), 'brightest_1': lit & (shared >= q99)}
        for key, (x0, y0, x1, y1) in ROIS[scene].items():
            mask = np.zeros_like(receiver)
            mask[y0:y1, x0:x1] = True
            masks[key] = mask[::-1] & receiver
        for row, _, variance, _ in captures:
            row['roi_coordinates'] = ROIS[scene]
            row['mask_pixels'] = {k: int(v.sum()) for k, v in masks.items()}
            row['variance'] = {k: float(variance[:, :, :3].sum(2)[v].mean()) for k, v in masks.items()}
        for row, _, _, _ in captures:
            row['variance_over_binary_off'] = {k: v / captures[0][0]['variance'][k] for k, v in row['variance'].items()}
        order = list(CASES) + list(reversed(CASES))
        for index, name in enumerate(order, 1):
            d = ROOT / 'profiles' / scene / f'{index:02d}-{name}'
            job = json.loads((d / 'job-result.json').read_text())
            assert job['success'] and len(job['cases']) == 1
            case = job['cases'][0]
            assert case['status'] == 'passed' and case['warmup_frames'] == case['frames'] == 128
            p = case['provenance']
            assert p['resolved_revision'] == CASES[name] and p['preset_id'] == scene
            assert not p['stale'] and not p['head_changed'] and not p['dirty_shader_delta_sha256']
            check_settings(p['effective_settings'], name == 'binary-ahead2', False)
            quality = next(r for r in rows if r['scene'] == scene and r['case'] == name)
            assert p['scene_sha256'] == quality['scene_sha256'] and p['source_snapshot_sha256'] == quality['source_sha256']
            required = {'composite41_compute', 'prepare2_compute'} | {f'prepare{i}_compute' for i in range(6, 20)}
            assert not any(c['metric_id'] in required for c in job['metric_conflicts'])
            metrics = {m['metric_id']: m for m in case['metrics']['metrics'] if m.get('main_source')}
            tree = [m for m in metrics.values() if m['main_source'] == '/pass/prepare/LightSamplingBuild.comp.glsl']
            assert {m['metric_id'] for m in tree} == {f'prepare{i}_compute' for i in range(6, 20)}
            assert metrics['composite41_compute']['main_source'] == '/pass/composite/LightSamplingInitial.comp.glsl'
            assert metrics['prepare2_compute']['main_source'] == '/pass/shadow/VoxelTreeBuilder.comp.glsl'
            timings.append(dict(scene=scene, case=name, order=index, initial_ms=metrics['composite41_compute']['p50_ns'] / 1e6,
                                brick_ms=metrics['prepare7_compute']['p50_ns'] / 1e6,
                                tree_sum_avg_ms=sum(m['average_ns'] for m in tree) / 1e6,
                                sentinel_ms=metrics['prepare2_compute']['p50_ns'] / 1e6))
        print(scene, 'validated', flush=True)
    (ROOT / 'images.json').write_text(json.dumps(dict(title='Wide light tree comparison', images=images), indent=2), encoding='utf-8')
    (ROOT / 'summary.json').write_text(json.dumps(dict(quality=rows, timing=timings), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(json.dumps(audit(Path(sys.argv[1]), int(sys.argv[2])), indent=2))
    else:
        analyze()
