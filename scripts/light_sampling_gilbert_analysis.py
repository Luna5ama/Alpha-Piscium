"""Analyze the five-scene Morton/Gilbert experiment from preserved Vibris data."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from light_sampling_runtime import LUMA, read_image, render, tree_stats
from light_sampling_geometry import audit_cube, audit_tree
from light_sampling_spatial_new_scenes import ROIS

ROOT = Path('builds/light-sampling/20261030-gilbert')
REVISIONS = {'morton': '181878b9c36d66bb8d109caac1e86c53d61a2d5a',
             'gilbert': '3e28b744833b88d48bb07e611984717032e0d3a5'}
CASES = [(curve, ahead) for curve in REVISIONS for ahead in (False, True)]
COMMON = dict(SETTING_RC_ENABLE='false', SETTING_GI_PATH_GUIDING='false', SETTING_GI_SPATIAL_REUSE='false',
              SETTING_GI_DECORRELATE='false', SETTING_GI_USE_REFERENCE='0', SETTING_PBR_MATERIAL='0',
              SETTING_NORMAL_MAPPING='false', SETTING_TAA='false', SETTING_TAA_JITTER='false',
              SETTING_DENOISER_SPATIAL='false', SETTING_DENOISER_ACCUM='false',
              SETTING_DENOISER_ANTI_FIREFLY='false', SETTING_VOXEL_GRID_SIZE='16')


def check_settings(settings, ahead, audit):
    expected = dict(COMMON, SETTING_DEBUG_LIGHT_CUBE='true', SETTING_DEBUG_LIGHT_WEIGHT='false',
                    SETTING_DEBUG_LIGHT_SAMPLING='2', SETTING_DEBUG_LIGHT_AUDIT=str(audit).lower(),
                    SETTING_DEBUG_LIGHT_LOOKAHEAD=str(ahead).lower())
    actual = {s['name']: s['value'] for s in settings['settings']}
    assert all(actual[k] == v for k, v in expected.items())


def tree_depths(path):
    words = np.fromfile(path, '<u4')[8:].reshape(-1, 8)
    values = words.view('<f4')
    pending, depths, checked = [(1, 0)], [], 0
    while pending:
        node, depth = pending.pop()
        if values[node, 3] == 0:
            continue
        link = int(words[node, 7])
        if link & 0x80000000:
            depths.append(depth)
        else:
            if node >= 8192:
                left, right = values[link:link + 2]
                assert left[3] > 0 and right[3] > 0
                assert np.any(left[4:7] <= right[:3]) or np.any(right[4:7] <= left[:3])
                checked += 1
            pending.extend([(link, depth + 1), (link + 1, depth + 1)])
    return dict(mean_leaf_depth=float(np.mean(depths)), max_leaf_depth=max(depths), disjoint_brick_nodes=checked)


def main():
    rows, images, timing = [], [], []
    for number in range(1, 6):
        scene = f'light-sampling-{number}'
        captures, hashes, leaves = [], set(), set()
        for curve, ahead in CASES:
            case = curve + ('-ahead' if ahead else '-plain')
            directory = ROOT / 'raw' / scene / case
            manifest = json.loads((directory / 'manifest.json').read_text())
            for artifact in manifest['files']:
                path = directory / artifact['relative_path']
                assert path.stat().st_size == artifact['byte_size']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
            job = json.loads((directory / 'job-result.json').read_text())
            assert job['success']
            result = job['result']
            assert all(a['status'] == 'RECEIPT_STATUS_OK' for a in result['action_receipts'])
            p = result['provenance']
            assert p['resolved_revision'] == REVISIONS[curve] and p['preset_id'] == scene
            assert not p['stale'] and not p['head_changed'] and not p['dirty_shader_delta_sha256']
            mutation = result['action_receipts'][0]['runtime_mutation']
            check_settings(mutation['effective_settings'], ahead, True)
            hashes.add(p['scene_sha256'])
            single, single_meta = read_image(directory / 'single.bin')
            mean, mean_meta = read_image(directory / 'mean32.mean.bin')
            variance, _ = read_image(directory / 'mean32.variance.bin')
            count, _ = read_image(directory / 'mean32.valid_count.bin')
            incomplete, _ = read_image(directory / 'mean32.incomplete_batches.bin')
            assert (count >= 0).all() and (count <= 32).all() and (variance >= 0).all()
            complete = (count == 32).all(2) & (incomplete == 0).all(2)
            accumulation = mean_meta['accumulation']
            assert accumulation['end_frame'] - accumulation['start_frame'] == 31
            assert single_meta['frame_id'] < accumulation['start_frame']
            assert json.loads((directory / 'tree.json').read_text())['frame_id'] == json.loads((directory / 'audit.json').read_text())['frame_id']
            tree = tree_stats(directory / 'tree.bin') | tree_depths(directory / 'tree.bin')
            leaves.add(tree['leaf_set_sha256'])
            row = dict(scene=scene, curve=curve, lookahead=2 if ahead else 0, revision=p['resolved_revision'],
                       scene_sha256=p['scene_sha256'], source_sha256=p['source_snapshot_sha256'],
                       incomplete_pixels=int((~complete).sum()),
                       missing_samples_by_channel=[int((32 - count[:, :, i]).sum()) for i in range(4)], tree=tree,
                       pdf=audit_tree(directory / 'tree.bin', directory / 'audit.bin', False, ahead),
                       cube_pdf=audit_cube(directory / 'audit.bin'))
            rows.append(row)
            captures.append((row, mean, variance, complete))
            for label, data in [('Single frame', single), ('32-frame mean', mean)]:
                name = f'{scene}-{case}-{label.replace(" ", "-")}.png'
                render(data, ROOT / name, 6)
                images.append(dict(path=name, dimensions={'Scene': scene, 'Curve': curve.title(),
                                   'Lookahead': '2 levels' if ahead else 'Off', 'Accumulation': label}))
        assert len(hashes) == len(leaves) == 1
        receiver = captures[0][1][:, :, 3] > 0
        assert all(np.array_equal(receiver, c[1][:, :, 3] > 0) for c in captures)
        receiver &= np.logical_and.reduce([c[3] for c in captures])
        shared = gaussian_filter(np.mean([m[:, :, :3] @ LUMA for _, m, _, _ in captures], axis=0), 2)
        lit = receiver & (shared > 1e-6)
        q50, q99 = np.quantile(shared[lit], [.5, .99])
        masks = {'dark_half': lit & (shared <= q50), 'brightest_1': lit & (shared >= q99)}
        for name, (x0, y0, x1, y1) in ROIS[scene].items():
            mask = np.zeros_like(receiver)
            mask[y0:y1, x0:x1] = True
            masks[name] = mask[::-1] & receiver
        for row, mean, variance, _ in captures:
            row['roi_coordinates'] = ROIS[scene]
            row['mask_pixels'] = {k: int(mask.sum()) for k, mask in masks.items()}
            row['variance'] = {k: float(variance[:, :, :3].sum(2)[mask].mean()) for k, mask in masks.items()}
        for row, _, _, _ in captures:
            row['variance_over_morton_plain'] = {k: v / captures[0][0]['variance'][k] for k, v in row['variance'].items()}
            baseline = captures[1 if row['lookahead'] else 0][0]['variance']
            row['variance_over_morton_same_lookahead'] = {k: v / baseline[k] for k, v in row['variance'].items()}

        profile = json.loads((ROOT / 'profiles' / scene / 'profile.json').read_text())
        assert profile['success'] and len(profile['cases']) == 8
        required = {'composite41_compute', 'prepare2_compute'} | {f'prepare{i}_compute' for i in range(6, 20)}
        assert not any(c['metric_id'] in required for c in profile['metric_conflicts'])
        for case in profile['cases']:
            assert case['status'] == 'passed' and case['warmup_frames'] == case['frames'] == 128
            curve, ahead = case['source_id'], case['config_id'] == 'ahead'
            p = case['provenance']
            assert p['resolved_revision'] == REVISIONS[curve] and p['preset_id'] == scene
            assert not p['stale'] and not p['head_changed'] and not p['dirty_shader_delta_sha256']
            check_settings(p['effective_settings'], ahead, False)
            quality = next(r for r in rows if r['scene'] == scene and r['curve'] == curve and r['lookahead'] == (2 if ahead else 0))
            assert p['scene_sha256'] == quality['scene_sha256'] and p['source_snapshot_sha256'] == quality['source_sha256']
            metrics = {m['metric_id']: m for m in case['metrics']['metrics'] if m.get('main_source')}
            tree_metrics = [m for m in metrics.values() if m['main_source'] == '/pass/prepare/LightSamplingBuild.comp.glsl']
            assert len(tree_metrics) == 14 and metrics['prepare7_compute'] in tree_metrics
            timing.append(dict(scene=scene, curve=curve, lookahead=2 if ahead else 0, run=case['run'],
                               initial_p50_ms=metrics['composite41_compute']['p50_ns'] / 1e6,
                               brick_p50_ms=metrics['prepare7_compute']['p50_ns'] / 1e6,
                               tree_sum_avg_ms=sum(m['average_ns'] for m in tree_metrics) / 1e6,
                               sentinel_p50_ms=metrics['prepare2_compute']['p50_ns'] / 1e6))
    mapping = json.loads((ROOT / 'mapping-validation.json').read_text())
    result = dict(protocol='1080p diffuse direct only; cube sampling, old weights, no visibility, RC, ReSTIR, denoising or TAA. 256 warmup, one single frame and 32-frame mean/variance in eight batches. Common complete-pixel mask; missing samples are reported. Variance is not reference MSE. Profiling uses 128 warmup and 128 measured frames, palindrome order, two runs per case; tree sum is the sum of average times for prepare6 through prepare19. Unrelated duplicate pass metrics have conflicts; all reported metrics are checked conflict-free.', mapping=mapping, quality=rows, timing=timing)
    (ROOT / 'images.json').write_text(json.dumps(dict(title='Morton and Gilbert light grouping', images=images), indent=2), encoding='utf-8')
    data = json.dumps(result, indent=2) + '\n'
    (ROOT / 'summary.json').write_text(data, encoding='utf-8')
    Path('experiments/light-sampling/20261030-gilbert.json').write_text(data, encoding='utf-8')
    print(f'Analyzed {len(rows)} quality cases, {len(images)} images and {len(timing)} timing runs')


if __name__ == '__main__':
    main()
