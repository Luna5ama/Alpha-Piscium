"""Analyze the dedicated native-resolution light-sampling captures."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from light_sampling_runtime import LUMA, render

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'builds/light-sampling/20261031-weights-presets1080'
NAMES = ('original', 'halfspace-a', 'cosine-b', 'near-far', 'moment-sphere', 'diffuse-sg', 'brdf')
REGIONS = ((750, 152), (800, 102), (1050, 152), (820, 252), (700, 502))
# Rectangles are fixed in display top-left coordinates, including dark receivers.
ROIS = (
    {'floor': (750, 800, 256, 128), 'wall': (50, 350, 150, 200), 'dark': (720, 200, 180, 120)},
    {'floor': (800, 850, 256, 128), 'wall': (1500, 140, 120, 160), 'dark': (380, 570, 200, 100)},
    {'floor': (1050, 800, 256, 128), 'wall': (1500, 300, 180, 140), 'dark': (720, 500, 160, 140)},
    {'near_light': (820, 700, 256, 128), 'wall': (1560, 350, 120, 180), 'dark': (1050, 460, 180, 100)},
    {'near_light': (700, 450, 256, 128), 'wall': (550, 260, 170, 150), 'dark': (180, 700, 200, 130)},
)


def image(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    return np.fromfile(path, '<f4').reshape(meta['height'], meta['width'], 4)


def valid(path, prefix, frames):
    count = image(path / (prefix + '.valid_count.bin'))
    bad = image(path / (prefix + '.nonfinite.bin'))
    incomplete = image(path / (prefix + '.incomplete_batches.bin'))
    return np.all(count[:, :, :3] == frames, axis=2) & np.all(bad[:, :, :3] == 0, axis=2) & np.all(incomplete[:, :, :3] == 0, axis=2)


def ratio(numerator, denominator):
    return [float(a / b) if b != 0 else None for a, b in zip(numerator, denominator)]


def main():
    report = {'revision': '1df099667436c935b8527d1a1cc2f8eab4bd1fea', 'scenes': [], 'timing': [], 'quality_captures': {'accepted': 35, 'actual': 49, 'historical_excluded': ['light-sampling-1/{sampler}', 'light-sampling-3/{sampler}']}, 'profile_cases': {'accepted': 70, 'actual': 140, 'historical_unbracketed': 'light-sampling-{1..5}/profile'}, 'reference_captures': {'accepted': 5, 'actual': 6, 'historical_excluded': 'light-sampling-3/reference'}, 'limitations': ['Reference window difference is empirical uncertainty, not proof of convergence.', 'Required cost is light-sampling GPU cost, not full frame time; BRDF initial-only is separated from a run that still executes diagnostic builds.', 'Variance times cost assumes 1/N scaling and ignores temporal correlation; it is not a measured equal-time image.']}
    ready = json.loads((RUN / 'source-ready.json').read_text())
    manifest = {'title': 'Light Sampling Weights at 1080p', 'images': []}
    for i, (x, y) in enumerate(REGIONS, 1):
        scene = f'light-sampling-{i}'
        p = RUN / scene
        quality_root = p / 'quality-verified' if i in (1, 3) else p
        ref = p / ('reference-verified' if i == 3 else 'reference')
        ref_images = [image(ref / f'ref{j}.mean.bin') for j in (1, 2)]
        refs = [a[:, :, :3] for a in ref_images]
        mask = valid(ref, 'ref1', 1024) & valid(ref, 'ref2', 1024)
        mask &= (ref_images[0][:, :, 3] > 0) & (ref_images[1][:, :, 3] > 0)
        refmean = (refs[0] + refs[1]) * .5
        mask &= np.isfinite(refmean).all(axis=2)
        quality = {}
        full_mask = np.ones((1080, 1920), dtype=bool)
        full_receivers = np.ones((1080, 1920), dtype=bool)
        input_stats = {}
        for name in NAMES:
            q = quality_root / name
            mean_full = image(q / 'mean32.mean.bin')
            full_receivers &= mean_full[:, :, 3] > 0
            complete = valid(q, 'mean32', 32) & np.isfinite(mean_full).all(axis=2) & (mean_full[:, :, 3] > 0)
            full_mask &= complete
            nonfinite = image(q / 'mean32.nonfinite.bin')
            coords = np.argwhere(np.any(nonfinite[:, :, :3] > 0, axis=2))
            counts = image(q / 'mean32.valid_count.bin')
            raw_single = image(q / 'single.bin')
            input_stats[name] = {'nonfinite_samples_per_channel': np.sum(nonfinite, axis=(0, 1)).tolist(), 'nonfinite_raw_xy_and_rgb_counts': [[int(xx), int(yy), *nonfinite[yy, xx, :3].tolist()] for yy, xx in coords], 'missing_rgb_samples': [int(np.maximum(32-counts[:, :, c], 0).sum()) for c in range(3)], 'single_display_nonfinite_components_replaced': int(np.count_nonzero(~np.isfinite(raw_single))), 'mean_display_nonfinite_components_replaced': int(np.count_nonzero(~np.isfinite(mean_full)))}
        mask &= full_mask[y:y+128, x:x+256]
        roi_stats = {}
        for name in NAMES:
            q = quality_root / name
            mean_full = image(q / 'mean32.mean.bin')[:, :, :3]
            var_full = image(q / 'mean32.variance.bin')[:, :, :3]
            mean = mean_full[y:y+128, x:x+256]
            var = var_full[y:y+128, x:x+256]
            diff = mean - refmean
            low = {}
            for sigma in (2, 12):
                weight = gaussian_filter(mask.astype(float), sigma=sigma)
                filtered = gaussian_filter(np.where(mask[:, :, None], diff, 0), sigma=(sigma, sigma, 0)) / np.maximum(weight[:, :, None], 1e-20)
                low[str(sigma)] = float(np.mean(filtered[mask] ** 2))
            lum = mean @ LUMA
            quality[name] = {'rgb_mse': float(np.mean(diff[mask] ** 2)), 'low_frequency_rgb_mse_by_sigma_pixels': low, 'mean_luma': float(np.mean(lum[mask])), 'rgb_sample_variance_mean': np.mean(var[mask], axis=0).tolist(), 'zero_luma_fraction': float(np.mean(lum[mask] == 0)), 'luma_p99_exploratory': float(np.quantile(lum[mask], .99))}
            full_v = np.mean(var_full[full_mask], axis=0)
            quality[name]['full_common_receiver'] = {'rgb_mean': np.mean(mean_full[full_mask], axis=0).tolist(), 'rgb_sample_variance_mean': full_v.tolist(), 'sum_rgb_sample_variance_mean': float(full_v.sum())}
            roi_stats[name] = {}
            for label, (rx, ry, rw, rh) in ROIS[i-1].items():
                bottom = 1080 - ry - rh
                m = full_mask[bottom:bottom+rh, rx:rx+rw]
                a = mean_full[bottom:bottom+rh, rx:rx+rw][m]
                v = var_full[bottom:bottom+rh, rx:rx+rw][m]
                roi_stats[name][label] = {'display_xywh': [rx, ry, rw, rh], 'raw_xywh': [rx, bottom, rw, rh], 'pixels': int(m.sum()), 'rgb_mean': np.mean(a, axis=0).tolist(), 'rgb_sample_variance_mean': np.mean(v, axis=0).tolist(), 'zero_rgb_fraction': float(np.mean(np.all(a == 0, axis=1)))}
            for label, filename in [('single', 'single.bin'), ('32mean', 'mean32.mean.bin')]:
                output = q / (label + '.png')
                values = image(q / filename)
                render(np.nan_to_num(values, nan=0, posinf=0, neginf=0), output, 4)
                manifest['images'].append({'path': str(output), 'dimensions': {'Scene': scene, 'Sampler': name, 'Accumulation': label}, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
        for name in NAMES:
            for label in roi_stats[name]:
                base = np.array(roi_stats['original'][label]['rgb_sample_variance_mean'])
                v = np.array(roi_stats[name][label]['rgb_sample_variance_mean'])
                roi_stats[name][label]['rgb_variance_ratio_to_original'] = ratio(v, base)
                roi_stats[name][label]['rgb_variance_ratio_numerator'] = v.tolist()
                roi_stats[name][label]['rgb_variance_ratio_denominator'] = base.tolist()
                base_mean = np.array(roi_stats['original'][label]['rgb_mean'])
                current_mean = np.array(roi_stats[name][label]['rgb_mean'])
                roi_stats[name][label]['rgb_mean_difference_from_original'] = (current_mean - base_mean).tolist()
        refdiff = refs[0] - refs[1]
        ref_mse = float(np.mean(refdiff[mask] ** 2))
        ref_low = {}
        for sigma in (2, 12):
            weight = gaussian_filter(mask.astype(float), sigma=sigma)
            filtered = gaussian_filter(np.where(mask[:, :, None], refdiff, 0), sigma=(sigma, sigma, 0)) / np.maximum(weight[:, :, None], 1e-20)
            ref_low[str(sigma)] = float(np.mean(filtered[mask] ** 2) / 4)
        report['scenes'].append({'scene': scene, 'region_raw_bottom_left': [x, y, 256, 128], 'region_display_top_left': [x, 1080-y-128, 256, 128], 'full_common_receiver_pixels': int(full_mask.sum()), 'full_receiver_pixels_before_completeness': int(full_receivers.sum()), 'full_receiver_pixels_excluded': int((full_receivers & ~full_mask).sum()), 'reference_common_pixels': int(mask.sum()), 'reference_excluded_pixels': int(mask.size-mask.sum()), 'reference_mean_luma': float(np.mean((refmean @ LUMA)[mask])), 'reference_window_difference_rgb_mse': ref_mse, 'reference_mean_empirical_error_rgb_mse': ref_mse / 4, 'reference_filtered_mean_error_by_sigma': ref_low, 'input_stats': input_stats, 'fixed_roi': roi_stats, 'quality': quality})
        identity = json.loads((p / 'profile-verified/identity.json').read_text())
        accepted = all(identity[k] for k in ('leaf_match', 'topology_match', 'voxel_match', 'post_no_load', 'post_matches_last_profile'))
        job = json.loads((p / 'profile-verified/profile.json').read_text()) if accepted else {'cases': []}
        assert not accepted or (job['success'] and len(job['cases']) == 14)
        expected_preset = ready['presets'][i-1]['preset_sha256']
        for order, case in enumerate(job['cases']):
            metrics = {m['metric_id']: m for m in case['metrics']['metrics'] if m.get('main_source')}
            build = [m for m in metrics.values() if m['main_source'] == '/pass/prepare/LightSamplingBuild.comp.glsl']
            initial = [m for m in metrics.values() if m['main_source'] == '/pass/composite/LightSamplingInitial.comp.glsl']
            assert len(build) == 14 and len(initial) == 1 and all(len(m['samples_ns']) == 128 for m in build + initial)
            b = sum(m['average_ns'] for m in build) / 1e6
            t = initial[0]['average_ns'] / 1e6
            siblings = [metrics[n] for n in ('prepare2_compute', 'prepare3_compute', 'composite42_compute', 'begin3_a_compute') if n in metrics]
            provenance = case['provenance']
            settings = {v['name']: v['value'] for v in provenance['effective_settings']['settings']}
            assert provenance['resolved_revision'] == report['revision'] and provenance['preset_id'] == scene and provenance['preset_sha256'] == expected_preset and not provenance['stale'] and not provenance['head_changed']
            assert settings['SETTING_DEBUG_LIGHT_AUDIT'] == 'false' and settings['SETTING_DEBUG_LIGHT_WEIGHT_MODEL'] == str(NAMES.index(case['config_id']) if case['config_id'] != 'brdf' else 0)
            assert settings['SETTING_DEBUG_LIGHT_SAMPLING'] == ('1' if case['config_id'] == 'brdf' else '2')
            report['timing'].append({'scene': scene, 'sampler': case['config_id'], 'order': order, 'provenance': provenance, 'build_avg_ms': b, 'initial_avg_ms': t, 'required_avg_ms': t + (0 if case['config_id'] == 'brdf' else b), 'shared_prepare_avg_ms': sum(m['average_ns'] for m in siblings if m['metric_id'] in ('prepare2_compute', 'prepare3_compute')) / 1e6, 'stages': {m['metric_id']: {'avg_ms': m['average_ns']/1e6, 'main_source': m['main_source'], 'samples_ns': m['samples_ns']} for m in build + initial + siblings}})
        paired = {}
        for name in NAMES:
            pair = [r for r in report['timing'] if r['scene'] == scene and r['sampler'] == name]
            full = quality[name]['full_common_receiver']
            variance = full['sum_rgb_sample_variance_mean']
            original_variance = quality['original']['full_common_receiver']['sum_rgb_sample_variance_mean']
            full['sum_rgb_variance_ratio_to_original'] = variance / original_variance if original_variance else None
            full['sum_rgb_variance_ratio_numerator'] = variance
            full['sum_rgb_variance_ratio_denominator'] = original_variance
            if not accepted:
                continue
            assert len(pair) == 2
            cost = sum(r['required_avg_ms'] for r in pair) / 2
            shared = sum(r['shared_prepare_avg_ms'] for r in pair) / 2
            sentinel = [r['stages']['begin3_a_compute']['avg_ms'] for r in pair]
            roi_proxy = {label: {'rgb_variance_times_required_cost': (np.array(data['rgb_sample_variance_mean']) * cost).tolist(), 'rgb_variance_times_cost_plus_shared_prepares': (np.array(data['rgb_sample_variance_mean']) * (cost + shared)).tolist()} for label, data in roi_stats[name].items()}
            paired[name] = {'required_avg_ms_pair_mean': cost, 'shared_prepare_avg_ms_pair_mean': shared, 'required_cost_forward_reverse_ms': [r['required_avg_ms'] for r in pair], 'sky_lut_sentinel_forward_reverse_ms': sentinel, 'full_sum_rgb_variance_times_required_cost': variance * cost, 'full_sum_rgb_variance_times_cost_plus_shared_prepares': variance * (cost + shared), 'fixed_roi_proxy': roi_proxy}
        report['scenes'][-1]['paired_cost_and_proxy'] = paired
        sentinel_values = [r['stages']['begin3_a_compute']['avg_ms'] for r in report['timing'] if r['scene'] == scene]
        report['scenes'][-1]['sentinel'] = {'metric': 'begin3_a_compute', 'main_source': '/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl', 'min_ms': min(sentinel_values), 'max_ms': max(sentinel_values), 'min_over_max': min(sentinel_values) / max(sentinel_values), 'interpretation': 'Measured workload sentinel varies across cases; do not normalize target timings by it or claim small differences are stable.'} if sentinel_values else None
        report['scenes'][-1]['audit'] = json.loads(((p / 'quality-verified/identity-audit.json') if i in (1, 3) else (p / 'identity-audit.json')).read_text())
        report['scenes'][-1]['reference_identity'] = json.loads((ref / 'identity.json').read_text())
        report['scenes'][-1]['profile_identity'] = json.loads((p / 'profile-verified/identity.json').read_text())
        if i in (1, 3):
            report['scenes'][-1]['quality_identity'] = json.loads((p / 'quality-verified/identity.json').read_text())
        baseline = quality_root / 'original/baseline-review.json'
        if baseline.exists():
            report['scenes'][-1]['baseline_review'] = json.loads(baseline.read_text())
    (RUN / 'comparison-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    diagnostic = json.loads((RUN / 'nonfinite-diagnostic/summary.json').read_text())
    report['nonfinite_diagnostic'] = {'dump_count': len(diagnostic), 'reason3_sum': sum(r['reason3'] for r in diagnostic), 'reason10_sum': sum(r['reason10'] for r in diagnostic), 'frames': diagnostic, 'interpretation': 'Five warmup-separated batches; reads are not independent and are not all consecutive. No reason-3 event was observed; reason-10 was observed. Exact nonfinite cause remains unresolved.'}
    report['comparison'] = {'manifest': 'comparison-manifest.json', 'html': 'comparison.html', 'assets': 'comparison.assets', 'images': len(manifest['images']), 'exposure_stops': 4, 'nonfinite_display_policy': 'Nonfinite components replaced with zero for previews only; counts are recorded per case; raw captures are unchanged.', 'browser_qa': 'Unavailable: browser-client default returned No browser is available and browser list was empty.'}
    report['independent_review'] = {'quality': 'parent-independent-review.json', 'timing': 'parent-independent-timing-review.json', 'scope': 'Parent independently reviewed the current 35 primary quality cases and 70 bracketed timing cases.'}
    (RUN / 'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
