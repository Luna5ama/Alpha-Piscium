"""Recompute the accepted 2026-10-31 near/far experiment from preserved native data."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'builds/light-sampling/20261031-nearfar-iteration'
METHODS = ('a', 'nearfar', 'candidate6')
# Frozen raw bottom-left xywh, chosen from user pixels or surface identity, not error.
REGIONS = (
    (1, 'point1', (1384, 534, 32, 32), 3328, 33153, 'pillar'),
    (1, 'point2', (104, 684, 32, 32), 10938, 33153, 'wall'),
    (1, 'floor', (750, 152, 256, 128), None, None, 'floor'),
    (2, 'floor', (800, 102, 256, 128), None, None, 'floor'),
    (4, 'birch_leaves', (400, 380, 32, 32), 832, 32512, 'birch_leaves'),
    (5, 'wall', (688, 500, 32, 32), 9965, 33151, 'wall'),
    (5, 'near_light', (700, 502, 256, 128), None, None, 'guard'),
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def image(path):
    meta = read(path.with_suffix('.json'))
    assert meta['scalar_type'] == 'FLOAT32' and meta['channel_count'] == 4
    return np.memmap(path, '<f4', mode='r', shape=(meta['height'], meta['width'], 4))


def quality(base, scene, method):
    prefix = 'ls4-final32' if scene == 4 else f'ls{scene}'
    return base / f'{prefix}-{method}-quality'


def receiver(folder, names, region):
    x, y, w, h = region
    sl = np.s_[y:y+h, x:x+w]
    g = np.memmap(folder / names[0], '<u4', mode='r', shape=(1080, 1920, 4))[sl]
    a = np.memmap(folder / names[1], 'u1', mode='r', shape=(1080, 1920, 4))[sl]
    z = np.memmap(folder / names[2], '<f4', mode='r', shape=(1080, 1920))[sl]
    return g[:, :, 0] & 65535, g[:, :, 3] >> 16, a[:, :, :3], z


def same_receiver(a, b):
    return (a[0] == b[0]) & (a[1] == b[1]) & np.all(a[2] == b[2], axis=2) & (a[3] == b[3])


def complete(folder, prefix, frames, sub):
    return np.all(image(folder / f'{prefix}.valid_count.bin')[sub] == frames, axis=2) & np.all(
        image(folder / f'{prefix}.nonfinite.bin')[sub] == 0, axis=2) & np.all(
        image(folder / f'{prefix}.incomplete_batches.bin')[sub] == 0, axis=2)


def analyze(run):
    base = run / 'candidate-runtime-evidence'
    output = dict(scope='Linear RGB, common complete actual receivers; variance is sum of channel sample variances, not luminance variance.', regions=[], timings=[])
    for scene, label, region, material, normal, refname in REGIONS:
        x, y, w, h = region
        sub = np.s_[y:y+h, x:x+w]
        q0 = receiver(quality(base, scene, 'a'), ('colortex8.bin', 'colortex9.bin', 'colortex10.bin'), region)
        mask = np.ones((h, w), bool)
        if material is not None:
            mask &= (q0[0] == normal) & (q0[1] == material)
        values = {}
        for method in METHODS:
            folder = quality(base, scene, method)
            acc = read(folder / 'quality.mean.json')['accumulation']
            assert acc['frames'] == 32 and acc['batches'] == 8 and acc['batch_lengths'] == [4]*8
            assert acc['end_frame'] - acc['start_frame'] == 31
            g = receiver(folder, ('colortex8.bin', 'colortex9.bin', 'colortex10.bin'), region)
            mask &= same_receiver(q0, g) & complete(folder, 'quality', 32, sub)
            mean = image(folder / 'quality.mean.bin')[sub].astype(float)
            variance = image(folder / 'quality.variance.bin')[sub].astype(float)
            mask &= np.isfinite(mean).all(axis=2) & np.isfinite(variance).all(axis=2) & (mean[:, :, 3] > 0)
            single_name = 'single.bin' if scene == 4 else 'iris_custom_image_uimg_lightSampling.bin'
            values[method] = mean, variance, image(folder / single_name)[sub].astype(float)
        ref_folder = base / ({1: 'ls1-verified-reference', 2: 'ls2-verified-reference', 4: 'ls4-final-reference', 5: 'ls5-new-reference'}[scene])
        names = ('before-colortex8.bin', 'before-colortex9.bin', 'before-colortex10.bin') if scene == 4 else ('gbuffer1.bin', 'gbuffer2.bin', 'depth.bin')
        refmask = mask & same_receiver(q0, receiver(ref_folder, names, region))
        if scene == 4:
            refmask &= same_receiver(q0, receiver(ref_folder, ('after-colortex8.bin', 'after-colortex9.bin', 'after-colortex10.bin'), region))
        refs, windows = [], []
        for window in (1, 2):
            prefix = f'{refname}-ref{window}'
            meta = read(ref_folder / f'{prefix}.mean.json')
            r = meta['region']
            assert r['x'] <= x and r['y'] <= y and x+w <= r['x']+r['width'] and y+h <= r['y']+r['height']
            crop = np.s_[y-r['y']:y-r['y']+h, x-r['x']:x-r['x']+w]
            acc = meta['accumulation']
            assert acc['frames'] == 1024 and acc['end_frame']-acc['start_frame'] == 1023
            windows.append((acc['start_frame'], acc['end_frame']))
            im = image(ref_folder / f'{prefix}.mean.bin')[crop].astype(float)
            refmask &= complete(ref_folder, prefix, 1024, crop) & np.isfinite(im).all(axis=2) & (im[:, :, 3] > 0)
            refs.append(im[:, :, :3])
        assert windows[0][1] < windows[1][0] and refmask.any() and mask.any()
        target = (refs[0] + refs[1]) * .5
        row = dict(scene=scene, label=label, raw_xywh=list(region), material=material, normal_oct16=normal,
                   common_pixels=int(mask.sum()), excluded_pixels=int(mask.size-mask.sum()),
                   reference=dict(complete_pixels=int(refmask.sum()), window_difference_mse_over_4=float(np.mean((refs[0][refmask]-refs[1][refmask])**2)/4)), methods={})
        for method, (mean, variance, single) in values.items():
            v = variance[:, :, :3][mask].mean(axis=0)
            finite = mask & np.isfinite(single).all(axis=2)
            difference = mean[:, :, :3] - target
            m = dict(rgb_mean=mean[:, :, :3][mask].mean(axis=0).tolist(), rgb_variance=v.tolist(), variance_sum=float(v.sum()),
                     zero_rgb_fraction=float(np.mean(np.all(mean[:, :, :3][mask] == 0, axis=1))),
                     single_rgb_mean=single[:, :, :3][finite].mean(axis=0).tolist(), single_rgb_p99=float(np.quantile(single[:, :, :3][finite], .99)),
                     reference_mse=float(np.mean(difference[refmask]**2)), absolute_error_p95=float(np.quantile(np.abs(difference[refmask]), .95)))
            for sigma in (2, 12):
                weight = gaussian_filter(refmask.astype(float), sigma)
                filtered = gaussian_filter(np.where(refmask[:, :, None], difference, 0), (sigma, sigma, 0)) / np.maximum(weight[:, :, None], 1e-20)
                m[f'low_frequency_mse_sigma{sigma}'] = float(np.mean(filtered[refmask]**2))
            row['methods'][method] = m
        output['regions'].append(row)
    for scene in (1, 2, 4, 5):
        for order, method in enumerate(('a', 'nearfar', 'candidate6', 'candidate6', 'nearfar', 'a'), 1):
            prefix = 'ls4-final' if scene == 4 else f'ls{scene}'
            raw = read(base / f'{prefix}-profile-{order:02d}-{method}' / 'job-result.json')
            raw = raw.get('result', raw)
            if scene == 4:
                assert next(x['wait_frames'] for x in raw['action_receipts'] if 'wait_frames' in x)['completed_frames'] == 128
                measured = next(x['gpu_metrics'] for x in raw['action_receipts'] if 'gpu_metrics' in x)
            else:
                c = raw['cases'][0]
                assert c['frames'] == c['warmup_frames'] == 128
                measured = c['metrics']
            assert measured['sampled_frames'] == 128
            metrics = [m for m in measured['metrics'] if m.get('main_source')]
            for m in metrics:
                assert len(m['samples_ns']) == 128 and abs(np.mean(m['samples_ns'])-m['average_ns']) <= 1
            build = [m for m in metrics if m['main_source'] == '/pass/prepare/LightSamplingBuild.comp.glsl']
            initial = [m for m in metrics if m['main_source'] == '/pass/composite/LightSamplingInitial.comp.glsl']
            sentinel = [m for m in metrics if m['main_source'] == '/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl']
            siblings = [m for m in metrics if m['metric_id'] in ('prepare2_compute', 'prepare3_compute')]
            assert len(build) == 14 and len(initial) == len(sentinel) == 1 and len(siblings) == 2
            output['timings'].append(dict(scene=scene, order=order, method=method, build_stages=14,
                build_avg_ms=sum(m['average_ns'] for m in build)/1e6, initial_avg_ms=initial[0]['average_ns']/1e6,
                sibling_avg_ms={m['metric_id']:m['average_ns']/1e6 for m in siblings}, sky_lut_avg_ms=sentinel[0]['average_ns']/1e6))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / 'recomputed-analysis.json'
    result = analyze(args.run_root)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(regions=len(result['regions']), timings=len(result['timings']), output=str(output))))
