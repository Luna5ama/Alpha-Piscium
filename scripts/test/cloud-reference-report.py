#!/usr/bin/env python3
"""Summarize independent CPU cloud-reference replicates in linear radiance.

python scripts/test/cloud-reference-report.py <reference-output-directory>
Requires numpy, scipy and matplotlib. Each light direction uses one linear scale;
all estimates and uncertainties use the original float64 radiance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]


def camera_geometry(manifest):
    height, width = manifest['shape'][3:5]
    camera = manifest['camera']
    xs = np.linspace(*camera['x'], width, endpoint=False) + np.diff(camera['x'])[0] / (2 * width)
    ys = np.linspace(*camera['y_top_bottom'], height, endpoint=False) + np.diff(camera['y_top_bottom'])[0] / (2 * height)
    x, y = np.meshgrid(xs, ys)
    thickness = np.zeros_like(x)
    for j in range(height):
        for i in range(width):
            segments = []
            for center, radius in manifest['ellipsoids_center_radius']:
                remaining = 1 - ((x[j, i] - center[0]) / radius[0]) ** 2 - ((y[j, i] - center[1]) / radius[1]) ** 2
                if remaining > 0:
                    half = np.sqrt(remaining) * radius[2]
                    segments.append((center[2] - half, center[2] + half))
            end = -np.inf
            for near, far in sorted(segments):
                thickness[j, i] += max(0, far - max(near, end))
                end = max(end, far)
    tau = thickness * manifest['extinction_per_metre']
    return x, y, tau


def stats(values):
    """Each entry is one independently seeded replicate's scalar estimate."""
    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    radius = float(t.ppf(.975, n-1) * se)
    return {'mean': mean, 'standard_error': se, 'confidence_95': [mean-radius, mean+radius],
            'relative_standard_error': se / abs(mean) if mean else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--compare', type=Path, help='independent earlier reference run with identical physical inputs')
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / 'reference.json').read_text(encoding='utf-8'))
    raw = np.fromfile(directory / 'radiance.bin', dtype='<f8').reshape(manifest['shape'])
    assert np.isfinite(raw).all() and np.min(raw[..., :2]) >= 0
    x, y, tau = camera_geometry(manifest)
    masks = {
        'cloud': tau > 0,
        'upper_thin_rim': (tau > 0) & (tau < 1.5) & (y > 100),
        'right_notch': (tau > 0) & (x >= 30) & (x <= 65) & (y >= 120) & (y <= 155),
        'core': (tau > 2.5) & (abs(x) < 25) & (y >= 70) & (y <= 160),
    }
    assert all(np.any(mask) for mask in masks.values())
    values = np.stack((raw[..., 0], raw[..., 1], raw[..., 0] + raw[..., 1]), axis=-1)
    per_path_var = np.stack((raw[..., 2], raw[..., 3], raw[..., 2] + raw[..., 3] + 2 * raw[..., 4]), axis=-1)
    assert np.min(per_path_var) >= -1e-9
    means = np.mean(values, axis=1)
    replicates = raw.shape[1]
    errors = np.std(values, axis=1, ddof=1) / np.sqrt(replicates)
    levels = np.asarray(manifest['spp_per_replicate'])
    report = {
        'scope': 'Continuous-angle all-order scattering for the frozen finite ellipsoid-union fixture; no real-time comparison yet.',
        'work_item': 'cloud-light-transport-comparison@d2f1ef0fe2b1ac9a962a062fe7d2af7f922611ba',
        'baseline': 'c09cfdc76e79cb3e4e986d16e29730777283e78d',
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {name: hashlib.sha256((ROOT / 'scripts/test' / name).read_bytes()).hexdigest()
                          for name in ('cloud-reference.cpp', 'cloud-reference-report.py')},
        'reference_manifest_sha256': hashlib.sha256((directory / 'reference.json').read_bytes()).hexdigest(),
        'radiance_sha256': hashlib.sha256((directory / 'radiance.bin').read_bytes()).hexdigest(),
        'independent_replicates': replicates,
        'pixels_per_region': {name: int(mask.sum()) for name, mask in masks.items()},
        'spp_per_replicate': levels.tolist(),
        'total_spp_per_level': (levels * replicates).tolist(),
        'views': {},
        'limitations': ['Pixel-centre orthographic rays; no pixel-area antialiasing.',
                        'Piecewise constant sample tower, not the production eroded procedural cloud.',
                        'Sun-only incident light; no sky, ground, air or neighboring tower yet.',
                        'Finite-sample uncertainty is measured; no claim of exact noise-free radiance.',
                        'No Iris compilation, image acceptance or GPU timing in this CPU stage.'],
    }
    for light, label in enumerate(('side', 'back')):
        view = {'regions': {}, 'rms_relative_noise': {}}
        for component, name in enumerate(('single', 'multiple', 'total')):
            mean = means[light, :, :, :, component]
            se = errors[light, :, :, :, component]
            rms_noise = np.sqrt(np.mean(se[:, masks['cloud']] ** 2, axis=1))
            rms_signal = np.sqrt(np.mean(mean[:, masks['cloud']] ** 2, axis=1))
            view['rms_relative_noise'][name] = (rms_noise / rms_signal).tolist()
        for region, mask in masks.items():
            result = {}
            for component, name in enumerate(('single', 'multiple', 'total')):
                # Pixel streams are independent, but uncertainty is measured across complete replicates.
                estimates = values[light, :, :, :, :, component][:, :, mask].mean(axis=-1)
                result[name] = [stats(estimates[:, i]) for i in range(len(levels))]
                previous = estimates[:, -2]
                tail = (estimates[:, -1] * levels[-1] - previous * levels[-2]) / (levels[-1] - levels[-2])
                difference = stats(tail - previous)
                result[name + '_last_disjoint_block_change'] = difference
            view['regions'][region] = result
        total = values[light, :, -1, :, :, 2]
        half = replicates // 2
        split_difference = total[:half].mean(axis=0) - total[half:].mean(axis=0)
        view['independent_half_difference_rms'] = float(np.sqrt(np.mean(split_difference[masks['cloud']] ** 2)))
        view['independent_half_difference_relative_rms'] = float(
            np.sqrt(np.mean(split_difference[masks['cloud']] ** 2)) / np.sqrt(np.mean(means[light, -1, :, :, 2][masks['cloud']] ** 2)))
        view['final_pixel_relative_standard_error_percentiles'] = np.percentile(
            errors[light, -1, :, :, 2][masks['cloud']] / means[light, -1, :, :, 2][masks['cloud']], [50, 90, 99]).tolist()
        view['replicate_vs_per_path_variance_ratio'] = float(
            np.sum(errors[light, -1, :, :, 2][masks['cloud']] ** 2) /
            np.sum(per_path_var[light, :, -1, :, :, 2].mean(axis=0)[masks['cloud']] / replicates))
        report['views'][label] = view
    if args.compare:
        earlier_manifest = json.loads((args.compare / 'reference.json').read_text(encoding='utf-8'))
        physical_keys = ('fixture', 'camera', 'light_directions_to_sun', 'sun_angular_radius_degrees',
                         'sun_normal_irradiance', 'sky_radiance', 'ground', 'density_inside_union',
                         'extinction_per_metre', 'single_scattering_albedo', 'phase', 'ellipsoids_center_radius')
        assert all(manifest[key] == earlier_manifest[key] for key in physical_keys), 'comparison physical inputs differ'
        earlier = np.fromfile(args.compare / 'radiance.bin', dtype='<f8').reshape(earlier_manifest['shape'])
        assert earlier.shape[:2] == raw.shape[:2] and earlier.shape[3:] == raw.shape[3:]
        compare = {'earlier_manifest_sha256': hashlib.sha256((args.compare / 'reference.json').read_bytes()).hexdigest(),
                   'earlier_spp_per_replicate': earlier_manifest['spp_per_replicate'][-1],
                   'current_spp_per_replicate': int(levels[-1]),
                   'meaning': 'Current minus earlier estimate; paired independent replicate differences, not a noise-free truth test.',
                   'views': {}}
        for light, label in enumerate(('side', 'back')):
            compare['views'][label] = {}
            for region, mask in masks.items():
                result = {}
                for component, name in enumerate(('single', 'multiple', 'total')):
                    old = earlier[light, :, -1, :, :, component] if component < 2 else earlier[light, :, -1, :, :, :2].sum(axis=-1)
                    difference = values[light, :, -1, :, :, component][:, mask].mean(axis=-1) - old[:, mask].mean(axis=-1)
                    result[name] = stats(difference)
                compare['views'][label][region] = result
        report['comparison'] = compare
    (directory / 'convergence.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    fig, axes = plt.subplots(2, 3, figsize=(11, 8), constrained_layout=True)
    # Different sun directions span five orders of magnitude. Keep each row's
    # single/multiple/total scale identical and label the actual radiance units.
    for light, label in enumerate(('Side light', 'Backlight')):
        vmax = float(np.max(means[light, -1, :, :, 2]))
        for column, (title, value) in enumerate((('Single', means[light, -1, :, :, 0]),
                                                ('Multiple', means[light, -1, :, :, 1]),
                                                ('Total', means[light, -1, :, :, 2]))):
            im = axes[light, column].imshow(value, cmap='gray', vmin=0, vmax=vmax, interpolation='nearest',
                                           extent=[-100, 100, -10, 240])
            axes[light, column].set_title(f'{label}: {title}')
            axes[light, column].set_xlabel('x / m')
            if column == 0:
                axes[light, column].set_ylabel('y / m')
        fig.colorbar(im, ax=axes[light, :], label='Linear L / E_sun [sr^-1]', shrink=.8)
    fig.suptitle(f'Fixed tower and camera | {levels[-1] * replicates:,} paths/pixel | linear scale per row')
    fig.savefig(directory / 'reference.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for light, label in enumerate(('side', 'back')):
        for column, region in enumerate(('upper_thin_rim', 'right_notch', 'core')):
            for component, color in (('single', 'tab:orange'), ('multiple', 'tab:blue'), ('total', 'black')):
                results = report['views'][label]['regions'][region][component]
                m = np.array([v['mean'] for v in results])
                radius = np.array([v['confidence_95'][1]-v['mean'] for v in results])
                axes[light, column].errorbar(levels * replicates, m, yerr=radius, color=color, label=component, marker='o', capsize=3)
            axes[light, column].set_xscale('log', base=4)
            axes[light, column].set_title(f'{label} / {region}')
            axes[light, column].set_xlabel('paths per pixel')
            axes[light, column].set_ylabel('mean linear L / E_sun')
            axes[light, column].grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle(f'Independent replicate means and 95% Student-t intervals ({replicates} streams)')
    fig.savefig(directory / 'convergence.png', dpi=140)
    plt.close(fig)
    print(json.dumps({
        'regions': report['pixels_per_region'],
        'total_spp': report['total_spp_per_level'],
        'views': {label: {'rms_relative_noise': view['rms_relative_noise'],
                          'final_regions': {region: result['total'][-1] for region, result in view['regions'].items()}}
                  for label, view in report['views'].items()}}, indent=2))


if __name__ == '__main__':
    main()
