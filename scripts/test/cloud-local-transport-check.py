#!/usr/bin/env python3
"""Frozen-source local cloud transport and control-variate reference checks.

These are numerical model tests, not GPU image or global power validation.
Requires numpy and scipy; run from scripts/ with --output PATH.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
TABLE = np.fromfile(ROOT / 'shaders/textures/cloud_transport.bin', dtype='<f4').reshape(26, 1024, 4).astype(float)
PHASE = .1 * TABLE[2, :576, :3].reshape(24, 24, 3) + .9 * TABLE[3, :576, :3].reshape(24, 24, 3)
DIRS = np.array([(np.ones(3) + (np.arange(3) == a // 8)) *
                 (1 - 2 * ((a >> np.arange(3)) & 1)) / np.sqrt(6) for a in range(24)])


def equilibrium():
    rng = np.random.default_rng(842)
    tau = rng.uniform(0, 15, (512, 19, 3))
    tau[rng.random(tau.shape) < .25] = 0
    c = np.array([.15, 1.7, 4.2])
    t = np.ones((512, 3))
    local = np.zeros_like(t)
    for segment in tau.transpose(1, 0, 2):
        local += t * -np.expm1(-segment) * c
        t *= np.exp(-segment)
    local += t * c
    error = np.max(abs(local - c))
    assert error < 4e-15
    # Pure attenuation would incorrectly lose almost all of this equilibrium field.
    ao_error = np.max(abs(t * c - c))
    assert ao_error > 4
    epsilon = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    second_order = -np.expm1(-epsilon * .7) * -np.expm1(-epsilon * 1.3)
    exponent = np.log(second_order[:-1] / second_order[1:]) / np.log(10)
    assert np.min(exponent) > 1.99
    signed = np.array([-2., -.0625, 3.]).astype(np.float16).astype(float)
    assert np.array_equal(signed, [-2., -.0625, 3.])
    return {'maximum_equilibrium_error': float(error), 'attenuation_only_error': float(ao_error),
            'thin_second_order_exponents': exponent.tolist(), 'signed_half_roundtrip': signed.tolist()}


def reservoir():
    rng = np.random.default_rng(149)
    depths, trials = 37, 160000
    # Integrate real spectral camera weights, foreground air, and an opaque tail.
    tau = rng.uniform(.01, .2, (depths, 3))
    transmittance = np.exp(-np.cumsum(tau, axis=0))
    opacity = np.vstack((np.ones(3), transmittance[:-1])) * -np.expm1(-tau)
    opacity[-1] += transmittance[-1]
    weights = opacity * rng.uniform(.2, 1, (depths, 3)) * np.array([.99, .98, .97])
    angular = rng.dirichlet(np.ones(4))
    row = np.einsum('a,ajc->jc', angular, PHASE[[3, 7, 11, 19]])
    # The baseline P S and sampled S use identical spatial interpolation.
    spatial = rng.dirichlet(np.ones(8), depths)
    cache_corners = rng.uniform(.05, 3, (depths, 8, 24, 3))
    scattered = np.einsum('di,dijc->djc', spatial, cache_corners)
    cached_source = np.einsum('di,jc,dijc->dc', spatial, row, cache_corners)
    consistency = np.max(abs(cached_source - np.einsum('jc,djc->dc', row, scattered)))
    assert consistency < 2e-15
    local = scattered * rng.uniform(.05, 1.9, scattered.shape)
    residual = local - scattered
    base = np.sum(weights * cached_source, axis=0)
    target = np.einsum('dc,jc,djc->c', weights, row, local)
    # Stream depths exactly as WRS, with independent random dimensions.
    q = abs(weights).max(axis=1)
    total = 0.
    selected = np.zeros(trials, dtype=int)
    for i in range(depths):
        total += q[i]
        selected[rng.random(trials) * total < q[i]] = i
    proposal = row.mean(axis=1)
    angular_total = 0.
    angles = np.zeros(trials, dtype=int)
    for j in range(24):
        angular_total += proposal[j]
        angles[rng.random(trials) * angular_total < proposal[j]] = j
    estimate = base + (total / q[selected])[:, None] * weights[selected] * (
        angular_total / proposal[angles])[:, None] * row[angles] * residual[selected, angles]
    mean = estimate.mean(axis=0)
    stderr = estimate.std(axis=0, ddof=1) / np.sqrt(trials)
    z = abs(mean - target) / stderr
    assert z.max() < 5
    # Exact enumeration also checks the PDF factors without a statistical tolerance.
    expectation = base + np.einsum('d,j,dc,jc,djc->c', q / total, proposal / angular_total,
                                   weights * (total / q)[:, None],
                                   row * (angular_total / proposal)[:, None], residual)
    assert np.max(abs(expectation - target)) < 1e-13
    paired = (estimate[::2] + estimate[1::2]) * .5
    variance_ratio = paired.var(axis=0) / estimate.var(axis=0)
    assert np.all((variance_ratio > .48) & (variance_ratio < .52))
    assert estimate.min() < 0
    return {'trials': trials, 'predictor_consistency_error': float(consistency), 'reference': target.tolist(),
            'sample_mean': mean.tolist(), 'standard_errors_from_reference': z.tolist(),
            'two_sample_variance_ratio': variance_ratio.tolist(), 'minimum_signed_estimate': float(estimate.min()),
            'zero_clamp_bias': (np.maximum(estimate, 0).mean(axis=0) - mean).tolist()}


def geometry():
    axis = -DIRS[23]
    tangent = np.cross(axis, [0, 1, 0]); tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    def unscaled_density(x, offset):
        z, u, v = x @ axis, x @ tangent, x @ bitangent
        return 24 * np.exp(-((z - .18) / .085)**2) * (
            1 - .985 * np.exp(-((u - offset)**2 + v*v) / .0025))
    cube = np.stack(np.meshgrid(*([np.linspace(-.5, .5, 48)] * 3), indexing='ij'), axis=-1)
    means = [unscaled_density(cube, offset).mean() for offset in (0., .18)]
    scales = [1., means[0] / means[1]]
    def density(x, arrangement):
        return unscaled_density(x, (0., .18)[arrangement]) * scales[arrangement]
    def source(x):
        return .15 + 1.3 * np.exp(-((x @ axis - .3) / .08)**2)
    radius = .6
    def reference(direction, arrangement):
        def rhs(s, y):
            x = -direction * s
            rho = density(x, arrangement)
            return [-rho * y[0], rho * y[0] * source(x)]
        sol = solve_ivp(rhs, (0., radius), [1., 0.], method='DOP853', rtol=1e-11, atol=1e-13, max_step=.005)
        assert sol.success
        return sol.y[1, -1] + sol.y[0, -1] * 2.
    ref = np.array([[reference(direction, arrangement) for direction in DIRS] for arrangement in range(2)])
    # Frozen positive source; ray-bin estimates are compared with the continuous ODE.
    errors = []
    for steps in (4, 8, 16, 32):
        edges = np.expm1(np.linspace(0, np.log(16), steps + 1)) * (radius / 15)
        values = []
        for arrangement in range(2):
            jitter = (np.arange(128) + .5) / 128
            t = np.ones((len(jitter), 24)); result = np.zeros_like(t)
            for start, end in zip(edges[:-1], edges[1:]):
                s = start + (end - start) * jitter
                pos = -s[:, None, None] * DIRS[None, :, :]
                tau = density(pos, arrangement) * (end - start)
                result += t * -np.expm1(-tau) * source(pos)
                t *= np.exp(-tau)
            values.append((result + 2*t).mean(axis=0))
        errors.append(float(np.sqrt(np.mean((np.array(values) - ref)**2))))
    assert errors[-1] < errors[0] * .15, errors
    assert abs(ref[0, 23] - ref[1, 23]) > .5
    coarse_means = [density(cube, arrangement).mean() for arrangement in range(2)]
    assert abs(coarse_means[0] - coarse_means[1]) < 1e-13
    return {'same_coarse_density': list(map(float, coarse_means)), 'channel_radiance': ref[:, 23].tolist(),
            'steps': [4, 8, 16, 32], 'directional_rms_error_against_ODE': errors,
            'scope': 'Same coarse cell mass, displaced narrow channel, positive frozen source; not a converged full cloud solve.'}


def gpu_history(path):
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert metadata['internal_format'] == 'RGBA32UI' and not metadata['y_flipped']
    width, height = metadata['width'], metadata['height']
    assert width % 2 == 0 and height % 3 == 0
    raw = np.fromfile(path, dtype='<u4').reshape(height, width, 4)
    # Frame-end history occupies tile (0,0); GI has reused the transient cloud tiles.
    packed = raw[:height//3, :width//2].copy()
    valid = np.any(packed != 0, axis=-1)
    history = packed.view('<f2').reshape(height//3, width//2, 8).astype(float)[valid]
    assert history.size and np.isfinite(history).all()
    assert np.all((history[:, 3:6] >= 0) & (history[:, 3:6] <= 1))
    assert np.all(history[:, 6] >= 0)
    assert np.all((history[:, 7] >= 0) & (history[:, 7] <= 120))
    return {'valid_pixels': int(valid.sum()), 'finite': True,
            'negative_scattering_channels': int(np.sum(history[:, :3] < 0)),
            'scattering_min': history[:, :3].min(axis=0).tolist(),
            'scattering_max': history[:, :3].max(axis=0).tolist(),
            'history_length_range': [float(history[:, 6].min()), float(history[:, 6].max())],
            'cloud_depth_km_range': [float(history[:, 7].min()), float(history[:, 7].max())],
            'scope': 'Final accumulated GPU history, not an unbiasedness test of the raw camera estimator.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-history', type=Path, help='Frame-end RGBA32UI atlas dump with its adjacent JSON metadata')
    args = parser.parse_args()
    report = gpu_history(args.gpu_history) if args.gpu_history else {
              'equilibrium': equilibrium(), 'spectral_reservoir': reservoir(), 'fine_geometry': geometry(),
              'not_established': ['Full transport unbiasedness or global power balance',
                                  'Angular convergence', 'Unbiased temporal filtering', 'Photo agreement']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
