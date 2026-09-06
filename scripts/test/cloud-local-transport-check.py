#!/usr/bin/env python3
"""Local cloud random walks, surface integration, and control-variate checks.

These are numerical model tests, not GPU image or global power validation.
Requires numpy and scipy; run from scripts/ with --output PATH.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import erf

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
    # The fine angular operator can differ from the coarse P matrix. Store and
    # subtract the entire per-depth camera predictor, rather than sampled S_c.
    fine_row = rng.uniform(.01, 1, (24, 3))
    fine_row /= fine_row.sum(axis=0)
    single_scattering = rng.uniform(.1, .8, (depths, 3))
    base = np.sum(weights * cached_source, axis=0)
    target = np.einsum('dc,jc,djc->c', weights, fine_row, local) + np.sum(weights * single_scattering, axis=0)
    # Stream depths exactly as WRS, with independent random dimensions.
    q = abs(weights).max(axis=1)
    total = 0.
    selected = np.zeros(trials, dtype=int)
    for i in range(depths):
        total += q[i]
        selected[rng.random(trials) * total < q[i]] = i
    proposal = fine_row.mean(axis=1)
    angular_total = 0.
    angles = np.zeros(trials, dtype=int)
    for j in range(24):
        angular_total += proposal[j]
        angles[rng.random(trials) * angular_total < proposal[j]] = j
    estimate = base + (total / q[selected])[:, None] * weights[selected] * (
        (angular_total / proposal[angles])[:, None] * fine_row[angles] * local[selected, angles]
        - cached_source[selected] + single_scattering[selected])
    mean = estimate.mean(axis=0)
    stderr = estimate.std(axis=0, ddof=1) / np.sqrt(trials)
    z = abs(mean - target) / stderr
    assert z.max() < 5
    # Exact enumeration also checks the PDF factors without a statistical tolerance.
    fine_integral = np.einsum('jc,djc->dc', fine_row, local)
    expectation = base + np.sum(weights * (fine_integral - cached_source + single_scattering), axis=0)
    cancellation = np.max(abs(base - np.sum(weights * cached_source, axis=0)))
    assert cancellation < 1e-13
    assert np.max(abs(expectation - target)) < 1e-13
    paired = (estimate[::2] + estimate[1::2]) * .5
    variance_ratio = paired.var(axis=0) / estimate.var(axis=0)
    assert np.all((variance_ratio > .48) & (variance_ratio < .52))
    assert estimate.min() < 0
    return {'trials': trials, 'predictor_consistency_error': float(consistency), 'stored_predictor_cancellation_error': float(cancellation), 'reference': target.tolist(),
            'sample_mean': mean.tolist(), 'standard_errors_from_reference': z.tolist(),
            'two_sample_variance_ratio': variance_ratio.tolist(), 'minimum_signed_estimate': float(estimate.min()),
            'zero_clamp_bias': (np.maximum(estimate, 0).mean(axis=0) - mean).tolist()}


def flight_distribution(direction, density, source, radius, bound, extinction, scattering, endpoint, steps=16):
    reference = extinction.mean()
    scale = .25 / (reference * bound)
    edges = scale * np.expm1(np.linspace(0, np.log1p(radius/scale), steps+1))
    pos = -.5*(edges[:-1]+edges[1:])[:, None]*direction
    columns = density(pos)*np.diff(edges)
    prefix = np.r_[0., np.cumsum(columns)]
    probabilities = np.r_[np.exp(-reference*prefix[:-1])*(-np.expm1(-reference*columns)), np.exp(-reference*prefix[-1])]
    integrated = np.exp(-extinction*prefix[:-1, None])*(-np.expm1(-extinction*columns[:, None]))*(scattering/extinction)*source(pos)[:, None]
    integrated = np.vstack([integrated, np.exp(-extinction*prefix[-1])*endpoint])
    return probabilities, integrated


def local_update(rng, count, direction, density, source, radius, bound, extinction, scattering, endpoint, steps=16):
    probabilities, integrated = flight_distribution(direction, density, source, radius, bound, extinction, scattering, endpoint, steps)
    cdf = np.cumsum(probabilities); cdf[-1] = 1.
    selected = np.searchsorted(cdf, rng.random(count), side='right')
    return integrated[selected]/probabilities[selected, None]


def spectral_segments():
    rng = np.random.default_rng(1129)
    extinction = np.array([.9, 1., 1.1])
    scattering = np.array([.88, .96, 1.05])
    density = lambda pos: .5+.8*(-pos[:, 0])
    constant = lambda pos: np.full(len(pos), 1.7)
    endpoint = np.array([.2, .4, .8])
    direction = np.array([1., 0., 0.])
    count = 200000
    samples = local_update(rng, count, direction, density, constant, 1.2, 1.46, extinction, scattering, endpoint)
    _, integrated = flight_distribution(direction, density, constant, 1.2, 1.46, extinction, scattering, endpoint)
    exact = integrated.sum(axis=0)
    column = .5*1.2+.4*1.2**2
    tr = np.exp(-extinction*column)
    analytic = tr*endpoint+(1-tr)*(scattering/extinction)*1.7
    # Midpoints integrate this nonuniform density exactly, independent of the event sampler.
    assert np.max(abs(exact-analytic)) < 1e-14
    z = abs(samples.mean(axis=0)-analytic)/(samples.std(axis=0, ddof=1)/np.sqrt(count))
    assert z.max() < 5, z
    steady = local_update(rng, 10000, direction, density, constant, 1.2, 1.46, np.ones(3), np.ones(3), np.full(3, 1.7))
    vacuum = local_update(rng, 10000, direction, lambda p: np.zeros(len(p)), constant, 1.2, 1.46, extinction, scattering, endpoint)
    assert np.max(abs(steady-1.7)) < 1e-14 and np.max(abs(vacuum-endpoint)) == 0
    return {'trials': count, 'analytic': analytic.tolist(), 'mean': samples.mean(axis=0).tolist(),
            'standard_errors_from_reference': z.tolist(), 'constant_equilibrium_error': float(np.max(abs(steady-1.7))),
            'vacuum_error': 0., 'scope': 'Spectral segment-source selection against a nonuniform analytic slab.'}


def angular_guiding():
    # HG convolution supplies an analytic integral of two sharp phase lobes.
    rng = np.random.default_rng(318)
    count = 300000
    g = np.array([.82, .86, .9])
    outgoing = np.array([0., 0., 1.])
    beam = np.array([.8, 0., .6])
    def hg(mu):
        return (1-g*g) / (4*np.pi*(1+g*g-2*g*mu[:, None])**1.5)
    guided = rng.random(count) < .0625
    axis = np.where(guided[:, None], beam, outgoing)
    tangent = np.cross(axis, [0., 1., 0.])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    bitangent = np.cross(axis, tangent)
    selected_g = g[rng.integers(0, 3, count)]
    xi = rng.random(count)
    mu = (1+selected_g**2 - ((1-selected_g**2)/(1-selected_g+2*selected_g*xi))**2)/(2*selected_g)
    phi = rng.random(count) * 2*np.pi
    direction = axis * mu[:, None] + np.sqrt(1-mu*mu)[:, None] * (tangent*np.cos(phi)[:, None] + bitangent*np.sin(phi)[:, None])
    phase = hg(direction @ outgoing)
    light_phase = hg(direction @ beam)
    proposal = (.9375*phase + .0625*light_phase).mean(axis=1)
    samples = phase*light_phase / proposal[:, None]
    exact = (1-g**4) / (4*np.pi*(1+g**4-2*g*g*.6)**1.5)
    z = abs(samples.mean(axis=0)-exact) / (samples.std(axis=0, ddof=1)/np.sqrt(count))
    assert z.max() < 5, z
    return {'trials': count, 'analytic_convolution': exact.tolist(), 'means': samples.mean(axis=0).tolist(),
            'standard_errors_from_reference': z.tolist(), 'beam_proposal_probability': .0625,
            'scope': 'RGB mixture PDF and source weighting; this does not establish angular LUT convergence.'}


def scattering_orders():
    # Two-stream slab has an independent, analytic transport solution.
    # Cache=0 makes the truncated local walk a partial sum of positive scattering orders.
    sigma, albedo, length, x0 = 1.3, .9, 3., .3
    phase = np.array([[.8, .2], [.2, .8]])
    direction = np.array([1., -1.])
    matrix = np.zeros((3, 3))
    matrix[:2, :2] = direction[:, None] * (sigma * albedo * phase - sigma * np.eye(2))
    matrix[:2, 2] = direction * sigma * albedo * phase[:, 0]
    matrix[2, 2] = -sigma
    propagation = expm(matrix * length)
    initial = np.array([0., -propagation[1, 2] / propagation[1, 1], 1.])
    exact = (expm(matrix * x0) @ initial)[1]
    count = 200000
    estimates, errors = [], []
    for bounces in (1, 4, 8, 16):
        rng = np.random.default_rng(910)
        x = np.full(count, x0)
        angle = np.ones(count, dtype=int)
        throughput = np.ones(count)
        result = np.zeros(count)
        active = np.arange(count)
        for bounce in range(bounces):
            x[active] -= direction[angle[active]] * -np.log1p(-rng.random(len(active))) / sigma
            active = active[(x[active] >= 0) & (x[active] < length)]
            throughput[active] *= albedo
            result[active] += throughput[active] * phase[angle[active], 0] * np.exp(-sigma*x[active])
            angle[active] = (rng.random(len(active)) >= phase[angle[active], 0]).astype(int)
        estimates.append(float(result.mean()))
        errors.append(float(result.std(ddof=1) / np.sqrt(count)))
    assert np.all(np.diff(estimates) > 0)
    assert exact - estimates[-1] < .02
    assert estimates[-1] < exact + 5*errors[-1]
    return {'bounces': [1, 4, 8, 16], 'analytic_full_transport': float(exact), 'means': estimates,
            'standard_errors': errors, 'scope': 'Two-stream slab; the GPU local walk uses continuous directions and a nonzero terminal cache.'}


def surface_integration():
    # A sunlit homogeneous half-space has a closed-form camera integral.
    # Test both shallow and steep light attenuation, with the same extinction used for camera transmission.
    ratios = np.array([.2, 1., 4.])
    exact = 1 / (1 + ratios)
    errors = []
    jitter = (np.arange(2048) + .5) / 2048
    for dt in (.5, .25, .125):
        start = np.arange(0, 24., dt)
        source = np.exp(-ratios[:, None, None] * (start[None, :, None] + dt*jitter))
        weights = np.exp(-start) * -np.expm1(-dt)
        estimate = np.sum(source * weights[None, :, None], axis=1).mean(axis=1)
        errors.append(float(max(abs(estimate - exact))))
    assert errors[-1] < errors[0] * .08
    assert errors[0] < .016
    old = np.exp(-ratios)
    assert max(abs(old-exact)) > .18
    return {'optical_steps': [.5, .25, .125], 'maximum_errors': errors,
            'old_one_mean_free_path_error': float(max(abs(old-exact)))}


def path_source_reservoir():
    rng = np.random.default_rng(8499)
    events, trials = 32, 160000
    throughput = rng.uniform(.05, 2., (events, 3))
    phase = np.geomspace(.001, 1000., 2*events).reshape(events, 2, 1)
    weight = (throughput[:, None, :]*phase).reshape(-1, 3)
    incident = np.broadcast_to(np.array([1., 1.2, .9]), weight.shape)
    cached = incident*rng.uniform(0., .9, weight.shape)
    actual = incident*rng.uniform(0., 1., weight.shape)
    cached[:4] = 0
    sky_cached = rng.uniform(0., .3, (events, 3))
    sky_actual = rng.uniform(0., .3, (events, 3))
    beam_p = np.max(weight*(cached+incident), axis=1)
    sky_p = np.max(throughput*np.array([.8, 1., .7]), axis=1)
    def select(proposal):
        selected = np.zeros(trials, dtype=int)
        total = 0.
        for i, p in enumerate(proposal):
            total += p
            selected[rng.random(trials)*total < p] = i
        return selected
    beam_i, sky_i = select(beam_p), select(sky_p)
    base = np.sum(weight*cached, axis=0)+np.sum(throughput*sky_cached, axis=0)
    samples = base+beam_p.sum()*weight[beam_i]/beam_p[beam_i, None]*(actual[beam_i]-cached[beam_i])
    samples += sky_p.sum()*throughput[sky_i]/sky_p[sky_i, None]*(sky_actual[sky_i]-sky_cached[sky_i])
    target = np.sum(weight*actual, axis=0)+np.sum(throughput*sky_actual, axis=0)
    z = abs(samples.mean(axis=0)-target)/(samples.std(axis=0, ddof=1)/np.sqrt(trials))
    assert z.max() < 5, z
    exact = base+np.sum(weight*(actual-cached), axis=0)+np.sum(throughput*(sky_actual-sky_cached), axis=0)
    error = float(np.max(abs(exact-target)))
    assert error < 1e-10
    return {'events': events, 'trials': trials, 'exact_predictor_cancellation_error': error,
            'standard_errors_from_full_source_sum': z.tolist(),
            'scope': 'Separate beam and sky selection; six orders of phase variation and missed coarse sources.'}


def spectral_path_variance():
    extinction = np.array([.18067367051236774, .18215559145291857, .19358580629452912])
    reference = extinction.mean()
    probabilities, weighted = flight_distribution(np.array([1., 0., 0.]), lambda p: np.ones(len(p)),
        lambda p: np.ones(len(p)), 8/reference, 1., extinction, extinction, np.ones(3))
    weights = weighted/probabilities[:, None]
    second = np.ones(3)
    for _ in range(32):
        second = (probabilities[:-1, None]*weights[:-1]**2).sum(axis=0)*second + probabilities[-1]*weights[-1]**2
    assert np.max(abs(weighted.sum(axis=0)-1)) < 1e-14
    assert second.max() < 1.3
    return {'order': 32, 'unit_equilibrium_mean': weighted.sum(axis=0).tolist(), 'second_moment': second.tolist(),
            'scope': 'Spectral segment proposal at fixed order; excludes phase and spatial-model error.'}


def shadow_quadrature():
    radius, bound = 2., 1.25
    extinction = np.array([.3, .7, 1.])
    center, width = .31, .055
    column = .05*radius+1.2*width*np.sqrt(np.pi/2)*(erf((radius-center)/(np.sqrt(2)*width))-erf(-center/(np.sqrt(2)*width)))
    exact = np.exp(-extinction*column)
    results = []
    scale = .25/(bound*extinction.max())
    for steps in (16, 32, 64, 128):
        edges = scale*np.expm1(np.linspace(0, np.log1p(radius/scale), steps+1))
        mid = .5*(edges[:-1]+edges[1:])
        density = .05+1.2*np.exp(-.5*((mid-center)/width)**2)
        tr = np.exp(-extinction*np.dot(density, np.diff(edges)))
        results.append({'steps': steps, 'transmittance': tr.tolist(), 'max_error': float(np.max(abs(tr-exact)))})
    assert results[-1]['max_error'] < results[0]['max_error']/20
    assert results[-1]['max_error'] < 1e-4
    return {'analytic_transmittance': exact.tolist(), 'quadrature': results,
            'scope': 'Finite exponential shadow bins; reports quadrature bias and convergence.'}


def boundary_moments():
    mu, weight = np.polynomial.legendre.leggauss(160)
    bulk = np.array([.01, .5, 2., 40., 7.])
    incident = np.array([2., .5, .01, 0., 20.])
    mean = .5*(bulk+incident)
    g = (bulk-incident)/(4*mean)
    field = mean[:, None]*(1-g[:, None]**2)/(1+g[:, None]**2-2*g[:, None]*mu)**1.5
    integral = .5*(field@weight)
    moment = .5*(field@(weight*mu))
    assert field.min() >= 0 and np.max(abs(g)) <= .5
    energy_error = np.max(abs(integral-mean))
    flux_error = np.max(abs(moment-(bulk-incident)/4))
    assert energy_error < 1e-11 and flux_error < 1e-11
    # Constant isotropic incident and interior fields must survive every geometry/blend weight.
    c = 3.7
    blend = np.linspace(0, 1, 101)
    equilibrium = (1-blend)*c+blend*(.5*(c+c))
    assert np.max(abs(equilibrium-c)) < 1e-14
    return {'angular_power_error': float(energy_error), 'angular_flux_error': float(flux_error),
            'equilibrium_error': float(np.max(abs(equilibrium-c))),
            'scope': 'Positive angular closure and Marshak matching; the sampled local boundary and full image remain approximate.'}


def ground_illumination():
    # Independent angular quadrature of a Lambertian receiver. A horizon-bright
    # sky distinguishes irradiance from an unweighted hemisphere average.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    mu = (nodes + 1) * .5
    weights *= .5
    skies = np.array([np.ones_like(mu), 1-mu, mu**4])
    reflected = 2 * np.sum(skies * mu * weights, axis=1)
    exact = np.array([1., 1/3, 1/3])
    error = float(np.max(abs(reflected-exact)))
    assert error < 2e-14
    old = np.sum(skies * weights, axis=1)
    assert abs(old[1] - exact[1]) > .16
    # Sun -> ground -> cloud: the two cloud legs attenuate the same ground bounce.
    # Moving a cloud over the sun leg must affect the receiver even if its own
    # view of the ground is unchanged. The old clear-ground input omitted this.
    sun_tau = np.array([0., .2, 2., 12.])
    return_tau = .4
    relative_light = np.exp(-(sun_tau + return_tau))
    assert np.all(np.diff(relative_light) < 0)
    assert relative_light[-1] / relative_light[0] < 1e-5
    return {'lambertian_quadrature_error': error, 'horizon_sky_radiance': float(reflected[1]),
            'old_unweighted_horizon_sky': float(old[1]), 'sun_optical_depths': sun_tau.tolist(),
            'ground_bounce_relative_radiance': relative_light.tolist(),
            'scope': 'Angular normalization and two-leg uncollided ground paths; no cloud-ground MS coupling proof.'}


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
    rng = np.random.default_rng(413)
    count = 60000
    means, standard_errors = [], []
    for arrangement in range(2):
        estimates, errors = [], []
        for direction in DIRS:
            samples = local_update(rng, count, direction, lambda p: density(p, arrangement), source,
                                   radius, 24*max(scales), np.ones(3), np.ones(3), np.full(3, 2.), steps=256)[:, 0]
            estimates.append(samples.mean())
            errors.append(samples.std(ddof=1) / np.sqrt(count))
        means.append(estimates)
        standard_errors.append(errors)
    z = abs(np.array(means) - ref) / np.array(standard_errors)
    assert np.max(z) < 5, z
    assert abs(ref[0, 23] - ref[1, 23]) > .5
    coarse_means = [density(cube, arrangement).mean() for arrangement in range(2)]
    assert abs(coarse_means[0] - coarse_means[1]) < 1e-13
    return {'same_coarse_density': list(map(float, coarse_means)), 'channel_radiance': ref[:, 23].tolist(),
            'maximum_standard_errors_from_ODE': float(z.max()), 'trials_per_direction': count,
            'scope': 'Same coarse mass, displaced narrow channel; 256-segment quadrature versus the continuous frozen-source ODE.'}


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
              'equilibrium': equilibrium(), 'spectral_reservoir': reservoir(), 'spectral_segments': spectral_segments(),
              'angular_guiding': angular_guiding(), 'scattering_orders': scattering_orders(), 'surface_integration': surface_integration(),
              'path_source_reservoir': path_source_reservoir(), 'spectral_path_variance': spectral_path_variance(), 'shadow_quadrature': shadow_quadrature(),
              'ground_illumination': ground_illumination(), 'boundary_moments': boundary_moments(), 'fine_geometry': geometry(),
              'not_established': ['Full transport unbiasedness or global power balance',
                                  'Angular convergence', 'Unbiased temporal filtering', 'Photo agreement']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
