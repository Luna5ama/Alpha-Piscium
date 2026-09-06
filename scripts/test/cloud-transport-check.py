#!/usr/bin/env python3
"""Independent sparse-matrix checks of the cloud transport discretization.

python test/cloud-transport-check.py --output ../builds/cloud-transport/numerics.json
Requires numpy and scipy. The 3D reference solves the complete directional
system to a measured residual; it is not a reference path-traced cloud image.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags, kron
from scipy.sparse.linalg import splu, gmres, LinearOperator

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cloud_transport', ROOT / 'scripts/cloud-transport.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
TABLE = np.fromfile(ROOT / 'shaders/textures/cloud_transport.bin', dtype='<f4').reshape(26, 1024, 4).astype(float)
P = TABLE[2, :576, 0].reshape(24, 24)
BETA = np.r_[1, TABLE[4, :3, 0]]
DIRS = model.directions()
W = model.WEIGHT


def incoming_weight(tau):
    t = np.maximum(tau, .01)
    tr = np.exp(-t)
    return np.where(tau < .01, .5-tau/12+tau**3/720, 1/t-tr/(1-tr))


def transport(rho, omega=1.0):
    n = rho.shape[0]
    m = rho.size
    rho = rho.ravel()
    rows, cols, values, mean_values = [], [], [], []
    diagonal = np.zeros(24*m)
    mean_diagonal = np.zeros(24*m)
    escape = np.zeros(24 * m)
    for angle, direction in enumerate(DIRS):
        sum_rate = abs(direction).sum()
        wi = incoming_weight(rho / sum_rate)
        for index, p in enumerate(np.ndindex((n, n, n))):
            p = np.array(p)
            i = angle * m + index
            diagonal[i] = sum_rate + rho[index] * (1-wi[index])
            mean_diagonal[i] = 1-wi[index]
            for axis in range(3):
                rate = abs(direction[axis])
                q = p.copy()
                q[axis] -= int(np.sign(direction[axis]))
                if np.all(q >= 0) and np.all(q < n):
                    rows.append(i); cols.append(angle * m + np.ravel_multi_index(q, (n, n, n)))
                    mean_weight = wi[index]*rate/sum_rate
                    mean_values.append(mean_weight)
                    values.append(-rate+rho[index]*mean_weight)
                q = p.copy()
                q[axis] += int(np.sign(direction[axis]))
                if np.any(q < 0) or np.any(q >= n):
                    escape[i] += rate
    streaming = coo_matrix((values, (rows, cols)), shape=(24*m, 24*m)).tocsr() + diags(diagonal)
    mean = coo_matrix((mean_values, (rows, cols)), shape=(24*m, 24*m)).tocsr() + diags(mean_diagonal)
    scatter = kron(P, diags(rho * omega), format='csr') @ mean
    return streaming, scatter, escape, mean


def diffusion(rho, omega):
    shape = rho.shape
    m = rho.size
    rho = rho.ravel()
    present = rho > 0
    # Empty unknowns are pinned to zero; no density floor connects vacuum gaps.
    ds = np.zeros(m); df = np.zeros(m); w = np.zeros(m); exchange = np.zeros(m)
    rates = rho[present, None] * (1 - omega * BETA)
    ds[present], df[present], w[present], exchange[present] = model.channels(*rates.T)
    diagonal = np.r_[w * rho * (1-omega), (1-w) * rho * (1-omega)]
    diagonal[np.r_[~present, ~present]] = 1
    rows, cols, values = [], [], []
    def edge(i, j, g):
        diagonal[i] += g; diagonal[j] += g
        rows.extend((i, j)); cols.extend((j, i)); values.extend((-g, -g))
    boundary = np.zeros(2*m)
    for channel, (weight, d) in enumerate(((w, ds), (1-w, df))):
        k = weight * d
        for i, p in enumerate(np.ndindex(shape)):
            if not present[i]: continue
            for axis in range(3):
                for side in (-1, 1):
                    q = list(p); q[axis] += side
                    j = np.ravel_multi_index(q, shape) if all(0 <= q[a] < shape[a] for a in range(3)) else -1
                    if j >= 0 and present[j]:
                        if side == 1: edge(channel*m+i, channel*m+j, 2*k[i]*k[j]/(k[i]+k[j]))
                    else:
                        g = weight[i] / (.5/d[i] + 2)
                        diagonal[channel*m+i] += g
                        boundary[channel*m+i] += g
    for i in np.flatnonzero(present): edge(i, m+i, exchange[i])
    matrix = coo_matrix((values, (rows, cols)), shape=(2*m, 2*m)).tocsc() + diags(diagonal)
    return matrix, w, boundary


def solve_system(rho, omega, source):
    l, scatter, escape, mean = transport(rho, omega)
    a = l - scatter
    lu = splu(l.tocsc())
    solution, status = gmres(a, source, M=LinearOperator(a.shape, matvec=lu.solve), rtol=1e-11, atol=0, restart=80, maxiter=100)
    assert status == 0, status
    residual = np.linalg.norm(source - a @ solution) / np.linalg.norm(source)
    assert residual < 1e-9 and solution.min() >= -1e-10
    pin = W * source.sum()
    pout = W * (escape @ solution + np.tile(rho.ravel() * (1-omega), 24) @ (mean @ solution))
    assert abs(pout / pin - 1) < 1e-8
    return solution, a, scatter, lu, residual, abs(pout/pin-1)


def algebra():
    rng = np.random.default_rng(239)
    worst = 0
    for _ in range(200):
        omega = rng.uniform(0, 1)
        g = rng.uniform(-.9, .99)
        rates = 1 - omega * g**np.arange(4)
        ds, df, w, exchange = model.channels(*rates)
        r = rates[2] - rates[0]
        k = np.geomspace(.001, 100, 160)
        numerator = rates[0] + r + (w * df + (1-w) * ds) * k*k
        denominator = rates[0]*(rates[0]+r) + (rates[0]*(ds+df)+r/(3*rates[1]))*k*k + ds*df*k**4
        worst = max(worst, float(np.max(abs(numerator / denominator / model.pn_response(k, rates) - 1))))
        assert ds > 0 and df > ds and 0 < w < 1 and exchange >= 0
        assert abs(w*ds+(1-w)*df - 1/(3*rates[1])) < 1e-11
    assert worst < 1e-12
    return worst


def test_3d():
    n = 6
    x, y, z = np.indices((n, n, n))
    rho = np.where((x == 2) | (x == 3), 0.0, 3 + np.exp(-((y-2.3)**2+(z-3.1)**2)/3))
    power = np.exp(-((x-.5)**2+(y-2.5)**2+(z-3)**2)/2) * rho
    source = (P[:, 8, None] / W * power.ravel()).ravel()
    cases = []
    for omega in (1.0, .999, .95):
        reference, a, scatter, lu, residual, balance = solve_system(rho, omega, source)
        core, w, boundary = diffusion(rho, omega)
        core_lu = splu(core)
        q = power.ravel()
        u = core_lu.solve(np.r_[w*q, (1-w)*q])
        phi = w*u[:rho.size]+(1-w)*u[rho.size:]
        core_balance = abs(((rho.ravel()*(1-omega)*phi).sum()+boundary@u)/q.sum()-1)
        assert u.min() >= -1e-10 and core_balance < 1e-9
        methods = {}
        for method in ('source_iteration', 'isotropic_correction', 'transported_correction'):
            current = np.zeros_like(source)
            history = []
            for iteration in range(4):
                current = lu.solve(source + scatter @ current)
                r = source - a @ current
                before = np.linalg.norm(r)
                gamma = 0
                if method != 'source_iteration':
                    r0 = W * r.reshape(24, -1).sum(axis=0)
                    u = core_lu.solve(np.r_[w*r0, (1-w)*r0])
                    phi = w*u[:rho.size]+(1-w)*u[rho.size:]
                    delta = np.tile(phi / (4*np.pi), 24)
                    if method == 'transported_correction': delta = lu.solve(np.tile(rho.ravel()*omega, 24) * delta)
                    ad = a @ delta
                    dd = ad @ ad
                    limit = min(1, np.min(-.99*current[delta < 0]/delta[delta < 0], initial=1))
                    gamma = max(0, min(limit, r @ ad / dd)) if dd > 0 else 0
                    current += gamma * delta
                    assert current.min() >= -1e-10
                    assert np.linalg.norm(source-a@current) <= before * (1+1e-10)
                history.append({'residual': float(np.linalg.norm(source-a@current)/np.linalg.norm(source)), 'gamma': float(gamma)})
            methods[method] = {'relative_solution_error': float(np.linalg.norm(current-reference)/np.linalg.norm(reference)), 'iterations': history}
        gap = reference.reshape(24, n, n, n)[:, 2:4]
        assert gap.max() > 0
        cases.append({'omega': omega, 'directional_residual': residual, 'power_balance_error': balance,
                      'minimum': float(reference.min()), 'vacuum_gap_radiance_max': float(gap.max()),
                      'core_power_balance_error': core_balance, 'methods': methods})
    # Thin-limit MS camera source must include a second scattering coefficient.
    energies = []
    for epsilon in (1e-3, 5e-4, 2.5e-4):
        scaled = rho * epsilon
        b = source * epsilon
        s, *_ = solve_system(scaled, 1, b)
        _, scatter, _, _ = transport(scaled)
        energies.append(float((scatter @ s).sum()))
    powers = np.log(np.array(energies[:-1])/energies[1:]) / np.log(2)
    assert np.max(abs(powers-2)) < .02
    # Superposition and the pure-vacuum zero-source case.
    s1, a, scatter, lu, *_ = solve_system(rho, 1, source)
    s2, *_ = solve_system(rho, 1, source.reshape(24, -1)[::-1].ravel())
    combined, *_ = solve_system(rho, 1, source + source.reshape(24, -1)[::-1].ravel())
    linearity = np.linalg.norm(combined-s1-s2) / np.linalg.norm(combined)
    assert linearity < 1e-9
    vacuum_l, _, _, _ = transport(np.zeros_like(rho))
    assert np.all(splu(vacuum_l.tocsc()).solve(np.zeros_like(source)) == 0)
    return {'cases': cases, 'thin_limit_exponents': powers.tolist(), 'linearity_error': float(linearity)}


def spatial_limit():
    results = []
    # Actual cell-average/face-flux Fourier symbol, not the diffusion preconditioner.
    # This reports (and does not hide) the coarse-grid artificial diffusion.
    k = 1e-4
    for rho in (.1, 1, 10):
        rate = abs(DIRS).sum(axis=1)
        shift = np.exp(-1j*np.sign(DIRS[:, 0])*k)
        mean = 1 + incoming_weight(rho/rate)*abs(DIRS[:, 0])/rate*(shift-1)
        symbol = rho*(np.eye(24)-P) @ np.diag(mean)
        symbol += np.diag(abs(DIRS[:, 0]) * (1-shift))
        response = (mean * np.linalg.solve(symbol, np.ones(24))).mean()
        numerical_d = (1/response).real / (k*k)
        results.append({'sigma_t_h': rho, 'continuum_D0': float(1/(3*rho*(1-BETA[1]))), 'directional_grid_D': float(numerical_d)})
    return results


def source_and_history():
    tau = np.array([1e-7, .2, 4., 20., 100.])
    absorbed = -np.expm1(-tau)
    point_samples = tau*(np.exp(-.25*tau)+np.exp(-.75*tau))*.5
    assert absorbed[-1] > .999999 and point_samples[-1] < 1e-8
    # The characteristic's mean and outgoing value must balance their source.
    incoming = np.array([.2, .7, 1., 3., 4.])
    source = np.array([1., .8, .7, .5, .3])
    outgoing = np.exp(-tau)*incoming+absorbed/tau*source
    mean = (1-incoming_weight(tau))*outgoing+incoming_weight(tau)*incoming
    balance = float(np.max(abs(outgoing-incoming+tau*mean-source)))
    assert balance < 1e-12

    rho = np.full((4, 4, 4), 3.)
    rho[1, 1:3] = 0
    b = (P[:, 8, None]/W*rho.ravel()).ravel()
    reference, a, scatter, lu, *_ = solve_system(rho, 1., b)
    _, _, escape, mean_matrix = transport(rho)
    core, w, _ = diffusion(rho, 1.)
    core_lu = splu(core)
    current = np.zeros_like(b)
    history = []
    for frame in range(16):
        for iteration in range(4):
            current = lu.solve(b+scatter@current)
            r = b-a@current
            r0 = W*r.reshape(24, -1).sum(axis=0)
            u = core_lu.solve(np.r_[w*r0, (1-w)*r0])
            phi = w*u[:rho.size]+(1-w)*u[rho.size:]
            delta = lu.solve(np.tile(rho.ravel(), 24)*np.tile(phi/(4*np.pi), 24))
            ad = a@delta
            limit = min(1., np.min(-.99*current[delta < 0]/delta[delta < 0], initial=1.))
            gamma = max(0., min(limit, (r@ad)/(ad@ad)))
            power_before = escape@current
            corrected_power = power_before+gamma*(escape@delta)
            current += gamma*delta
            assert abs(corrected_power-escape@current) < 1e-10
            residual = np.linalg.norm(b-a@current)/np.linalg.norm(b)
            power_defect = abs((escape@current)/b.sum()-1)
            if residual < .005 and power_defect < .005: break
        history.append({'frame': frame+1, 'iterations': iteration+1, 'residual': float(residual), 'power_defect': float(power_defect)})
    assert history[-1]['iterations'] == 1 and power_defect < .005 and residual < .005
    assert (mean_matrix@current).min() >= 0
    return {'characteristic_balance_error': balance, 'beam_absorbed_power': absorbed.tolist(),
            'old_point_sample_power': point_samples.tolist(), 'warm_start': history,
            'final_solution_error': float(np.linalg.norm(current-reference)/np.linalg.norm(reference))}


def gpu_readback(path, n):
    tree = sum((n >> level)**3 for level in range(int(np.log2(n)) - 1))
    cells = n**3
    groups = cells // 64
    counts = (32, n+1, 576, 24, 24, cells*2, tree*4, tree*2, tree*2,
              cells*24, cells*24, cells*24, cells*24, groups*7)
    names = ('stats', 'edges', 'phase', 'sun_phase', 'moon_phase', 'material',
             'coefficients', 'rhs', 'error', 'radiance', 'camera_source', 'sky', 'lifted', 'partials')
    expected = sum(counts) * 16
    assert path.stat().st_size == expected, (path.stat().st_size, expected)
    raw = np.fromfile(path, dtype='<f4').reshape(-1, 4)
    assert np.isfinite(raw).all(), 'Non-finite GPU state'
    fields = {}
    offset = 0
    for name, count in zip(names, counts):
        fields[name] = raw[offset:offset+count]
        offset += count
    s = fields['radiance'][:, :3]
    assert s.min() >= -1e-6, float(s.min())
    matrix = fields['phase'][:, :3].reshape(24, 24, 3)
    assert matrix.min() >= 0 and np.max(abs(matrix.sum(axis=1)-1)) < 1e-6
    assert np.max(abs(matrix-matrix.transpose(1, 0, 2))) < 1e-6
    for name in ('sun_phase', 'moon_phase'):
        assert np.max(abs(fields[name][:, :3].sum(axis=0)*W-1)) < 1e-6
        assert fields[name].min() >= 0
    stats = fields['stats']
    assert np.all(stats[5:9, :3] <= stats[1:5, :3] + 1e-5)
    incident = stats[14, :3]
    emitted = stats[15, :3]
    ratio = np.divide(emitted, incident, out=np.zeros(3), where=incident > 0)
    mean_field = fields['lifted'][:, :3].reshape(24, cells, 3)
    cached_mean = fields['coefficients'][:4*cells].reshape(cells, 4, 4)[:, 1:4, 3]
    mean_error = np.max(abs(cached_mean-mean_field.mean(axis=0))/np.maximum(mean_field.mean(axis=0), 1e-7))
    convolution = np.einsum('aic,ijc->ajc', matrix, mean_field)
    camera_source = fields['camera_source'][:, :3].reshape(24, cells, 3)
    convolution_error = np.max(abs(convolution-camera_source)/np.maximum(abs(convolution), 1e-7))
    assert mean_field.min() >= -1e-6 and mean_error < 1e-5 and convolution_error < 1e-5
    iterations = int(stats[24, 0])
    return {'resolution': n, 'bytes': expected, 'all_finite': True,
            'minimum_radiance': float(s.min()), 'maximum_radiance': float(s.max()),
            'mean_cache_relative_error': float(mean_error), 'phase_convolution_relative_error': float(convolution_error),
            'iterations': iterations, 'reused_history': bool(stats[24, 1]), 'history_displacement_km': stats[25, :3].tolist(),
            'raw_residuals': stats[1:1+iterations, :3].tolist(), 'predicted_corrected_residuals': stats[5:5+iterations, :3].tolist(),
            'accepted_damping': stats[9:9+iterations, :3].tolist(), 'final_relative_residual': stats[13, :3].tolist(),
            'predicted_power_defects': stats[20:20+iterations, :3].tolist(),
            'input_power': incident.tolist(), 'escaped_plus_absorbed_power': emitted.tolist(),
            'finite_budget_power_ratio': ratio.tolist(), 'coarse_cg': stats[16:20].tolist(),
            'scope': 'Actual GPU state; fixed-budget residual and power defect, not a converged energy-conservation claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-buffer', type=Path)
    parser.add_argument('--resolution', type=int, choices=(32, 64), default=32)
    args = parser.parse_args()
    assert P.min() >= 0 and np.max(abs(P.sum(axis=1)-1)) < 1e-6
    assert np.max(abs(P-P.T)) < 1e-8
    report = gpu_readback(args.gpu_buffer, args.resolution) if args.gpu_buffer else {'p3_identity_error': algebra(), 'phase_matrix_row_error': float(np.max(abs(P.sum(axis=1)-1))),
              'phase_matrix_reciprocity_error': float(np.max(abs(P-P.T))), 'transport_3d': test_3d(),
              'characteristic_diffusion_limit': spatial_limit(), 'source_and_history': source_and_history(),
              'not_established': ['Angular or spatial grid convergence of rendered clouds', 'Final raymarch global power balance', 'Photo agreement']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
