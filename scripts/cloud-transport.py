#!/usr/bin/env python3
"""Build the cumulus angular transport table (numpy, scipy).

Run from scripts/: python cloud-transport.py --output ../builds/cloud-transport/phase.json
The table uses the actual filtered OPAC texture and the 11.54 um HG+Draine
diameter in Mediums.glsl. The two-channel closure preserves the homogeneous P3
scalar response; Jendersie and d'Eon supply the HG+Draine phase, not this solver.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.special import eval_legendre
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent.parent
DIAMETER = 11.54
COUNT = 24
WEIGHT = 4 * np.pi / COUNT


def directions() -> np.ndarray:
    signs = 1 - 2 * ((np.arange(8)[:, None] >> np.arange(3)) & 1)
    return np.concatenate([signs * (1 + (np.arange(3) == axis)) for axis in range(3)]) / np.sqrt(6)


def cell_ids(omega: np.ndarray) -> np.ndarray:
    return 8 * np.argmax(abs(omega), axis=-1) + np.sum((omega < 0) * (1 << np.arange(3)), axis=-1)


def hg_moments(diameter: float, count: int) -> np.ndarray:
    gh = np.exp(-0.0990567 / (diameter - 1.67154))
    gd = np.exp(-2.20679 / (diameter + 3.91029) - 0.428934)
    alpha = np.exp(3.62489 - 8.29288 / (diameter + 5.52825))
    wd = np.exp(-0.599085 / (diameter - 0.641583) - 0.665888)
    l = np.arange(count, dtype=float)
    cp = (l + 1) * (l + 2) / ((2 * l + 1) * (2 * l + 3))
    cc = (l + 1)**2 / ((2 * l + 1) * (2 * l + 3)) + l**2 / ((2 * l + 1) * (2 * l - 1))
    cm = l * (l - 1) / ((2 * l + 1) * (2 * l - 1))
    mu2 = cp * gd**(l + 2) + cc * gd**l + cm * gd**(l - 2)
    return (1 - wd) * gh**l + wd * (gd**l + alpha * mu2) / (1 + alpha * (1 + 2 * gd**2) / 3)


def phases(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.cos(theta)
    gh = np.exp(-0.0990567 / (DIAMETER - 1.67154))
    gd = np.exp(-2.20679 / (DIAMETER + 3.91029) - 0.428934)
    alpha = np.exp(3.62489 - 8.29288 / (DIAMETER + 5.52825))
    wd = np.exp(-0.599085 / (DIAMETER - 0.641583) - 0.665888)
    hg = lambda g: (1 - g*g) / (4 * np.pi * (1 + g*g - 2*g*mu)**1.5)
    blend = (1 - wd) * hg(gh) + wd * hg(gd) * (1 + alpha * mu**2) / (1 + alpha * (1 + 2*gd*gd) / 3)
    raw = np.fromfile(ROOT / 'shaders/textures/opac_cloud_phases.bin', dtype='<f2').reshape(3, 256, 4)[1].astype(float)
    u = (.672617934627 - .0713555761181 * theta + .0299320735609 * theta**2) * theta**.264767018876
    x = np.clip(u * 256 - .5, 0, 255)
    lo = x.astype(int)
    f = (x - lo)[:, None]
    rgbm = raw[lo] * (1 - f) + raw[np.minimum(lo + 1, 255)] * f
    return np.repeat(blend[:, None], 3, axis=1), rgbm[:, :3] * rgbm[:, 3:]


def pn_response(k: np.ndarray, rates: np.ndarray) -> np.ndarray:
    tail = np.full_like(k, rates[-1])
    for l in range(len(rates) - 2, -1, -1):
        tail = rates[l] + (l + 1)**2 * k*k / ((2*l + 1) * (2*l + 3) * tail)
    return 1 / tail


def channels(a0, a1, a2, a3):
    d0 = 1 / (3 * a1)
    s = 3 / (5 * a1) + 9 / (35 * a3)
    p = 3 / (35 * a1 * a3)
    df = (s + np.sqrt(s*s - 4*p)) / 2
    ds = p / df
    w = (df - d0) / (df - ds)
    return ds, df, w, w * (1 - w) * (a2 - a0)


def phase_matrix(theta: np.ndarray, cdf: np.ndarray, exponent: int = 20) -> np.ndarray:
    """Integrate pairs of angular cells by phase importance sampling.

    Cubic symmetry and interchange of the two cells reduce quadrature noise.
    All cells have the same area, so one common normalization preserves both
    row sums and reciprocity.
    """
    samples = qmc.Sobol(4, scramble=True, seed=5711).random_base2(exponent)
    y = 1 - 2 * samples[:, 0]
    azimuth = 2 * np.pi * samples[:, 1]
    radial = np.sqrt(1 - y*y)
    incoming = np.column_stack((radial * np.cos(azimuth), y, radial * np.sin(azimuth)))
    tangent = np.column_stack((-np.sin(azimuth), np.zeros(len(y)), np.cos(azimuth)))
    bitangent = np.cross(incoming, tangent)
    src = cell_ids(incoming)
    phi = samples[:, 3] * 2 * np.pi
    result = np.zeros((COUNT, COUNT, 3))
    for color in range(3):
        mu = np.cos(np.interp(samples[:, 2], cdf[:, color], theta))
        transverse = np.sqrt(np.maximum(1 - mu*mu, 0))
        outgoing = incoming * mu[:, None] + transverse[:, None] * (tangent * np.cos(phi)[:, None] + bitangent * np.sin(phi)[:, None])
        dst = cell_ids(outgoing)
        result[:, :, color] = np.bincount(src * COUNT + dst, minlength=COUNT*COUNT).reshape(COUNT, COUNT)
    result += result.transpose(1, 0, 2).copy()
    sym = np.zeros_like(result)
    dirs = directions()
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            mapping = cell_ids(dirs[:, perm] * signs)
            sym += result[mapping[:, None], mapping[None, :]]
    return sym / sym.sum(axis=1).mean(axis=0)


def build() -> dict:
    theta = np.r_[0, np.geomspace(1e-7, np.pi, 65536)]
    hg, mie = phases(theta)
    measure = 2 * np.pi * np.sin(theta)
    cdfs = [cumulative_trapezoid(p * measure[:, None], theta, axis=0, initial=0) for p in (hg, mie)]
    norms = [c[-1].copy() for c in cdfs]
    cdfs = [c / c[-1] for c in cdfs]
    beta_h = np.repeat(hg_moments(DIAMETER, 256)[:, None], 3, axis=1)
    beta_m = np.array([np.trapezoid(mie * (measure * eval_legendre(l, np.cos(theta)))[:, None], theta, axis=0) / norms[1] for l in range(256)])
    beta_m[0] = 1
    matrices = [phase_matrix(theta, cdf) for cdf in cdfs]
    # Rows 0/1: inverse angular CDF; 2/3: cell-integrated matrix; 4: metadata;
    # rows 5..25: a3 factors at phase ratio 0..1, 64 logarithmic Nyquist bands.
    table = np.zeros((26, 1024, 4), dtype='<f4')
    for phase, cdf in enumerate(cdfs):
        for color in range(3):
            table[phase, :, color] = np.cos(np.interp((np.arange(1024) + .5) / 1024, cdf[:, color], theta))
        table[phase + 2, :576, :3] = matrices[phase].reshape(576, 3)
    table[4, :3, :3] = beta_h[1:4]
    table[4, 3:6, :3] = beta_m[1:4]
    table[4, 6, :3] = norms[1]
    fit_error = 0.0
    for ratio_index in range(21):
        beta = beta_h * (1 - ratio_index / 20) + beta_m * ratio_index / 20
        for band_index, band in enumerate(np.exp2(np.linspace(-3, 3, 64))):
            for color in range(3):
                rates = 1 - beta[:, color]
                k = rates[1] * np.linspace(.001, band, 128)
                reference = pn_response(k, rates)
                def loss(log_factor):
                    fitted = rates[:4].copy()
                    fitted[3] *= np.exp(log_factor)
                    return np.mean(np.log(pn_response(k, fitted) / reference)**2)
                opt = minimize_scalar(loss, bounds=(0, 5), method='bounded')
                assert opt.success
                table[5 + ratio_index, band_index, color] = np.exp(opt.x)
                fit_error = max(fit_error, float(np.sqrt(opt.fun)))
    path = ROOT / 'shaders/textures/cloud_transport.bin'
    path.write_bytes(table.tobytes())
    return {'diameter_um': DIAMETER, 'shape': list(table.shape), 'bytes': table.nbytes,
            'sha256': hashlib.sha256(table.tobytes()).hexdigest(), 'mie_integral_before_normalization': norms[1].tolist(),
            'hg_moments': beta_h[:4].tolist(), 'mie_moments': beta_m[:4].tolist(),
            'matrix_samples_per_endpoint': 2**20, 'matrix_minimum': float(min(p.min() for p in matrices)),
            'matrix_row_error': float(max(np.max(abs(p.sum(axis=1) - 1)) for p in matrices)),
            'matrix_reciprocity_error': float(max(np.max(abs(p - p.transpose(1, 0, 2))) for p in matrices)),
            'maximum_fit_rms_log_error': fit_error, 'fit_band_k_over_a1': [.125, 8],
            'fit_scope': 'Conservative uniform medium, scalar response; per AP0 channel, actual phase blend, 256 moments.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
