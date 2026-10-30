"""Analysis and independent PDF checks for the visibility-free geometry matrix."""
import argparse
import json
from pathlib import Path
import numpy as np
from light_sampling_runtime import preserve, read_image, render, tree_stats, LUMA


def cube_integral(low, origin, normal):
    total = 0.0
    for axis in range(3):
        n = np.eye(3)[axis]
        o = origin[axis]
        if low[axis] <= o <= low[axis] + 1:
            continue
        center = low + 0.5
        center[axis] = low[axis] + (o > low[axis])
        u = np.roll(n, 1) * 0.5
        v = np.roll(n, 2) * (0.5 if o < low[axis] else -0.5)
        vertices = [center-u-v-origin, center+u-v-origin, center+u+v-origin, center-u+v-origin]
        clipped = []
        for a, b in zip(vertices, vertices[1:]+vertices[:1]):
            za, zb = a @ normal, b @ normal
            if za >= 0:
                clipped.append(a)
            if za * zb < 0:
                clipped.append(a + (b-a) * za/(za-zb))
        if len(clipped) < 3:
            continue
        vertices = np.array(clipped)
        vertices /= np.linalg.norm(vertices, axis=1)[:, None]
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            cross = np.cross(a, b)
            sine = np.linalg.norm(cross)
            if sine:
                total += 0.5 * (cross @ normal) * np.arctan2(sine, a @ b) / sine
    return max(0., total)


def audit_cube(path):
    data = np.fromfile(path, '<f4').reshape(64, 4, 4).astype(float)
    errors = []
    for origin4, normal4, direction4, low4 in data:
        if direction4[3] <= 0:
            continue
        origin, normal, direction, low = origin4[:3], normal4[:3], direction4[:3], low4[:3]
        assert np.isfinite(direction4).all()
        assert abs(np.linalg.norm(direction)-1) < 2e-5
        h = cube_integral(low.copy(), origin, normal)
        expected = (normal @ direction) / h
        errors.append(abs(direction4[3]/expected-1))
        t0, t1 = (low-origin)/direction, (low+1-origin)/direction
        assert np.max(np.minimum(t0,t1)) <= np.min(np.maximum(t0,t1)) + 1e-4
    assert errors
    return dict(samples=len(errors), max_relative_pdf_error=max(errors), p95_relative_pdf_error=float(np.quantile(errors,.95)))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_cube(args.audit), indent=2))
