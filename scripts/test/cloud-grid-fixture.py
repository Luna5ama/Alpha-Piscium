"""Compare the production 4 × 8 source probes on the finite reference tower, in metres."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np


def horizontal(n, thickness, original):
    scale = 16. if original else 4*max(thickness, .1)
    ratio = 4*(121/scale-.75)
    k = np.log(ratio) + np.log(np.log(ratio)+1)
    for _ in range(3):
        k -= (np.exp(k)-1-ratio*k)/(np.exp(k)-ratio)
    if original:
        k = 4.90219448065
    u = np.linspace(-1, 1, n+1)
    outer = np.maximum(np.abs(u)-.75, 0)*4
    edges = np.sign(u)*scale*(np.minimum(np.abs(u), .75) + .25*np.expm1(k*outer)/k)
    scaled = np.abs(edges)/scale
    inv = np.sign(edges)*(np.minimum(scaled, .75) + .25*np.log1p(4*k*np.maximum(scaled-.75, 0))/k)
    return edges*1000, np.max(np.abs(u-inv))


def volume(shapes, count):
    centers, radii = shapes[:, 0], shapes[:, 1]
    lo = np.min(centers-radii, axis=0)
    hi = np.max(centers+radii, axis=0)
    step = (hi-lo)/count
    x = lo[0]+(np.arange(count)+.5)*step[0]
    x2 = ((x[None, :]-centers[:, 0:1])/radii[:, 0:1])**2
    total = 0.
    for j in range(count):
        y = lo[1]+(j+.5)*step[1]
        disk = 1-x2-((y-centers[:, 1:2])/radii[:, 1:2])**2
        half = radii[:, 2:3]*np.sqrt(np.maximum(disk, 0))
        left, right = centers[:, 2:3]-half, centers[:, 2:3]+half
        order = left.argsort(axis=0)
        last = np.full(count, -np.inf)
        for a, b in zip(np.take_along_axis(left, order, axis=0), np.take_along_axis(right, order, axis=0)):
            total += np.maximum(b-np.maximum(a, last), 0).sum()
            last = np.maximum(last, b)
    return float(total*step[0]*step[1])


def probe(shapes, edges, axis):
    centers, radii = shapes[:, 0], shapes[:, 1]
    lo = (centers-radii).min(axis=0)
    hi = (centers+radii).max(axis=0)
    cells = [np.flatnonzero((e[:-1] < hi[i]) & (e[1:] > lo[i])) for i, e in enumerate(edges)]
    frac = []
    for quadrature, tap in product(range(4), range(8)):
        f = .25 + .5*np.array([quadrature & 1, quadrature >> 1, (quadrature ^ (quadrature >> 1)) & 1])
        f[axis] = 1-(tap+.5)/8
        frac.append(f)
    hits = occupied = 0
    estimate = 0.
    for cell in product(*cells):
        origin = np.array([e[i] for e, i in zip(edges, cell)])
        size = np.array([e[i+1] for e, i in zip(edges, cell)])-origin
        positions = origin+frac*size
        normalized = (positions[:, None]-centers)/radii
        n = int(np.any(np.sum(normalized**2, axis=-1) < 1, axis=1).sum())
        hits += n
        occupied += n > 0
        estimate += np.prod(size)*n/32
    return {'occupied_cells': occupied, 'density_hits': hits, 'volume_m3': estimate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text())
    tower = np.asarray(m['ellipsoids_center_radius'], dtype=float)
    neighbor = tower.copy()
    neighbor[:, 0] += m['neighbor_offset_m']
    coarse = volume(tower, 512)
    fine = volume(tower, 1024)
    report = {'camera_z_m': m['camera']['z'], 'layer_thickness_km': .3,
              'one_tower_volume_m3': {'512': coarse, '1024': fine}, 'grids': {}}
    for n, original in product((32, 64), (True, False)):
        x, inverse_error = horizontal(n, .3, original)
        y = np.linspace(-2500, 450, n+1)
        grid = {'inner_spacing_m': x[n//2+1]-x[n//2], 'far_edge_km': x[-1]*.001,
                'inverse_error': inverse_error, 'views': {}}
        for name, shapes, multiplier in [('solo', tower, 1), ('pair', np.concatenate((tower, neighbor)), 2)]:
            for light, axis in [('side', 1), ('back', 2)]:
                result = probe(shapes, (x, y, x+m['camera']['z']), axis)
                result['relative_volume_error'] = result['volume_m3']/(multiplier*fine)-1
                grid['views'][f'{name}/{light}'] = result
        # Translating the orthographic camera along -Z adds vacuum before
        # the first collision. It leaves the reference radiance unchanged.
        grid['camera_distance_m'] = {}
        for distance in (2000, 6000, 12000):
            grid['camera_distance_m'][distance] = {
                label: probe(tower, (x, y, x-distance), axis)
                for label, axis in [('side', 1), ('back', 2)]
            }
        report['grids'][('original' if original else 'layer')+f'/{n}'] = grid
    limits = [horizontal(64, h, False) for h in np.arange(0., 4.01, .1)]
    report['max_far_error_m'] = max(float(abs(e[-1]-121000)) for e, _ in limits)
    report['max_inverse_error'] = max(float(error) for _, error in limits)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
