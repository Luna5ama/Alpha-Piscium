"""Reproduce conditional RGB M2 and the two offline near/far correction checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from light_sampling_weight_reference import distribution, load_tree
from light_sampling_weight_analysis import leaf_identity, topology_identity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'builds/light-sampling/20261031-nearfar-iteration'


def components(nodes, origin, normal):
    half = (nodes[:, 4:7]-nodes[:, :3])*.5
    delta = nodes[:, :3]+half-origin
    radius2 = np.sum(half*half, axis=1)
    h = delta @ normal
    extent = half @ np.abs(normal)
    amp = nodes[:, 3]*np.clip((h+np.sqrt(radius2))/np.sqrt(np.maximum(np.sum(delta*delta, axis=1), .25)), .001, 1)
    amp[h+extent < -1e-6*(np.sum(np.abs(delta*normal), axis=1)+extent+1)] = 0
    dmin2 = np.sum(np.maximum(np.abs(delta)-half, 0)**2, axis=1)
    dmax2 = np.sum((np.abs(delta)+half)**2, axis=1)
    zero = (dmin2 == 0) & (amp > 0)
    near = np.where(zero, amp, 0) if zero.any() else amp/np.maximum(dmin2, 1e-300)
    far = np.divide(amp, dmax2, out=np.zeros_like(amp), where=dmax2 > 0)
    bounded = amp/np.maximum.reduce([dmin2, radius2, np.full(len(nodes), .25)])
    return [p/p.sum() if p.sum() else p for p in (near, far, bounded)]


def proposals(nodes, links, origin, normal, ids):
    # Track pure near, pure far, current local mixture, and the bounded-near mixture.
    pending = [(1, np.ones(4))]
    leaves, zero = {}, np.zeros(4)
    while pending:
        i, p = pending.pop()
        if nodes[i, 3] == 0:
            zero += p
        elif links[i] & 0x80000000:
            leaves[i] = p
        else:
            link = int(links[i]); base, count = link & 0xfffffff, ((link >> 28) & 7)+1
            near, far, bounded = components(nodes[base:base+count], origin, normal)
            if not far.sum():
                zero += p
            else:
                for j in range(count):
                    factors = np.array([near[j], far[j], .5*(near[j]+far[j]), .5*(bounded[j]+far[j])])
                    if factors.any():
                        pending.append((base+j, p*factors))
    np.testing.assert_allclose(sum(leaves.values())+zero, np.ones(4), atol=1e-10)
    by_leaf = np.array([leaves.get(int(i), np.zeros(4)) for i in ids])
    current, _ = distribution(nodes, links, origin, normal, 6)
    np.testing.assert_allclose(by_leaf[:, 2], [current.get(int(i), 0) for i in ids], atol=1e-15, rtol=1e-12)
    output = {}
    for m in (1, 3):
        q, _ = distribution(nodes, links, origin, normal, m)
        output[str(m)] = np.array([q.get(int(i), 0) for i in ids])
    output.update({'6': by_leaf[:, 2], 'pure_near': by_leaf[:, 0], 'pure_far': by_leaf[:, 1],
                   'coherent': .5*(by_leaf[:, 0]+by_leaf[:, 1]), 'bounded_near': by_leaf[:, 3]})
    assert np.all(output['3'] > 0)
    return output


def analyze(run):
    folder = run / 'candidate-runtime-evidence/ls1-point-windows'
    fixture = json.loads((ROOT / 'experiments/light-sampling/20261031-ls1-point-fixture.json').read_text(encoding='utf-8'))
    texture = ROOT / fixture['texture']
    assert hashlib.sha256(texture.read_bytes()).hexdigest() == fixture['texture_sha256']
    records = np.fromfile(texture, '<i4').reshape(-1, 4)[4:]
    _, nodes, links = load_tree(folder / 'tree.bin', 1)
    identity = leaf_identity(folder / 'tree.bin', 1)
    assert hashlib.sha256(repr(identity).encode()).hexdigest() == fixture['leaf_mass_sha256']
    assert topology_identity(folder / 'tree.bin', 1) == fixture['topology_sha256']
    # Only root-reachable nodes participate; reserved storage can contain stale duplicates.
    leaf_map, pending = {}, [1]
    while pending:
        i = pending.pop()
        if nodes[i, 3] == 0:
            continue
        link = int(links[i])
        if link & 0x80000000:
            leaf_map[tuple(nodes[i, :3].astype(int))] = i
        else:
            base, count = link & 0xfffffff, ((link >> 28) & 7)+1
            pending.extend(range(base, base+count))
    ids = np.array([leaf_map[tuple(r[:3])] for r in records])
    assert len(ids) == 3921 and all(int(links[i] & 0x7fffffff) == int(r[3]) for i, r in zip(ids, records))
    audit = np.fromfile(folder / 'audit.bin', '<f4')[:32].reshape(2, 4, 4)
    result = dict(scope='Conditional fixed-point finite-sample RGB moments; offline hypotheses are not GPU or full-ROI acceptance.', points=[])
    for point in range(2):
        expected = fixture['receivers'][point]
        origin, normal = audit[point, 0, :3].astype(float), audit[point, 1, :3].astype(float)
        assert audit[point, 2, 3] == 1 and int(audit[point, 0, 3]) == expected['material_id']
        assert np.linalg.norm(origin-expected['origin']) < .005 and normal @ expected['normal'] > .999
        assert int(audit[point, 1, 3]) == 264
        q = proposals(nodes, links, origin, normal, ids)
        windows = []
        previous_end = -1
        for window in (1, 2):
            meta = json.loads((folder / f'window{window}.mean.json').read_text(encoding='utf-8'))
            acc = meta['accumulation']
            assert acc['frames'] == 32 and acc['batches'] == 8 and acc['end_frame']-acc['start_frame'] == 31 and previous_end < acc['start_frame']
            previous_end = acc['end_frame']
            def read(kind):
                return np.memmap(folder / f'window{window}.{kind}.bin', '<f4', mode='r', shape=(2073600, 4))[:264*2*3921].reshape(264, 2, 3921, 4)[:, point]
            mean, variance = read('mean'), read('variance')
            good = np.isfinite(mean).all(axis=2)
            for kind, value in [('valid_count',32), ('nonfinite',0), ('incomplete_batches',0)]:
                good &= np.all(read(kind) == value, axis=2)
            count = good.sum(axis=0)
            assert np.all(count > 0)
            moments = np.sum(np.where(good[:, :, None], mean, 0), axis=0, dtype=float)/count[:, None]
            square_mean = np.sum(mean[:, :, :3].astype(float)**2, axis=2)
            variance_sum = variance[:, :, :3].sum(axis=2, dtype=float)
            residual = mean[:, :, 3]-square_mean-variance_sum*(31/32)
            scale = np.maximum(abs(mean[:, :, 3])+square_mean+variance_sum, 1e-20)
            max_error = float(np.max(np.abs(residual[good])/scale[good]))
            assert max_error < 1e-4
            total_mean = moments[:, :3].sum(axis=0)
            positive = moments[:, 3] > 0
            methods = {}
            for name, probs in q.items():
                failures = np.flatnonzero((probs == 0) & positive)
                terms = np.divide(moments[:, 3], probs, out=np.zeros_like(probs), where=probs > 0)
                prediction = None if len(failures) else float(terms.sum()-total_mean @ total_mean)
                methods[name] = dict(predicted_sum_rgb_variance=prediction, measured_positive_m2_support_failures=failures.tolist(),
                                     probability_on_measured_positive_m2_lights=float(probs[positive].sum()))
            assert all(not methods[m]['measured_positive_m2_support_failures'] for m in ('1','3','6'))
            windows.append(dict(window=window, complete_pixel_groups=int(good.sum()), missing_pixel_groups=int(good.size-good.sum()),
                positive_m2_lights=int(positive.sum()), rgb_mean_total=total_mean.tolist(), second_moment_identity_max_relative_error=max_error, methods=methods))
        result['points'].append(dict(raw_xy=expected['raw_xy'], origin=origin.tolist(), normal=normal.tolist(), windows=windows))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / 'recomputed-point-reference.json'
    result = analyze(args.run_root)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(points=len(result['points']), windows=4, output=str(output))))
