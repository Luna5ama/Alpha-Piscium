"""Screen cheap alternatives to raising every outer near fraction."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import cohorts
from light_sampling_clustering_repair import EVIDENCE, key_order, scores
from light_sampling_point_moments import sha
from light_sampling_point_screening import Tree


def probability(tree, origin, normal):
    baseline, _, endpoints = scores(tree, origin, normal)
    nodes = tree.nodes
    lo, hi = nodes[:, :3], nodes[:, 4:7]
    delta = .5 * (lo + hi) - origin
    half = .5 * (hi - lo)
    cross = ((lo//16) != ((hi-1)//16)).any(axis=1)
    d2 = np.square(delta).sum(axis=1)
    r2 = np.square(half).sum(axis=1)
    contains = (np.abs(delta) <= half).all(axis=1)
    names = ('outer-near', 'noncontaining', 'containing', 'near-winner-outside',
             'near-winner-inside', 'agreement', 'disagreement')
    q = np.zeros((len(nodes), len(names)))
    q[0] = 1.
    null = np.zeros(len(names))
    for i, children in enumerate(tree.children):
        if not len(children) or not q[i].any():
            continue
        halves = endpoints[children]
        if not halves.any():
            null += q[i]
            continue
        n, f = (2. * halves).T
        mix = np.full(len(names), .5)
        if cross[i]:
            has_containing = (contains[children] & (n > 0)).any()
            winner = children[np.argmax(n)]
            outside = d2[winner] > r2[winner]
            tv = .5 * np.abs(n - f).sum()
            mix += .25 * np.asarray((1., not has_containing, has_containing, outside,
                                    not outside, 1.-tv, tv))
        q[children] = q[i] * (n[:, None] * mix + f[:, None] * (1.-mix))
    np.testing.assert_allclose(q[tree.leaves].sum(axis=0)+null, 1., rtol=0, atol=2e-12)
    return {'prefix': baseline, **dict(zip(names, q.T, strict=True))}


def main(output):
    refs = [EVIDENCE/f'20261108-{tag}-reference.npz' for tag in
            ('clustering-repair', 'cluster-summaries', 'nearfar-combination')]
    combination = EVIDENCE/'20261108-nearfar-combination.json'
    combo = json.loads(combination.read_text(encoding='utf8'))['interior_m2']['cohort']
    rows = []
    with np.load(refs[0]) as fresh, np.load(refs[1]) as old, np.load(refs[2]) as mix:
        for group, _, keys, root, windows in cohorts(fresh, old, mix, combo):
            tree = Tree(root)
            order = key_order(tree, list(map(tuple, keys)))
            cached = {}
            for point, window, label, origin, normal, moments in windows:
                x = tuple(np.r_[origin, normal])
                if x not in cached:
                    cached[x] = {n: p[order] for n,p in probability(tree, origin, normal).items()}
                m2 = moments[:, 3]
                mu = moments[:, :3].sum(axis=0)
                methods = {}
                for name, p in cached[x].items():
                    missing = int(((p == 0.) & (m2 > 0.)).sum())
                    v = np.divide(m2, p, out=np.zeros_like(m2), where=p>0).sum()-mu@mu
                    methods[name] = dict(variance=None if missing else float(v), missing_positive_support=missing)
                base = methods['prefix']['variance']
                for stats in methods.values():
                    stats['ratio_vs_prefix'] = None if stats['variance'] is None else stats['variance']/base
                rows.append(dict(cohort=group, point=point, window=window, label=label, methods=methods))
    assert len(rows) == 28
    report = dict(scope='Point-window screen only. Same support floor and distances; adapt 50/50 to at most 75/25 only at cross-brick parents.',
                  input_sha256={p.name:sha(p) for p in [*refs, combination]}, rows=rows)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
