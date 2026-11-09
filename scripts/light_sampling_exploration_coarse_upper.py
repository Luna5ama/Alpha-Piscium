"""Keep coarse native subtrees and recluster only their roots."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_hierarchy import cohorts
from light_sampling_clustering_repair import EVIDENCE, build_primitives, key_order, scores
from light_sampling_clustering_rule_screen import RULES
from light_sampling_exploration_outer_mix import outer_mix
from light_sampling_moving_analysis import leaves, ordered_signature
from light_sampling_point_moments import sha
from light_sampling_point_screening import Tree


def cell_roots(root, size):
    roots, pending = {}, [root]
    while pending:
        node = pending.pop()
        lo, hi = node.data[:3], node.data[4:7]
        cell = tuple(int(x)//w for x,w in zip(lo,size,strict=True))
        if cell != tuple((int(x)-1)//w for x,w in zip(hi,size,strict=True)):
            assert node.children
            pending.extend(node.children)
        else:
            assert cell not in roots, (cell, size)
            roots[cell] = node
    return roots


def coarse_tree(root, size, rule='flux-diagonal'):
    atoms = cell_roots(root, size)
    result = build_primitives(np.asarray(list(atoms)), list(atoms.values()), True, partition=RULES[rule])
    assert sorted(leaves(result)) == sorted(leaves(root))
    restored = cell_roots(result, size)
    assert restored.keys() == atoms.keys() and all(restored[k] is v for k,v in atoms.items())
    return result, len(atoms)


def adaptive_tree(root):
    """Spend 128 atoms on the largest diagonal-cost reduction per added root."""
    atoms = list(cell_roots(root, (64,64,64)).values())
    assert len(atoms)<=128
    def key(node):
        return tuple(int(node.data[i]) for i in (0,1,2,4,5,6))
    def cost(node):
        d = node.data[4:7]-node.data[:3]
        return node.data[3]*float(d@d)
    while len(atoms)<128:
        choices = []
        for i,n in enumerate(atoms):
            if not n.children or (n.data[:3]//16 == (n.data[4:7]-1)//16).all():
                continue
            extra = len(n.children)-1
            assert extra>0
            if len(atoms)+extra>128:
                continue
            gain = (cost(n)-sum(cost(c) for c in n.children))/extra
            choices.append((-gain,key(n),i))
        if not choices or min(choices)[0]<=0.:
            break
        i = min(choices)[2]
        atoms[i:i+1] = atoms[i].children
    keys = np.asarray([key(n) for n in atoms])
    result = build_primitives(keys,atoms,True,partition=RULES['flux-diagonal'])
    assert sorted(leaves(result)) == sorted(leaves(root))
    return result,len(atoms)


def main(output, adaptive_only=False):
    refs = [EVIDENCE/f'20261108-{tag}-reference.npz' for tag in
            ('clustering-repair', 'cluster-summaries', 'nearfar-combination')]
    combination = EVIDENCE/'20261108-nearfar-combination.json'
    combo = json.loads(combination.read_text(encoding='utf8'))['interior_m2']['cohort']
    rows, trees = [], {}
    with np.load(refs[0]) as fresh, np.load(refs[1]) as old, np.load(refs[2]) as mix:
        for group, _, keys, root, windows in cohorts(fresh, old, mix, combo):
            keys = list(map(tuple, keys))
            variants = {'prefix': root}
            trees[group] = {}
            # A four-way Morton level consumes two bits. A 2-brick cube
            # (three bits) does not align with a native subtree.
            cells = [(16,16,16), (32,32,16), (64,32,32), (64,64,64)]
            definitions = [(c,'flux-diagonal') for c in cells] + [(c,'count-median') for c in cells[1:]]
            if adaptive_only:
                definitions = [((16,16,16),'flux-diagonal'),((64,64,64),'flux-diagonal')]
                variants['adaptive-128'], count = adaptive_tree(root)
                trees[group]['adaptive-128'] = dict(coarse_roots=count, ordered_sha256=ordered_signature(variants['adaptive-128']))
            for cell, rule in definitions:
                name = f'{"x".join(map(str,cell))}/{rule}'
                variants[name], count = coarse_tree(root, cell, rule)
                trees[group][name] = dict(coarse_roots=count, ordered_sha256=ordered_signature(variants[name]))
            variants = {n: Tree(r) for n,r in variants.items()}
            orders = {n: key_order(t, keys) for n,t in variants.items()}
            cached = {}
            for point, window, label, origin, normal, moments in windows:
                x = tuple(np.r_[origin, normal])
                if x not in cached:
                    pmfs = {}
                    for n,t in variants.items():
                        base = scores(t, origin, normal)
                        pmfs[n] = base[0][orders[n]]
                        pmfs[n+'+near'] = outer_mix(t, base, .75)[orders[n]]
                    cached[x] = pmfs
                m2 = moments[:, 3]
                mu = moments[:, :3].sum(axis=0)
                methods = {}
                for n,p in cached[x].items():
                    missing = int(((p == 0.) & (m2 > 0.)).sum())
                    variance = np.divide(m2,p,out=np.zeros_like(m2),where=p>0).sum()-mu@mu
                    methods[n] = dict(variance=None if missing else float(variance), missing_positive_support=missing)
                base = methods['prefix']['variance']
                for stats in methods.values():
                    stats['ratio_vs_prefix'] = None if stats['variance'] is None else stats['variance']/base
                rows.append(dict(cohort=group, point=point, window=window, label=label, methods=methods))
    assert len(rows)==28
    report = dict(scope=('Point-window screen. Start from level 3 and refine at most to single bricks, with a 128-root budget.'
                         if adaptive_only else 'Point-window screen. Preserve subtrees of actual 4-way Morton levels 0 through 3.'),
                  input_sha256={p.name:sha(p) for p in [*refs, combination]}, trees=trees, rows=rows)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--adaptive-only',action='store_true')
    a = parser.parse_args()
    main(a.output,a.adaptive_only)
