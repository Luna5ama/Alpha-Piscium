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

def audit_tree(tree_path, audit_path, improved, lookahead):
    words = np.fromfile(tree_path, '<u4')[8:].reshape(-1,8)
    nodes = words.view('<f4').astype(float)
    parents, leaves = {}, {}
    pending = [1]
    while pending:
        index = pending.pop()
        if nodes[index,3] == 0:
            continue
        link = int(words[index,7])
        if link & 0x80000000:
            leaves[tuple(nodes[index,:3])] = index
        else:
            parents[link] = parents[link+1] = index
            pending.extend((link,link+1))
    records=np.fromfile(audit_path,'<f4').reshape(64,4,4).astype(float)
    errors=[]
    for record in records:
        if record[3,3] == 0:
            continue
        origin,normal=record[0,:3],record[1,:3]
        def score(index, depth):
            node=nodes[index]; mass=node[3];link=int(words[index,7])
            if mass == 0:
                return 0.
            if depth and not link & 0x80000000:
                return score(link,depth-1)+score(link+1,depth-1)
            half=(node[4:7]-node[:3])*.5
            delta=node[:3]+half-origin
            d2=delta@delta;r2=half@half
            if improved:
                e=abs(normal)@half;h=normal@delta
                h=h if h>=e else (0 if h<=-e else (h+e)**2/(4*e))
                scale=d2+r2/3
                return mass*max(np.clip(h/np.sqrt(scale),0,1),.001)/scale
            cosine=np.clip((normal@delta+np.sqrt(r2))/np.sqrt(max(d2,.25)),.001,1)
            return mass*cosine/max(d2,r2,.25)
        index=leaves[tuple(record[3,:3])];pdf=1.
        while index!=1:
            parent=parents[index];left=int(words[parent,7]);depth=2 if lookahead else 0
            a,b=score(left,depth),score(left+1,depth)
            pdf*=a/(a+b) if index==left else b/(a+b)
            index=parent
        errors.append(abs(record[1,3]/pdf-1))
    assert errors and max(errors)<.005, errors
    return dict(samples=len(errors),max_relative_pdf_error=max(errors))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_cube(args.audit), indent=2))
