"""Match real motion receipts to the optional in-shader traversal trace."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_point_moments import read, sha


MARKER = 0x4d4f544e


def decode(folder, method, poses):
    path = folder/f'{method}-tree-trace.bin'
    trace = np.fromfile(path,'<u4',count=1024).reshape(64,4,4)
    trace = trace[trace[:,2,3]==MARKER]
    trace = trace[np.argsort(trace[:,0,3])]
    x = trace[:,0,:3].view('<f4').astype(float)
    expected = np.asarray(poses)[:,:3]
    used = np.flatnonzero((expected[:,0]>=111.89)&(expected[:,0]<=112.15))
    assert len(used)>1 and np.all(np.diff(used)==1)
    match = None
    for start in range(len(trace)-len(used)+1):
        end = start+len(used)
        seq = trace[start:end,0,3].astype(np.int64)
        if np.all(np.diff(seq)==1) and np.allclose(x[start:end],expected[used],rtol=0,atol=3e-5):
            match = slice(start,end)
    assert match is not None, (method,'no trace interval matches the trajectory',x.tolist(),expected[used].tolist())
    aligned = trace[match]
    assert np.all(aligned[:,1,3]==aligned[:,2,1])
    results = []
    for offset,row in zip(used,aligned,strict=True):
        b = np.floor(expected[offset]/16.).astype(int)
        leaf = (int(row[1,0]),int(row[1,1]),int(row[1,3]))
        shape = (int(row[1,2]),int(row[2,0]))
        results.append(dict(offset=int(offset), shader_frame=int(row[0,3]), brick=b.tolist(),
                            leaf_fingerprint=leaf, topology_fingerprint=shape, coarse_count=int(row[2,2])))
    cohorts = {}
    for row in results:
        cohorts.setdefault(row['leaf_fingerprint'],[]).append(row)
    groups = []
    for leaf,frames in cohorts.items():
        shapes = sorted({r['topology_fingerprint'] for r in frames})
        bricks = sorted({tuple(r['brick']) for r in frames})
        groups.append(dict(leaf_fingerprint=leaf, frames=len(frames), bricks=bricks, shapes=shapes,
                           cross_brick_input_match=len(bricks)>1, same_ordered_topology=len(shapes)==1))
    return dict(trace_path=path.as_posix(),sha256=sha(path),groups=groups,frames=results)


def main(folder, output):
    receipt = folder/'accepted.json'
    evidence = read(receipt)
    cases = {n:decode(folder,n,p['poses']) for n,p in evidence['methods'].items()}
    report = dict(job=evidence['job'],receipt_sha256=sha(receipt),methods=cases,
                  scope='Two commutative GPU leaf fingerprints plus ordered tree fingerprint, not a cryptographic or complete raw-tree identity. Motion diagnostic is not a performance measurement.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('output',type=Path)
    a = parser.parse_args()
    main(a.folder,a.output)
