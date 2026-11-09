"""Decode the real receiver, cube and random draw for numerical failures."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from light_sampling_geometry import cube_integral
from light_sampling_point_moments import read, sha


def main(folder, output, rejected_restore=False):
    capture = read(folder/('diagnostic-only.json' if rejected_restore else 'accepted.json'))
    if rejected_restore:
        assert capture['status']=='rejected_transaction_restore_failure'
    path = folder/'audit.bin'
    data = np.fromfile(path,'<u4')
    counts = data[1024:1040].astype(int)
    assert counts[15]==counts[11]+counts[12] and counts[15]>0,counts
    records = data[:1024].reshape(64,4,4)[:min(64,counts[15])]
    failures = []
    def vector(x):
        return [float(a) if np.isfinite(a) else None for a in x]
    for bits in records:
        row = bits.view('<f4').astype(float)
        o,n,d,lo = row[:,:3]
        packed = int(bits[3,3])
        flag = int(bits[0,3])
        reason = flag&255
        assert reason in (11,12)
        h = cube_integral(lo.copy(),o,n)
        nearest = np.clip(o,lo,lo+1.)-o
        item = dict(reason=reason,silhouette=bool(flag&256),pixel_gl=[packed&65535,packed>>16],
                    origin=vector(o),normal=vector(n),direction=vector(d),cube_low=vector(lo),
                    randoms=vector(row[1:3,3]),projected_solid_angle=float(h),
                    distance_to_cube=float(np.linalg.norm(nearest)))
        if np.isfinite(d).all():
            with np.errstate(divide='ignore',invalid='ignore'):
                a,b=(lo-o)/d,(lo+1.-o)/d
            entry,exit_=np.minimum(a,b).max(),np.maximum(a,b).min()
            item.update(cosine=float(d@n),dir_norm=float(np.linalg.norm(d)))
            if np.isfinite(entry) and np.isfinite(exit_):
                item.update(double_entry=float(entry),double_exit=float(exit_),gap=float(entry-exit_),
                            relative_gap=float((entry-exit_)/max(abs(entry),abs(exit_),1e-12)))
        failures.append(item)
    h = [row['projected_solid_angle'] for row in failures]
    gaps = [row['relative_gap'] for row in failures if 'relative_gap' in row and row['reason']==12]
    report = dict(job=capture['job'],source=capture['source'],audit_sha256=sha(path),failure_count=int(counts[15]),
                  rejected_restore=rejected_restore,
                  reasons=dict(Counter(x['reason'] for x in failures)),records=failures,
                  projected_solid_angle_min_median_max=np.quantile(h,[0,.5,1]).tolist(),
                  relative_miss_min_median_max=np.quantile(gaps,[0,.5,1]).tolist() if gaps else None,
                  scope='Real failing GPU draws only; double slab check distinguishes a rounded boundary from a ray outside the cube.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--rejected-restore',action='store_true')
    args = p.parse_args()
    main(args.folder,args.output,args.rejected_restore)
