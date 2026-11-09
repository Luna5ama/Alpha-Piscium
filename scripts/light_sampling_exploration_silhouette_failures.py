"""Decode the real receiver, cube and random draw for numerical failures."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from light_sampling_geometry import cube_integral
from light_sampling_point_moments import read, sha


def decode(folder, rejected_restore=False, far_plane=False, artifact="audit", rejected_input=False):
    capture = read(folder/('diagnostic-only.json' if (rejected_restore or rejected_input) else 'accepted.json'))
    if rejected_restore:
        assert capture['status']=='rejected_transaction_restore_failure'
    if rejected_input:
        assert capture['status']=='rejected_camera_input_drift'
    path = folder/f"{artifact}.bin"
    data = np.fromfile(path,'<u4')
    counts = data[1024:1040].astype(int)
    expected = (11,12,13,14) if far_plane else (11,12)
    assert counts[15]==sum(counts[i] for i in expected),counts
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
        assert reason in expected
        h = cube_integral(lo.copy(),o,n)
        nearest = np.clip(o,lo,lo+1.)-o
        item = dict(reason=reason,fast_proposal=bool(flag&256),pixel_gl=[packed&65535,packed>>16],
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
        if "gap" in item and item["gap"]>0.:
            mid = o+d*((item["double_entry"]+item["double_exit"])*.5)
            item["distance_outside_at_midpoint"] = float(np.linalg.norm(mid-np.clip(mid,lo,lo+1.)))
        failures.append(item)
    h = [row['projected_solid_angle'] for row in failures]
    gaps = [row['relative_gap'] for row in failures if 'relative_gap' in row and row['reason'] in (12,14)]
    report = dict(job=capture['job'],source=capture['source'],audit_sha256=sha(path),failure_count=int(counts[15]),
                  rejected_restore=rejected_restore,rejected_input=rejected_input,proposal="far_plane" if far_plane else "silhouette",
                  reasons=dict(Counter(x['reason'] for x in failures)),records=failures,
                  projected_solid_angle_min_median_max=np.quantile(h,[0,.5,1]).tolist() if h else None,
                  relative_miss_min_median_max=np.quantile(gaps,[0,.5,1]).tolist() if gaps else None,
                  scope='Real failing GPU draws only; double slab check distinguishes a rounded boundary from a ray outside the cube.')
    return report


def series(folder, far_plane, rejected_input=False):
    files = sorted(folder.glob("audit-*.bin"))
    stems = ["audit",*(f.stem for f in files)]
    assert len(stems)>1
    frames = [read(folder/f"{stem}.json")["frame_id"] for stem in stems]
    assert all(a<b for a,b in zip(frames,frames[1:]))
    decoded = [decode(folder,False,far_plane,stem,rejected_input) for stem in stems]
    reasons = Counter()
    for report in decoded:
        reasons.update(report["reasons"])
    return dict(job=decoded[0]["job"],source=decoded[0]["source"],frames=frames,reasons=dict(reasons),
                failures=sum(r["failure_count"] for r in decoded),frames_and_witnesses=decoded,rejected_input=rejected_input,
                scope="Distinct recorded native debug frames at the same scene and camera. Not a timing test.")


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--rejected-restore',action='store_true')
    p.add_argument('--far-plane',action='store_true')
    p.add_argument('--series',action='store_true')
    p.add_argument('--rejected-input',action='store_true')
    args = p.parse_args()
    if args.series:
        assert not args.rejected_restore
        result = series(args.folder,args.far_plane,args.rejected_input)
    else:
        result = decode(args.folder,args.rejected_restore,args.far_plane,rejected_input=args.rejected_input)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
