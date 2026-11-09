"""Archive camera projection checkpoints and reject a drifting native case."""
import argparse
import json

import numpy as np

from light_sampling_exploration_capture import ROOT,RUN,retain_manifest
from light_sampling_point_moments import read,sha


def main(job_id,name):
    job=ROOT/".vibris/jobs"/job_id
    req=read(job/"request.json")["arguments"]
    full=read(job/"result.json")
    assert full["success"]
    result=full["result"]
    assert result["restoration"]["status"]=="RECEIPT_STATUS_OK"
    assert len(result["action_receipts"])==len(req["actions"])
    assert all(a["status"]=="RECEIPT_STATUS_OK" for a in result["action_receipts"])
    folder=RUN/"projection-preflights"/req["preset_id"]/name
    manifest,ref=retain_manifest(job,result,folder)
    dumps=sorted(folder.glob("camera-*.bin"))
    assert len(dumps)==3
    frames=[read(p.with_suffix(".json"))["frame_id"] for p in dumps]
    assert all(a<b for a,b in zip(frames,frames[1:]))
    parts=[np.fromfile(p,"<f4") for p in dumps]
    vals=np.stack([np.r_[p[16:32],p[80:112]] for p in parts])
    assert np.isfinite(vals).all()
    v=[list(map(float,p[[80,85,96,101]])) for p in parts]
    fov=[float(np.degrees(2.*np.arctan(1./p[85]))) for p in parts]
    report=dict(job=job_id,scene=req["preset_id"],frames=frames,
                manifest_id=manifest["manifest_id"],manifest_sha256=ref["sha256"],
                camera_words_stable=bool(np.array_equal(vals,vals[[0,0,0]])),
                view_stable=bool(np.array_equal(vals[:,:16],vals[[0,0,0],:16])),
                projection_scales=v,actual_vertical_fov_degrees=fov,
                buffers={p.name:sha(p) for p in dumps})
    (folder/"accepted.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("job")
    p.add_argument("name")
    a=p.parse_args()
    main(a.job,a.name)
