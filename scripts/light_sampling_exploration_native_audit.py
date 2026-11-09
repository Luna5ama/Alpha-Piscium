"""Retain a native ray/PDF audit whose estimator inputs are written into the audit buffer.

The projection may vary. This only admits a raw diagnostic, never quality or timing.
"""
import argparse
import json

import numpy as np

from light_sampling_exploration_capture import ROOT,RUN,retain_manifest
from light_sampling_point_moments import input_identity,read,sha


def main(job_id,name):
    assert name.endswith("-audit")
    job=ROOT/".vibris/jobs"/job_id
    req=read(job/"request.json")["arguments"]
    saved=read(job/"result.json")
    assert saved["success"]
    r=saved["result"]
    p=r["provenance"]
    assert not p["stale"] and not p["patch_sha256"] and not p["dirty_shader_delta_sha256"]
    assert p["resolved_revision"]==req["sources"][0]["resolved_revision"]
    assert p["source_snapshot_sha256"]==req["sources"][0]["snapshot_sha256"]
    assert p["preset_id"]==req["preset_id"]
    assert r["restoration"]["status"]=="RECEIPT_STATUS_OK"
    assert len(req["actions"])==len(r["action_receipts"])
    assert all(x["status"]=="RECEIPT_STATUS_OK" and x["kind"]!="ACTION_KIND_GET_GPU_METRICS" for x in r["action_receipts"])
    c={x["name"]:x["value"] for x in p["effective_settings"]["settings"]}
    assert all(c[k]==v for k,v in req["configs"][0]["values"].items())
    assert c["SETTING_DEBUG_LIGHT_AUDIT"]=="true" and c["SETTING_DEBUG_LIGHT_SAMPLING"]=="2"
    folder=RUN/"cases"/p["preset_id"]/name
    manifest,ref=retain_manifest(job,r,folder)
    ctx=req["__vibris_scene_context"]
    phases={t:input_identity(folder,int(c["SETTING_VOXEL_GRID_SIZE"]),[0,0,0],t,ctx)[0] for t in ("before","after")}
    before,after=phases["before"],phases["after"]
    keys=("world_voxel_sha256","leaf_sha256","ordered_topology_sha256")
    assert all(before[k]==after[k] for k in keys)
    record=read(folder/"audit.json")["frame_id"]
    same_frame=[t for t in phases if read(folder/f"{t}_tree.json")["frame_id"]==record]
    assert same_frame, "An audit needs its exact tree checkpoint, not an inferred transient tree"
    headers={t:np.fromfile(folder/f"{t}_tree.bin","<u4",count=8).tolist() for t in same_frame}
    counters=np.fromfile(folder/"audit.bin","<u4")[1024:1040].tolist()
    result=dict(job=job_id,name=name,scene=p["preset_id"],context=ctx,settings=c,source=p["resolved_revision"],
                scene_sha256=p["scene_sha256"],restore=r["restoration"],path=folder.relative_to(ROOT).as_posix(),
                manifest_id=manifest["manifest_id"],manifest_sha256=ref["sha256"],
                bracket=phases,identity={k:before[k] for k in keys},audit_frame=record,
                same_frame_tree_header=headers,full_frame_reason_counters=counters,
                audit_sha256=sha(folder/"audit.bin"),quality_or_timing_qualified=False,timings=None,
                scope="Ray/PDF observation only. The matching native tree, full leaf/voxel brackets, and per-ray inputs support estimator checks independently of live projection.")
    (folder/"accepted.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("job")
    p.add_argument("name")
    a=p.parse_args()
    main(a.job,a.name)
