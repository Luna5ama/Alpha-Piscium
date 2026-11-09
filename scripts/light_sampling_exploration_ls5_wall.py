"""Archive the LS5 fixed light diagnostic and explain point-level variance deltas."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_clustering_repair import key_order, scores
from light_sampling_exploration_capture import ROOT, RUN, retain_manifest
from light_sampling_exploration_outer_mix import outer_mix
from light_sampling_moving_analysis import leaves, native_node
from light_sampling_point_moments import extract, input_identity, read, sha
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import load_tree


FIXTURE = RUN/"ls5-wall-mode3-fixture.json"
SCENE = "light-sampling-5"
CASES = RUN/"cases"/SCENE
NAME = "ls5-wall-mode3"


def archive(job_id):
    job = ROOT/".vibris/jobs"/job_id
    request = read(job/"request.json")["arguments"]
    saved = read(job/"result.json")
    assert saved["success"]
    result = saved["result"]
    provenance = result["provenance"]
    assert result["restoration"]["status"]=="RECEIPT_STATUS_OK"
    assert len(result["action_receipts"])==len(request["actions"])
    assert all(r["status"]=="RECEIPT_STATUS_OK" for r in result["action_receipts"])
    fixture = read(FIXTURE)
    assert provenance["resolved_revision"]==request["sources"][0]["resolved_revision"]==fixture["source"]
    assert provenance["source_snapshot_sha256"]==request["sources"][0]["snapshot_sha256"]
    assert request["preset_id"]==provenance["preset_id"]==SCENE and not provenance["stale"]
    assert not provenance["patch_sha256"] and not provenance["dirty_shader_delta_sha256"]
    config = {v["name"]:v["value"] for v in provenance["effective_settings"]["settings"]}
    assert all(config[k]==v for k,v in request["configs"][0]["values"].items())
    assert config["SETTING_DEBUG_LIGHT_SAMPLING"]=="3"
    assert config["SETTING_DEBUG_LIGHT_AUDIT"]=="false"
    folder = CASES/NAME
    manifest, ref = retain_manifest(job,result,folder)
    world_brick = np.floor(fixture["camera_position"]).astype(int)>>4
    context = request["__vibris_scene_context"]
    snapshots = [input_identity(folder,16,world_brick,p,context) for p in ("before","between","after")]
    for identity,_ in snapshots:
        assert all(identity[k]==v for k,v in fixture["identity"].items()),identity
        assert identity==snapshots[0][0]
    extracted = extract(folder,fixture,folder/"moments")
    assert len(extracted)==4 and all(not x["unavailable_lights"] for x in extracted)
    out = dict(job=job_id,source=fixture["source"],path=folder.relative_to(ROOT).as_posix(),
               manifest_id=manifest["manifest_id"],manifest_sha256=ref["sha256"],
               scene_sha256=provenance["scene_sha256"],identity=snapshots[0][0],
               fixture_sha256=sha(FIXTURE),moments_path=(folder/"moments/moments.json").relative_to(ROOT).as_posix(),
               moments_sha256=sha(folder/"moments/moments.json"),
               sampled_groups=[dict(point=x["point"],window=x["window"],
                   incomplete=x["missing_pixel_groups"],nonfinite=x["nonfinite_rgb_samples"],
                   samples_per_light=x["samples_per_frame"]*32) for x in extracted])
    (folder/"accepted.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print(json.dumps(out))


def replay():
    fixture=read(FIXTURE)
    moments=CASES/NAME/"moments"
    acceptance=read(CASES/NAME/"accepted.json")
    cases={"camera":"camera-paired-anchor","anchor-75":"dynamic-coarse-anchor",
           "anchor-50":"anchor-geometry-only","prefix":"prefix","coarse-sah":"dynamic-coarse-upper"}
    trees={}
    keys=None
    for name,path in cases.items():
        folder=CASES/path
        case=read(folder/"accepted.json")
        assert all(case["identity"][k]==v for k,v in fixture["identity"].items())
        _,nodes,links=load_tree(folder/"before_tree.bin",6)
        node=native_node(nodes,links,1)
        rows=sorted(leaves(node))
        if keys is None:
            keys=[(*map(int,row[:3]),row[3]) for row in rows]
        assert keys==[(*map(int,row[:3]),row[3]) for row in rows]
        trees[name]=Tree(node)
    order={n:key_order(t,keys) for n,t in trees.items()}
    audit=np.fromfile(CASES/NAME/"audit.bin","<f4")[:32].reshape(2,4,4).astype(float)
    out=[]
    for i,point in enumerate(fixture["receivers"]):
        origin,normal=audit[i,0,:3],audit[i,1,:3]
        probs={}
        for name,t in trees.items():
            base=scores(t,origin,normal)
            full=outer_mix(t,base,.75) if name in ("anchor-75","coarse-sah") else base[0]
            probs[name]=full[order[name]]
        for window in (1,2):
            src=moments/f"{point['id']}-window{window}.npz"
            with np.load(src) as z:
                rgb, m2=z["mean_rgb"],z["m2"]
            assert np.isfinite(rgb).all() and np.isfinite(m2).all()
            mu=rgb.sum(axis=0)
            result={}
            contribution={}
            for name,q in probs.items():
                assert not ((m2>0)&(q==0)).any()
                contribution[name]=np.divide(m2,q,out=np.zeros_like(m2),where=q>0)
                result[name]=dict(variance=float(contribution[name].sum()-mu@mu),
                                  positive_mass=float(q[m2>0].sum()))
            for name in result:
                delta=contribution[name]-contribution["camera"]
                result[name]["variance_vs_camera"]=result[name]["variance"]/result["camera"]["variance"]
                result[name]["largest_increases"]=[
                    dict(key=list(keys[j]),q_camera=float(probs["camera"][j]),q=float(probs[name][j]),
                         moment2=float(m2[j]),mean_rgb=rgb[j].tolist(),variance_delta=float(delta[j]))
                    for j in np.argsort(delta)[-8:][::-1] if delta[j]>0.]
            out.append(dict(point=point["id"],window=window,origin=origin.tolist(),normal=normal.tolist(),
                            raw_xy=point["raw_xy"],mean_rgb=mu.tolist(),methods=result,moments_sha256=sha(src)))
    report=dict(job=acceptance["job"],fixture_sha256=acceptance["fixture_sha256"],rows=out,
                scope="Exact native emitter set and measured per-light RGB moments. CPU replay of proposals at two fixed points; separate from whole-image statistics or timing.")
    (RUN/"ls5-wall-mode3-variance-replay.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps([dict(point=r["point"],window=r["window"],variances={n:m["variance_vs_camera"] for n,m in r["methods"].items()}) for r in out]))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("operation",choices=("archive","replay"))
    p.add_argument("job",nargs="?")
    a=p.parse_args()
    if a.operation=="archive":
        archive(a.job)
    else:
        replay()
