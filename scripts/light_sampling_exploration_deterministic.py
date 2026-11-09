"""Verify fixed-projection, native shader block means recorded during deterministic warmup."""
import argparse
import json

import numpy as np

from light_sampling_exploration_capture import ROOT,RUN,retain_manifest
from light_sampling_exploration_lowfreq import checkpoint,pool,stats
from light_sampling_point_moments import read,sha


def read_pose(folder,name):
    p=folder/f"{name}--projection.bin"
    u=np.fromfile(p,"<u4",count=1024).reshape(128,2,4)
    a=u.view("<f4")
    assert np.array_equal(a[:,1,3],np.arange(16,144))
    assert np.isfinite(a[:,0,:2]).all() and np.isfinite(a[:,1,:]).all()
    keys=np.concatenate((u[:,0,:],u[:,1,:3]),axis=1)
    constant=np.equal(keys,keys[:1]).all(axis=1)
    fov=2.*np.degrees(np.arctan(1./a[:,0,1].astype(float)))
    return keys,dict(sha256=sha(p),constant_frames=int(constant.sum()),total_frames=len(keys),
                     projection_and_view_unchanged=bool(constant.all()),
                     vertical_fov_min_max=[float(fov.min()),float(fov.max())],
                     first_position=a[0,1,:3].astype(float).tolist())


def measurements(folder,tag,h,w):
    sum_path=folder/f"{tag}--sums.bin"
    rec_path=folder/f"{tag}--receiver.bin"
    s=np.memmap(sum_path,"<f4","r",shape=(h*2,w*2,4))
    r=np.memmap(rec_path,"<u4","r",shape=(h,w,4))[::-1]
    data=[s[y*h:(y+1)*h,x*w:(x+1)*w][::-1] for y in range(2) for x in range(2)]
    stable=(r[:,:,1]&0x80000000)==0
    common=stable.copy()
    for image in data:
        n=image[:,:,3]
        assert np.isfinite(n).all() and (n>=0).all() and (n<=32).all() and (np.floor(n)==n).all()
        common&=(n==32)&np.isfinite(image[:,:,:3]).all(axis=2)
    return data,r,common


def main(job_id,name):
    job=ROOT/".vibris/jobs"/job_id
    req=read(job/"request.json")["arguments"]
    saved=read(job/"result.json")
    assert saved["success"]
    result=saved["result"]
    assert result["restoration"]["status"]=="RECEIPT_STATUS_OK"
    assert len(req["actions"])==len(result["action_receipts"])
    assert all(a["status"]=="RECEIPT_STATUS_OK" for a in result["action_receipts"])
    folder=RUN/"deterministic-quality"/req["preset_id"]/name
    manifest,ref=retain_manifest(job,result,folder)
    context=req["__vibris_scene_context"]
    identity={phase:checkpoint(folder,phase,context) for phase in ("before","after")}
    same_world=all(identity["before"][k]==identity["after"][k] for k in ("emitters","leaf_sha256","world_voxel_sha256"))
    sources={r["id"]:r for r in req["sources"]}
    config={r["id"]:r["values"] for r in req["configs"]}
    methods={}
    tags=[]
    h,w=(context["resolution"][k] for k in ("height","width"))
    keys={}
    data={}
    for action,receipt in zip(req["actions"],result["action_receipts"],strict=True):
        if action["type"]!="load_shader":
            continue
        tag=action["config_id"]
        tags.append(tag)
        expected=sources[action["source_id"]]
        mutation=receipt["runtime_mutation"]
        assert expected["snapshot_sha256"]==mutation["source_sha256"]
        current={r["name"]:r["value"] for r in mutation["effective_settings"]["settings"]}
        assert all(current[k]==v for k,v in config[tag].items())
        assert current["SETTING_DEBUG_LIGHT_SAMPLING"]=="2" and current["SETTING_DEBUG_LIGHT_AUDIT"]=="false"
        keys[tag],pose=read_pose(folder,tag)
        data[tag]=measurements(folder,tag,h,w)
        uv=sha(folder/f"{tag}--fixed-uv.bin")
        crops={p.name.split("--",1)[1]:sha(p) for p in sorted(folder.glob(f"{tag}--*-atlas-*.bin"))}
        assert len(crops)==8
        methods[tag]=dict(source=expected["resolved_revision"],scene_sha256=mutation["scene_sha256"],
                          pose=pose,uv_sha256=uv,atlas_endpoint_crops=crops,
                          complete_and_constant_receiver_pixels=int(data[tag][2].sum()))
    a,b=tags
    same_pose=np.array_equal(keys[a],keys[b]) and all(methods[k]["pose"]["projection_and_view_unchanged"] for k in tags)
    same_scene=methods[a]["scene_sha256"]==methods[b]["scene_sha256"]
    same_uv=methods[a]["uv_sha256"]==methods[b]["uv_sha256"]
    probes=sorted({k.split("atlas-",1)[1] for k in methods[a]["atlas_endpoint_crops"]})
    same_atlas={p:len({v for m in methods.values() for k,v in m["atlas_endpoint_crops"].items() if k.endswith(p)})==1
                for p in probes}
    joint=(data[a][1]==data[b][1]).all(axis=2)&data[a][2]&data[b][2]
    qualified=bool(same_world and same_pose and same_scene and same_uv and all(same_atlas.values()) and joint.sum() > w*h*.5)
    if qualified:
        for tag in tags:
            means=[x[:,:,:3]/32. for x in data[tag][0]]
            metrics={}
            for size in (16,64):
                tiles=[pool(m,joint,size) for m in means]
                pairs=[np.square(x-y).sum(axis=1)/2. for x,y in zip(tiles[::2],tiles[1::2],strict=True)]
                metrics[str(size)]=dict(pairs=[stats(p) for p in pairs],average_of_pairs=stats(np.mean(pairs,axis=0)))
            methods[tag]["tile_noise"]=metrics
    report=dict(job=job_id,path=folder.relative_to(ROOT).as_posix(),scene=req["preset_id"],resolution=context["resolution"],
                manifest_id=manifest["manifest_id"],manifest_sha256=ref["sha256"],qualified=qualified,
                same_world=same_world,same_pose=bool(same_pose),same_scene=same_scene,same_fixed_uv=same_uv,
                atlas_crops_match=same_atlas,common_complete_stable_receivers=int(joint.sum()),
                identity=identity,methods=methods,
                scope="4 distinct native 32-frame blocks in static camera-motion time. Shader records sum and valid count per pixel; no performance numbers. Actual camera hashes and consumed receiver data checked throughout. Atlas checks cover four crops at the endpoints.")
    (folder/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("job")
    p.add_argument("name")
    a=p.parse_args()
    main(a.job,a.name)
