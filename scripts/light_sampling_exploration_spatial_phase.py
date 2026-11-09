"""Check whether spatial noise ordering is sensitive to the origin of the tile grid."""
import argparse
import json
from pathlib import Path

import numpy as np

from light_sampling_exploration_capture import RUN,pixels
from light_sampling_point_moments import read,sha


SCENE="light-sampling-failure-moving-4"
PAIRS=[("uv-camera","uv-diagonal"),("uv-camera-b","uv-outer-near")]


def sat(a):
    p=np.zeros((a.shape[0]+1,a.shape[1]+1,*a.shape[2:]),dtype=np.float64)
    p[1:,1:]=a.cumsum(axis=0,dtype=np.float64).cumsum(axis=1,dtype=np.float64)
    return p


def rect(table,y,x,side):
    return table[y+side,x+side]-table[y,x+side]-table[y+side,x]+table[y,x]


def qualification(case):
    assert case["restore"]["status"]=="RECEIPT_STATUS_OK"
    assert case["settings"]["SETTING_DEBUG_LIGHT_SAMPLING"]=="2"
    return case


def matched_mask(folders,dimension):
    a,b=folders
    h,w=dimension["height"],dimension["width"]
    mask=np.ones((h,w),dtype=bool)
    for name,dt,shape in (("colortex8","<u4",(h,w,4)),("colortex9","<u4",(h,w)),("colortex10","<f4",(h,w))):
        x,y=[np.memmap(f/f"{name}.bin",dt,"r",shape=shape)[::-1] for f in (a,b)]
        if name=="colortex8":
            mask&=(((x[:,:,0]^y[:,:,0])&0xffff)==0)&(((x[:,:,3]^y[:,:,3])>>16)==0)
        elif name=="colortex9":
            mask&=((x^y)&0xffffff)==0
        else:
            mask&=x==y
    results=[]
    for folder in folders:
        measures=[pixels(folder,dimension,i) for i in (1,2)]
        for item in measures:
            mask&=item[2]
        results.append(measures)
    return mask,results


def summarize(data,mask):
    (a,va,*_),(b,vb,*_)=data
    delta=np.where(mask[:,:,None],a[:,:,:3].astype(float)-b[:,:,:3],0.)
    independence=np.where(mask,(va[:,:,:3].sum(axis=2,dtype=np.float64)+
                                vb[:,:,:3].sum(axis=2,dtype=np.float64))/64.,0.)
    return sat(delta),sat(independence)


def tail(vals):
    k=int(np.ceil(len(vals)*.05))
    return float(np.partition(vals,len(vals)-k)[-k:].mean())


def measure(names):
    dirs=[RUN/"cases"/SCENE/n for n in names]
    cases=[qualification(read(d/"accepted.json")) for d in dirs]
    a,b=cases
    k=("camera_sha256","world_voxel_sha256","leaf_sha256","frozen_face_uv_sha256")
    assert all(a["identity"][p]==b["identity"][p] for p in k)
    assert a["scene_sha256"]==b["scene_sha256"] and a["context"]==b["context"]
    probes=[c["identity"]["atlas_probes"][phase] for c in cases for phase in ("before","after")]
    assert all(p==probes[0] for p in probes)
    mask,vals=matched_mask(dirs,a["resolution"])
    table=sat(mask)
    by_method=[summarize(x,mask) for x in vals]
    h,w=mask.shape
    out={}
    for side in (16,32,64,128):
        origins=[]
        all_mse=[[],[]]
        for dy,dx in np.ndindex((4,4)):
            sy=dy*side//4
            sx=dx*side//4
            y,x=np.meshgrid(np.arange(sy,h-side+1,side),np.arange(sx,w-side+1,side),indexing="ij")
            counts=rect(table,y,x,side)
            ok=counts>=np.ceil(.95*side**2)
            y,x,n=y[ok],x[ok],counts[ok]
            values=[]
            for index,(delta,variance) in enumerate(by_method):
                sums=rect(delta,y,x,side)/n[:,None]
                mse=np.square(sums).sum(axis=1)/2.
                iid=rect(variance,y,x,side)/np.square(n)
                values.append(dict(mean=float(mse.mean()),tail=tail(mse),over_iid=float(mse.mean()/iid.mean())))
                all_mse[index].append(mse)
            origins.append(dict(offset=[sx,sy],tiles=len(n),methods=dict(zip(names,values,strict=True)),
                                mean_ratio=values[1]["mean"]/values[0]["mean"],
                                tail_ratio=values[1]["tail"]/values[0]["tail"]))
        combined=[np.concatenate(x) for x in all_mse]
        out[str(side)]=dict(phase_checks=origins,phase_pooled_tiles=len(combined[0]),
            phase_pooled_tail_ratio=tail(combined[1])/tail(combined[0]),
            phase_pooled_mean_ratio=float(combined[1].mean()/combined[0].mean()),
            mean_ratio_min_median_max=np.quantile([t["mean_ratio"] for t in origins],[0,.5,1]).tolist(),
            tail_ratio_min_median_max=np.quantile([t["tail_ratio"] for t in origins],[0,.5,1]).tolist())
    return dict(methods=names,matched_complete_pixels=int(mask.sum()),
                cases={n:sha(d/"accepted.json") for n,d in zip(names,dirs,strict=True)},tile_sizes=out)


def main(output):
    result=dict(pairs=[measure(n) for n in PAIRS],
                scope="Sensitivity to tile-grid placement from two existing matched 32f windows, not extra independent temporal windows. No comparisons across the two UV/atlas cohorts, no GPU timing or covariance cause claim.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    for pair in result["pairs"]:
        print(pair["methods"],{k:v["tail_ratio_min_median_max"] for k,v in pair["tile_sizes"].items()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    main(p.parse_args().output)
