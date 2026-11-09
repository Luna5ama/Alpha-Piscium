"""Estimate how often the selective near rule would visit an extra native child."""
import json

import numpy as np
from matplotlib.path import Path as Polygon

from light_sampling_exploration_capture import RUN,pixels
from light_sampling_exploration_containment import selective_near
from light_sampling_moving_analysis import native_node
from light_sampling_point_moments import read,sha
from light_sampling_point_screening import Tree
from light_sampling_weight_reference import load_tree


def prepare(scene):
    p=RUN/"cases"/scene/"dynamic-coarse-anchor"
    receipt=read(p/"accepted.json")
    camera=np.fromfile(p/"before_camera.bin","<f4")
    model=camera[16:32].reshape(4,4).T.astype(float)
    proj=camera[96:112].reshape(4,4).T.astype(float)
    h,w=(receipt["resolution"][k] for k in ("height","width"))
    gb=np.memmap(p/"colortex8.bin","<u4","r",shape=(h,w,4))[::-1]
    depth=np.memmap(p/"colortex10.bin","<f4","r",shape=(h,w))[::-1]
    masks=[pixels(p,receipt["resolution"],k)[2] for k in (1,2)]
    eligible=masks[0]&masks[1]
    _,v,l=load_tree(p/"before_tree.bin",6)
    tree=Tree(native_node(v,l,1))
    return p,tree,model,proj,h,w,gb,depth,eligible


def sample_pixels(mask,region=None):
    if region is not None:
        positions=np.argwhere(mask&region)
        return positions[np.linspace(0,len(positions)-1,128,dtype=int)]
    h,w=mask.shape
    y,x=np.meshgrid(np.linspace(0,h-1,25,dtype=int),np.linspace(0,w-1,37,dtype=int),indexing="ij")
    out=[]
    for iy,ix in np.ndindex((24,36)):
        y0,y1=y[iy,ix],y[iy+1,ix]
        x0,x1=x[iy,ix],x[iy,ix+1]
        ys,xs=np.nonzero(mask[y0:y1,x0:x1])
        if len(ys):
            near=np.argmin((ys-(y1-y0)*.5)**2+(xs-(x1-x0)*.5)**2)
            out.append((ys[near]+y0,xs[near]+x0))
    return np.array(out)


def measure(data,pos,points):
    p,tree,model,proj,h,w,gb,depth,eligible=data
    bounds=tree.nodes
    lo,hi=bounds[:,:3],bounds[:,4:7]
    half=(hi-lo)*.5
    broad=((lo//16)!=((hi-1)//16)).any(1)&(tree.parents>=0)
    counts=[]
    for y,x in points:
        d=float(depth[y,x])
        coord=np.array([(2*(x+.5)/w-1.)*-d*proj[0,0],(1.-2*(y+.5)/h)*-d*proj[1,1],d,1.])
        key=int(gb[y,x,0])&65535
        xy=np.maximum(-1.,np.array([key&255,key>>8],dtype="u1").view("i1").astype(float)/127.)
        normal=np.r_[xy,1.-abs(xy).sum()]
        normal[:2]+=np.where(normal[:2]>=0.,-1.,1.)*max(0.,-normal[2])
        normal/=np.linalg.norm(normal)
        origin=(model@coord)[:3]+pos+.003*normal
        tests=np.flatnonzero(broad&(origin>=lo).all(1)&(origin<=hi).all(1))
        extent=half[tests]@abs(normal)
        axial=(lo[tests]+half[tests]-origin)@normal
        nodes=tests[(axial+extent<.25*extent)&(axial+extent>=-1e-6)]
        if len(nodes):
            q=selective_near(tree,origin,normal)
            expect=float(q[tree.parents[nodes]].sum())
            nonempty=float(sum(q[tree.parents[k]]*len(tree.children[k]) for k in nodes))
        else:
            expect=nonempty=0.
        counts.append([len(nodes),expect,nonempty,4.*expect])
    a=np.array(counts)
    return dict(points=len(points),eligible_gates_any=float((a[:,0]>0).mean()),
                expected_expansions_per_ray_mean=float(a[:,1].mean()),
                expected_expansions_p50_p95_max=np.quantile(a[:,1],[.5,.95,1.]).tolist(),
                expected_extra_loads_nonempty_mean=float(a[:,2].mean()),
                expected_extra_loads_conservative_four_slots_mean=float(a[:,3].mean()))


def main():
    wall=read(RUN/"ls5-wall-mode3-fixture.json")["camera_position"]
    motion=read(RUN/"deterministic-quality/light-sampling-failure-moving-4/uv-camera-diagonal-stationary/report.json")
    moving=motion["methods"]["uv-camera"]["pose"]["first_position"]
    data={}
    for scene,pos in (("light-sampling-5",wall),("light-sampling-failure-moving-4",moving)):
        d=prepare(scene)
        mask=d[-1]
        scope=measure(d,np.array(pos),sample_pixels(mask))
        data[scene]=dict(camera_position=pos,case_sha256=sha(d[0]/"accepted.json"),screen=scope)
        if scene=="light-sampling-5":
            h,w=mask.shape
            y,x=np.indices((h,w))
            poly=Polygon([(25,225),(175,245),(175,360),(25,360)])
            region=poly.contains_points(np.c_[x.ravel(),y.ravel()]).reshape(mask.shape)
            data[scene]["sandstone_polygon"]=measure(d,np.array(pos),sample_pixels(mask,region))
    result=dict(results=data,
        scope="CPU proxy: geometry from accepted camera payload, sampled screen receivers and expected visits under replayed proposals. No GPU cost, register, divergence or image claim. Zero-count gate cases still have the predicate cost.")
    (RUN/"selective-near-gate-workload-proxy.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(data))


if __name__=="__main__":
    main()
