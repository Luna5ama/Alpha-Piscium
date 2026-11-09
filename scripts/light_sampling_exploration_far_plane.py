"""Independent geometry/PDF screen for the far-cube plane proposal."""
import argparse
import itertools
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.spatial import ConvexHull

from light_sampling_geometry import cube_integral, cube_pdf
from light_sampling_point_moments import read


SHADER = "shaders/techniques/debug/LightSamplingCube.glsl"
CORNER = ((np.arange(8)[:, None] >> np.arange(3)) & 1)
RANDOM_ENDPOINTS = np.array(list(itertools.product([0., 2**-24, 0.5, 1.-2**-24], repeat=2)))


def table(source):
    text = subprocess.check_output(["git", "show", f"{source}:{SHADER}"], text=True)
    src = text.split("ls_cubeSilhouette[27] = uint[27](", 1)[1].split(");", 1)[0]
    words = [int(x) for x in re.findall(r"(\d+)u", src)]
    assert len(words) == 27
    return words


def geometry(low, origin, normal, words):
    lo = (low - origin).astype(np.float32)
    hi = (low + 1. - origin).astype(np.float32)
    support = np.clip(np.zeros(3, np.float32), lo, hi)
    d2 = support @ support
    front = normal.astype(np.float32) @ np.where(normal >= 0., lo, hi)
    if not (d2 >= 64. and front > 0. and front*front > 0.04*d2):
        return None
    axis = int(np.argmax(abs(support)))
    perm = (np.arange(3) + axis + 1) % 3
    plane = support[axis]
    zone = int(((lo <= 0.).astype(int)+(hi < 0.)) @ [1,3,9])
    word = words[zone]
    p = np.where(CORNER, hi, lo)[:, perm]
    points = p[:, :2] * (plane / p[:, 2, None])
    outline = points[[(word >> (3*i)) & 7 for i in range(word >> 18)]]
    fan = outline[1:-1]-outline[0], outline[2:]-outline[0]
    area2 = abs((fan[0][:,0].astype(float)*fan[1][:,1]
               - (fan[0][:,1]*fan[1][:,0]).astype(np.float32)).astype(np.float32))
    total = np.cumsum(area2, dtype=np.float32)[-1]
    assert total > 0
    hull = ConvexHull(points.astype(float))
    area_error = abs(float(total)/(2.*hull.volume)-1.)
    return axis,perm,plane,outline,fan,area2,total,hull,area_error,zone


def draw(case, uv):
    axis,perm,plane,outline,fan,a,total,hull,_,_ = case
    target = (uv[:,0]*total).astype(np.float32)
    u = np.ones(len(uv),np.float32)
    tri = np.zeros(len(uv),dtype=int)
    active = np.ones(len(uv),dtype=bool)
    for i, area in enumerate(a):
        if area <= 0:
            continue
        tri[active] = i
        select = active & (target <= area)
        u[select] = np.maximum(0.,target[select]/area)
        active &= ~select
        target[active] -= area
    xy = outline[0]+np.sqrt(u)[:,None]*((1-uv[:,1,None])*fan[0][tri]+uv[:,1,None]*fan[1][tri])
    p = np.column_stack((xy,np.full(len(xy),plane,np.float32))).astype(np.float32)
    d2 = np.einsum("ij,ij->i",p,p)
    plane_dir = p/np.sqrt(d2[:,None])
    d = plane_dir[:,np.argsort(perm)].astype(float)
    pdf = 2.*d2/(total*abs(plane_dir[:,2]))
    # Independent inverse Jacobian: ray meets the plane at t=plane/d[axis].
    expected = (plane/d[:,axis])**2/(hull.volume*abs(d[:,axis]))
    assert np.isfinite(pdf).all() and (pdf > 0).all()
    return d,pdf,abs(pdf/expected-1.)


def evaluate(case, low, origin, normal, uv):
    d,pdf,error = draw(case,uv)
    with np.errstate(divide="ignore",invalid="ignore"):
        a = (low-origin)/d
        b = (low+1.-origin)/d
    # Exact parallel-axis rays inside their slab impose no constraint.
    inside = (low<=origin)&(origin<=low+1.)
    parallel = d==0
    enter = np.where(parallel,np.where(inside,-np.inf,np.inf),np.minimum(a,b)).max(axis=1)
    exit_ = np.where(parallel,np.where(inside,np.inf,-np.inf),np.maximum(a,b)).min(axis=1)
    gap = (enter-exit_)/np.maximum.reduce([abs(enter),abs(exit_),np.full(len(enter),1e-12)])
    assert np.isfinite(gap).all()
    cosine = d@normal
    assert (cosine > 0).all() and (exit_ > 0).all()
    return error,gap,cosine/pdf


def main(args):
    words = table(args.source)
    rng = np.random.default_rng(20261109)
    low = np.zeros(3)
    fixtures = []
    for zones in itertools.product((-1,0,1),repeat=3):
        if not any(zones):
            continue
        for _ in range(24):
            origin = np.array([rng.uniform(-220.,-8.) if z<0 else rng.uniform(9.,221.) if z else rng.uniform(0.,1.)
                               for z in zones])
            n = 0.5-origin
            n /= np.linalg.norm(n)
            n += rng.normal(0,.28,3)
            n /= np.linalg.norm(n)
            fixtures.append((low,origin,n))
    for _ in range(768):
        low2 = rng.integers(0,257,size=3).astype(float)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        origin = low2+0.5+direction*np.exp(rng.uniform(np.log(7.),np.log(216.)))
        n = -direction + rng.normal(0,.55,3)
        n /= np.linalg.norm(n)
        fixtures.append((low2,origin,n))
    for axis in range(3):
        for x in (0.,1.):
            origin = np.full(3,0.5)
            origin[axis] = -24.
            origin[(axis+1)%3] = x
            origin[(axis+2)%3] = 1.-x
            n = np.eye(3)[axis]
            fixtures.append((low,origin,n))

    witness = read(args.witness)["records"]
    witness_summary = []
    for item in witness:
        l,o,n = (np.array(item[k]) for k in ("cube_low","origin","normal"))
        g = geometry(l,o,n,words)
        value = dict(pixel_gl=item["pixel_gl"],ellipse_gap=item.get("relative_gap"),plane_eligible=g is not None)
        if g is not None:
            _,gap,_ = evaluate(g,l,o,n,np.array([item["randoms"]],dtype=np.float32))
            value["plane_gap"] = float(gap[0])
        witness_summary.append(value)

    areas,errors,gaps,cv,means = [],[],[],[],[]
    zones = set()
    draws = 0
    endpoint_gaps = []
    for l,o,n in fixtures:
        o,n = o.astype(np.float32).astype(float),n.astype(np.float32).astype(float)
        g = geometry(l,o,n,words)
        if g is None:
            continue
        zones.add(g[-1])
        samples = rng.random((256,2),dtype=np.float32)
        uv = np.vstack([samples,RANDOM_ENDPOINTS]).astype(np.float32)
        err,gap,w = evaluate(g,l,o,n,uv)
        h = cube_integral(l,o,n)
        assert h > 0.
        areas.append(g[-2])
        errors.append(float(err.max()))
        gaps.append(float(gap[:len(samples)].max()))
        endpoint_gaps.append(float(gap[len(samples):].max()))
        cv.append(float(np.var(w[:len(samples)])/h**2))
        means.append(abs(float(w[:len(samples)].mean())/h-1.))
        draws += len(uv)
    assert len(zones) == 26 and len(areas)>1000
    assert max(areas) < 3e-4 and max(errors) < 4e-4
    assert max(gaps) < 1e-6 and max(endpoint_gaps) < 2e-5
    quant = lambda x: np.quantile(x,[.5,.95,1]).tolist()
    result = dict(source=args.source,fixtures=len(fixtures),eligible=len(areas),zones=len(zones),draws=draws,
                  area_error_max=max(areas),pdf_relative_error_max=max(errors),random_gap_max=max(gaps),
                  endpoint_gap_max=max(endpoint_gaps),geometry_only_normalized_variance_p50_p95_max=quant(cv),
                  mean_relative_mc_error_p50_p95_max=quant(means),witnesses=witness_summary,
                  scope="Float32 plane draws with an independent hull, ray-plane Jacobian, double slabs and unit diffuse projected integral. CPU geometry only; visibility, textures, GPU error and speed remain untested.")
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="witnesses"}))


def live(folder, output):
    case = read(folder/"accepted.json")
    assert case["restore"]["status"] == "RECEIPT_STATUS_OK"
    data = np.fromfile(folder/"audit.bin","<u4")
    assert len(data) >= 1040
    counters = data[1024:1040].tolist()
    source = case["source"]
    words = table(source)
    errors = {"far_plane":[],"legacy":[]}
    for o,n,d,l in data[:1024].view("<f4").reshape(64,4,4).astype(float):
        if not (d[3] > 0.):
            continue
        assert np.isfinite(d).all() and abs(np.linalg.norm(d[:3])-1.) < 2e-5
        direction = d[:3]
        if l[3] == 2.:
            geom = geometry(l[:3],o[:3],n[:3],words)
            assert geom is not None
            axis,_,plane,_,_,_,_,hull,_,_ = geom
            expected = (plane/direction[axis])**2/(hull.volume*abs(direction[axis]))
            tag = "far_plane"
        else:
            expected = cube_pdf(l[:3].copy(),o[:3],n[:3],direction)
            tag = "legacy"
        assert expected > 0.
        errors[tag].append(abs(d[3]/expected-1.))
    evidence = {name:dict(samples=len(v),max_relative_pdf_error=max(v) if v else None)
                for name,v in errors.items()}
    qualified = bool(errors["far_plane"]) and max(errors["far_plane"]+errors["legacy"]) < .005
    qualified &= not any(counters[i] for i in (10,11,12))
    report = dict(source=source,job=case["job"],scene=case["scene"],counters=counters,verified=evidence,
                  qualified=qualified,scope="Native debug frame; independently validate sampled conditional PDFs and require zero full-frame far-plane numerical/support failures.")
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(report))
    if not qualified:
        raise SystemExit(1)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode",required=True)
    cpu=sub.add_parser("cpu")
    cpu.add_argument("source")
    cpu.add_argument("witness",type=Path)
    cpu.add_argument("output",type=Path)
    native=sub.add_parser("live")
    native.add_argument("folder",type=Path)
    native.add_argument("output",type=Path)
    args=p.parse_args()
    if args.mode=="live":
        live(args.folder,args.output)
    else:
        main(args)
