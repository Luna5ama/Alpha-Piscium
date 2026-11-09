"""Build the compact scene-by-scene native baseline ledger from accepted receipts."""
import argparse
import json

from light_sampling_exploration_capture import RUN, compare
from light_sampling_point_moments import read,sha


DEFAULT=("camera","prefix","cosine-brdf")
MOVING="light-sampling-failure-moving-"
COHORTS=[
    ("light-sampling-1","standard",("camera-paired-outer","prefix-paired-coarse","cosine-brdf")),
    ("light-sampling-2","camera-prefix",("camera-paired-prefix","prefix-paired-camera")),
    ("light-sampling-2","camera-brdf",("camera-paired-brdf","cosine-brdf-paired")),
    *((f"light-sampling-{i}","standard",DEFAULT) for i in range(3,6)),
    *((MOVING+str(i),"standard",DEFAULT) for i in range(1,4)),
    (MOVING+"4","standard",("camera","prefix-paired-anchor","cosine-brdf")),
    *((f"flatroom-light-sampling-{i}-720p","standard",DEFAULT) for i in (1,2)),
]
NAMES=dict(camera="Camera grouping + A + Near/Far",prefix="World Prefix + A + Near/Far",
           cosine="Cosine BRDF-only")


def main(output):
    data=[]
    for scene,cohort,methods in COHORTS:
        path=RUN/(scene+"-"+"-".join(methods)+".json")
        if path.is_file():
            r=read(path)
            assert set(r["cases"])==set(methods)
        else:
            r=compare(scene,list(methods))
        accepted=[]
        for method in methods:
            source=r["cases"][method]
            t=source["timings"]
            k=method.split("-")[0]
            whole=[]
            for window in r["windows"]:
                q=window["whole"][method]
                whole.append({k:q[k] for k in ("frames","mean_rgb","mean","p05","p50","p95","p99",
                                                "worst05_mean","spatial_std","zero_mean_fraction","variance_times_mean_ms")})
            accepted.append(dict(id=method,name=NAMES[k],source=source["source"],job=source["job"],
                                 initial_mean_ms=t["initial"]["mean_ns"]/1e6,
                                 initial_median_ms=t["initial"]["p50_ns"]/1e6,
                                 build_p50_sum_ms=t["build_p50_sum_ns"]/1e6,
                                 mean_time_ratio=r["mean_time_ratios"][method],
                                 whole_windows=whole))
        data.append(dict(scene=scene,cohort=cohort,world_sha256=r["cases"][methods[0]]["identity"]["world_voxel_sha256"],
                         match_pixels=r["joint_complete_pixels"],resolution=r["resolution"],
                         emitting_uv_matched=r.get("emitting_uv_matched"),
                         source_report=path.relative_to(RUN).as_posix(),source_report_sha256=sha(path),
                         methods=accepted))
    result=dict(scope="Native Iris direct diffuse initial only, one 1 spp frame plus two disjoint 32f means. Two separate LS2 voxel cohorts. Per-scene common complete receiver mask, no cross-scene aggregation; build medians are separate. The ledger does not claim complete emissive texture identity.",
                baseline="Camera grouping + Half-space A + Near/Far",qualification="Static image/timing controls; candidate timing requires paired/reversed native runs.",
                presets=len({x["scene"] for x in data}),cohorts=data)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(dict(presets=result["presets"],cohorts=len(data))))


if __name__=="__main__":
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    main(p.parse_args().output)
