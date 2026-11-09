"""Type-check the actual changed GLSL snippets without touching the Iris runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


INITIAL="shaders/pass/composite/LightSamplingInitial.comp.glsl"
SELECT="shaders/techniques/debug/LightSampling.glsl"
CUBE="shaders/techniques/debug/LightSamplingCube.glsl"
POLYGON="shaders/techniques/debug/LightSamplingPolygon.glsl"


def src(rev,path):
    return subprocess.check_output(["git","show",f"{rev}:{path}"],text=True)


COMMON=r"""
layout(local_size_x=16,local_size_y=16) in;
struct LightSamplingNode { vec4 boundsMinMass; vec4 boundsMaxLink; };
layout(std430,binding=12) buffer LightNodes { LightSamplingNode ls_nodes[]; };
layout(std430,binding=14) buffer Audit { vec4 ls_audit[256]; uint ls_reasons[16]; };
const uint LS_LEAF_BIT=0x80000000u;
const uint LS_INVALID=0xffffffffu;
uniform int frameCounter;
uniform ivec2 texelPos;
uniform ivec2 uval_mainImageSizeI;
uint hash_pcg_11(uint v) {
    uint s=v*747796405u+2891336453u;
    uint w=((s>>((s>>28u)+4u))^s)*277803737u;
    return (w>>22u)^w;
}
float ls_random(inout uint s) {
    s=hash_pcg_11(s);
    return (float(s>>9u)+0.5)*(1.0/8388608.0);
}
LightSamplingNode ls_emptyNode() { return LightSamplingNode(vec4(0.0),vec4(0.0)); }
"""

SELECT_STUB=r"""
#define LS_TREE_ARITY 4
#define VOXEL_GRID_SIZE 16
#define SETTING_DEBUG_LIGHT_WEIGHT_MODEL 6
#define LS_NEAR_FAR 1
#define SETTING_DEBUG_LIGHT_AUDIT 1
uint ls_childBase(uint link) { return link&0x0fffffffu; }
uint ls_childCount(uint link) { return ((link>>28u)&7u)+1u; }
vec3 ls_nativeScores(LightSamplingNode n, vec3 o, vec3 normal) { return n.boundsMinMass.www; }
void runSelect();
"""

SELECT_MAIN=r"""
void main() {
    uint s=uint(frameCounter);
    float p;
    LightSamplingNode n=ls_select(vec3(0),vec3(0,1,0),s,0u,p);
    ls_audit[0]=vec4(n.boundsMinMass.xyz,p);
}
"""

MODE3_STUB=r"""
#define SETTING_DEBUG_LIGHT_SAMPLING 3
#define COLORS_WORKING_COLORSPACE 0
#define COLORS_COLORSPACES_SRGB 0
layout(rgba32f) uniform writeonly image2D uimg_lightSampling;
uniform isampler2D usam_lightSamplingReference;
uniform uint ls_emitterCount;
uniform uint ls_overflowCount;
void voxel_initShared() {}
uint voxel_getMaterialID(ivec3 position) { return uint(position.x); }
bool ls_isEmitter(uint materialID) { return materialID==99u; }
vec3 nzpacking_unpackNormalOct32(uint bits) { return vec3(0,0,1); }
vec3 colors_material_toWorkSpace(vec3 v) { return v; }
vec3 colors_colorspaces_convert(int x,int y,vec3 v) { return v; }
vec3 ls_evaluate(vec3 o, vec3 n, vec3 a, inout uint rng, uint slot, out uint reason,
                 LightSamplingNode e) { reason=0u; return a; }
"""


def select(rev,gate=False,defines=""):
    s=src(rev,SELECT)
    if gate:
        i=s.index("#if SETTING_DEBUG_LIGHT_WEIGHT_MODEL == 6\nvec3 ls_resolveFalseNearContainment")
    else:
        i=s.find("#ifdef LS_RESIDUAL_SELECT\ndouble ls_drawWide")
        if i<0:
            i=s.index("LightSamplingNode ls_select")
    s=s[i:].removesuffix("\n#endif\n")
    return COMMON+SELECT_STUB+defines+s+SELECT_MAIN


def far(rev):
    cube=src(rev,CUBE).replace('#include "/techniques/debug/LightSamplingPolygon.glsl"',src(rev,POLYGON))
    i=src(rev,INITIAL)
    g=i[i.index("    // A plane sample at a projected silhouette edge"):i.index("    target = origin + direction * nearT;")]
    stub="float nearT=0.0, farT=1.0; \n"
    # Use the code that follows the actual existing slab interval calculation.
    fragment=r"""
#define SETTING_DEBUG_LIGHT_CUBE 1
vec3 runSample() {
    vec3 d;
    float p;
    ls_sampleCube(LightSamplingNode(vec4(1),vec4(2)),vec3(0),vec3(1,0,0),vec2(0.3,0.7),d,p);
""" + stub + g + """
    return d*p;
}
void main() { ls_audit[0]=vec4(runSample(),1.0); }
"""
    return COMMON+cube+fragment


def mode3(rev):
    a=src(rev,INITIAL)
    start=a.index("#if SETTING_DEBUG_LIGHT_SAMPLING == 3\nvoid main()")
    end=a.index("#else\nvoid main()",start)
    return COMMON+MODE3_STUB+a[start:end]+"\n#endif\n"


def deterministic(rev):
    a=src(rev,INITIAL)
    start=a.index("layout(rgba32f) uniform image2D uimg_lsEvidenceSum;")
    end=a.index("\n}",a.index("void ls_recordEvidenceWindow",start))+2
    globals=r"""
uniform mat4 global_camProj,global_camProjInverse,gbufferModelViewInverse;
uniform vec3 cameraPosition;
uniform usampler2D usam_gbufferSolidData1,usam_gbufferSolidData2;
uniform sampler2D usam_gbufferSolidViewZ;
"""
    return COMMON+globals+a[start:end]+"\nvoid main(){ ls_recordEvidenceWindow(texelPos,vec4(1)); }\n"


def main(output):
    exe=shutil.which("glslangValidator")
    assert exe
    definitions=[
        ("anchor-near-gate","e3cb50a1ff0c4edc4b5650943e707554d2e4989f",lambda r:select(r,True)),
        ("far-cube","43bdb20bf8a050acbd1800a8f2522d0fa4bfd2be",far),
        ("far-cube-entry-faces","661cf8539f612fa00de9ba3be633c3e6ddda628c",far),
        ("stratified-f32","f9fdbcf836a9ecb1c9f0ff8199b7ee25d2ac0d76",lambda r:select(r,defines="#define LS_RESIDUAL_F32\n#define LS_RESIDUAL_STRATA\n")),
        ("stratified-f64","16b13a4808a41e22c0d6423ff6e1c2760999412e",lambda r:select(r,defines="#define LS_RESIDUAL_SELECT\n#define LS_RESIDUAL_STRATA\n")),
        ("fixed-ls5-receiver","d3d0e978579a54a1ed9ae758ccda84559262c331",mode3),
        ("deterministic-receiver-and-sum","7498a20f1fc0c6f20491015c922a0be661823130",deterministic),
    ]
    cases=[]
    with tempfile.TemporaryDirectory(prefix="light-sampling-glsl-") as td:
        for name,rev,make in definitions:
            path=Path(td)/f"{name}.comp"
            glsl="#version 460 compatibility\n"+make(rev)
            path.write_text(glsl,encoding="utf8")
            compiled=subprocess.run([exe,"-S","comp",str(path)],capture_output=True,text=True)
            cases.append(dict(case=name,source=rev,exit_code=compiled.returncode,
                              diagnostics=(compiled.stdout+compiled.stderr).replace(str(path),path.name).strip(),
                              harness_sha256=hashlib.sha256(glsl.encode()).hexdigest()))
    result=dict(cases=cases,compiler=subprocess.check_output([exe,"--version"],text=True).splitlines()[0],
                scope="Type-check extracted actual changes against minimal declared dependencies. This is not a full Iris pack compile, GPU execution, or performance result.")
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(cases))
    if any(r["exit_code"]!=0 for r in cases):
        raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    main(p.parse_args().output)
