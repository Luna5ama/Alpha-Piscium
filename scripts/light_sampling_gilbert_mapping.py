"""Check the actual GLSL integer mapping against both pinned upstream implementations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'builds/light-sampling/20261030-gilbert'
REFERENCE = BUILD / 'reference'
REVISION = '9b080a74c5e3b6fe189785c52742f83ac85ba181'


def load(name):
    path = REFERENCE / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    # The C++ shim supplies GLSL integer vector operators; the function itself is
    # included verbatim, so validation exercises the maintained shader source.
    shader = ROOT / 'shaders/techniques/debug/LightSamplingGilbert.glsl'
    source = r'''
#include <cstdio>
#include <cstdlib>
using uint = unsigned int;
struct ivec3 {
    int x, y, z;
    ivec3() = default;
    explicit ivec3(int n) : x(n), y(n), z(n) {}
    ivec3(int x, int y, int z) : x(x), y(y), z(z) {}
    ivec3& operator+=(ivec3 b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
};
ivec3 operator+(ivec3 a, ivec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
ivec3 operator-(ivec3 a, ivec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
ivec3 operator-(ivec3 a) { return {-a.x,-a.y,-a.z}; }
ivec3 operator*(ivec3 a, int n) { return {a.x*n,a.y*n,a.z*n}; }
ivec3 operator>>(ivec3 a, int n) { return {a.x>>n,a.y>>n,a.z>>n}; }
int sgn(int n) { return (n>0)-(n<0); }
ivec3 sign(ivec3 a) { return {sgn(a.x),sgn(a.y),sgn(a.z)}; }
'''
    source += f'#include "{shader.as_posix()}"\n'
    source += r'''
int main() {
    for (uint i=0; i<4096; ++i) {
        ivec3 p = ls_gilbertPosition(i);
        std::printf("%d %d %d\n",p.x,p.y,p.z);
    }
}
'''
    cpp, exe = BUILD / 'mapping.cpp', BUILD / 'mapping.exe'
    cpp.write_text(source)
    subprocess.run(['clang++', '-std=c++20', '-O2', str(cpp), '-o', str(exe)], check=True)
    actual = [tuple(map(int, line.split())) for line in subprocess.check_output([str(exe)], text=True, timeout=10).splitlines()]
    random_access = load('gilbert_d2xyz').gilbert_d2xyz
    expected = list(load('gilbert3d').gilbert3d(16, 16, 16))
    assert len(actual) == len(set(actual)) == 4096
    assert all(0 <= c < 16 for p in actual for c in p)
    assert actual == expected
    assert actual == [random_access(i, 16, 16, 16) for i in range(4096)]
    steps = [sum(abs(a-b) for a,b in zip(p,q)) for p,q in zip(actual,actual[1:])]
    assert max(steps) == 1
    overlap = {}
    for size in (2,4,8,16,32,64,128,256,512,1024,2048,4096):
        count = 0
        for start in range(0, 4096, size):
            left, right = actual[start:start+size//2], actual[start+size//2:start+size]
            if all(max(p[k] for p in left)+1 > min(p[k] for p in right) and max(p[k] for p in right)+1 > min(p[k] for p in left) for k in range(3)):
                count += 1
        overlap[str(size)] = count
    result = dict(upstream_revision=REVISION, source_sha256=hashlib.sha256(shader.read_bytes()).hexdigest(), checked_positions=4096, bijective=True, max_manhattan_step=max(steps), upstream_sequential_match=True, upstream_random_access_match=True, overlapping_sibling_aabbs_by_interval_size=overlap, scope='CPU execution of actual integer GLSL body; not Iris/GPU validation')
    (BUILD/'mapping-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
