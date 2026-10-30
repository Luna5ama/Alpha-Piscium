"""Syntax-check the wide-tree GLSL bodies with external pipeline inputs stubbed."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'builds/light-sampling/20261030-arity-default'


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    header_path = ROOT / 'shaders/techniques/debug/LightSampling.glsl'
    builder_path = ROOT / 'shaders/pass/prepare/LightSamplingBuild.comp.glsl'
    header = header_path.read_text()
    header = '\n'.join(line for line in header.splitlines() if not line.startswith('#include'))
    start, end = header.index('bool ls_loadReceiver('), header.index('float ls_geometricWeight(')
    header = header[:start] + header[end:]
    stub = '''
#define SETTING_VOXEL_POOL_SIZE 1024
#define SETTING_EMISSIVE_PBR_VALUE_CURVE 1.0
#define VOXEL_GRID_SIZE SETTING_VOXEL_GRID_SIZE
#define VOXEL_GRID_BRICKS (VOXEL_GRID_SIZE * VOXEL_GRID_SIZE * VOXEL_GRID_SIZE)
layout(std430, binding=0) buffer VoxelInput {
    uint voxel_brickAllocCounter;
    uint voxel_allocatedBricks[1024];
    uint voxel_materials[];
};
uniform ivec3 cameraPositionInt;
struct HardcodedPBR { float emissive; int emissiveMultiplier; };
HardcodedPBR hardcodedpbr_decode(uint value) { return HardcodedPBR(1.0, 0); }
uint voxel_decodeMaterialID(uint value) { return value; }
uvec3 morton3D_30bDecode(uint value) { return uvec3(value); }
uint hash_pcg_11(uint value) { return value; }
'''
    builder = builder_path.read_text().replace('#include "/techniques/debug/LightSampling.glsl"', '')
    checks = []
    for bits in (1, 2, 3):
        variant = f'#define SETTING_DEBUG_LIGHT_ARITY {1 << bits}\n' + header
        for grid in (16, 32, 64):
            for stage, level in [(0, 0), (1, 0)] + [(2, n) for n in range(1, 3 * (grid.bit_length() - 1) + 1)]:
                defines = f'#version 450\n#define SETTING_DEBUG_LIGHT_AUDIT\n#define SETTING_VOXEL_GRID_SIZE {grid}\n#define LS_BUILD_STAGE {stage}\n#define LS_REDUCE_LEVEL {level}\n#define LS_REDUCE_GROUPS 1\n'
                source = defines + stub + variant + '\n' + builder
                result = subprocess.run(['glslangValidator', '--stdin', '-S', 'comp'], input=source, text=True, capture_output=True)
                if result.returncode:
                    (BUILD / 'failed.comp').write_text(source)
                    raise RuntimeError(f'bits={bits}, grid={grid}, stage={stage}, level={level}\n{result.stdout}\n{result.stderr}')
                checks.append(dict(arity=1 << bits, grid=grid, stage=stage, level=level))
    result = dict(scope='Static GLSL validation with stubbed external inputs, not target Iris or runtime evidence', checked=len(checks), sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (header_path, builder_path)}, checks=checks)
    (BUILD / 'syntax-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Passed {len(checks)} stage/arity/grid combinations')


if __name__ == '__main__':
    main()
