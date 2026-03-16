// VoxelRayState.glsl
// Suspendable/resumable ray state for the hierarchical voxel DDA tracer.
//
// Provides:
//   VoxelRay struct        – all state needed to suspend and resume a ray
//   voxelray_pack / unpack – 2×uvec4 (256-bit) SSBO serialisation
//   voxelray_encodeSortKey – 32-bit direction-priority sort key (22-bit Hilbert + 10-bit Morton)
//
// Must be included AFTER /Base.glsl.
// voxelray_setup() lives in VoxelTrace.glsl (requires shared-memory spread LUT).

#ifndef INCLUDE_techniques_voxel_VoxelRayState_glsl
#define INCLUDE_techniques_voxel_VoxelRayState_glsl

#include "/techniques/voxel/Voxelization.glsl"
#include "/util/NZPacking.glsl"
#include "/util/Coords.glsl"
#include "/util/Morton.glsl"

// ---------------------------------------------------------------------------
// Ray state struct
// ---------------------------------------------------------------------------
struct VoxelRay {
    vec3  worldRayOrigin; // world-space ray origin (stored explicitly — see VoxelTrace.glsl notes)
    vec3  worldRayDir;    // normalized world-space ray direction
    uint  fullMorton;     // current 3-D Morton code of blockPos (≤ 30 bits)
    float lastT;          // parametric t of the last DDA cell-boundary crossing
    int   lastAxis;       // axis of last crossing: 0/1/2  (-1 → stored as 3, maps to z default)
    int   level;          // current tree level: 1 … VOXEL_TREE_TOP_LEVEL; 0 = done sentinel
    uint  callbackData;   // caller-defined write-back ID (e.g. packed texelPos, probe index)
};

VoxelRay voxelray_init() {
    VoxelRay ray;
    ray.worldRayOrigin = vec3(0.0);
    ray.worldRayDir    = vec3(0.0, 1.0, 0.0);
    ray.fullMorton     = 0u;
    ray.lastT          = 0.0;
    ray.lastAxis       = -1;
    ray.level          = 0;
    ray.callbackData   = 0u;
    return ray;
}

// ---------------------------------------------------------------------------
// Pack / unpack  (2 × uvec4, 256 bits per ray)
//
//  p0 – ray geometry
//    .x  floatBitsToUint(worldRayOrigin.x)
//    .y  floatBitsToUint(worldRayOrigin.y)
//    .z  floatBitsToUint(worldRayOrigin.z)
//    .w  nzpacking_packNormalOct32(worldRayDir)
//
//  p1 – DDA state
//    .x  floatBitsToUint(lastT)
//    .y  fullMorton[29:0]  | lastAxis[31:30]
//    .z  callbackData[27:0] | level[31:28]
//    .w  reserved (0)
// ---------------------------------------------------------------------------
void voxelray_pack(VoxelRay ray, out uvec4 p0, out uvec4 p1) {
    p0.x = floatBitsToUint(ray.worldRayOrigin.x);
    p0.y = floatBitsToUint(ray.worldRayOrigin.y);
    p0.z = floatBitsToUint(ray.worldRayOrigin.z);
    p0.w = nzpacking_packNormalOct32(ray.worldRayDir);

    p1.x = floatBitsToUint(ray.lastT);
    p1.y = ray.fullMorton & 0x3FFFFFFFu;                    // 30 bits
    p1.y = bitfieldInsert(p1.y, uint(ray.lastAxis) & 3u, 30, 2);
    p1.z = ray.callbackData & 0x0FFFFFFFu;                  // 28 bits
    p1.z = bitfieldInsert(p1.z, uint(ray.level), 28, 4);
    p1.w = 0u;
}

VoxelRay voxelray_unpack(uvec4 p0, uvec4 p1) {
    VoxelRay ray;
    ray.worldRayOrigin = vec3(uintBitsToFloat(p0.x),
                              uintBitsToFloat(p0.y),
                              uintBitsToFloat(p0.z));
    ray.worldRayDir    = nzpacking_unpackNormalOct32(p0.w);
    ray.lastT          = uintBitsToFloat(p1.x);
    ray.fullMorton     = bitfieldExtract(p1.y,  0, 30);
    ray.lastAxis       = int(bitfieldExtract(p1.y, 30,  2));
    ray.callbackData   = bitfieldExtract(p1.z,  0, 28);
    ray.level          = int(bitfieldExtract(p1.z, 28,  4));
    return ray;
}

// ---------------------------------------------------------------------------
// Sort key  (32 bits: 22-bit Hilbert direction MSB + 10-bit Morton spatial LSB)
//
//   binLocalPos : morton_16bEncode(uvec2(localPos)) for a 32×32 bin, 0..1023
// ---------------------------------------------------------------------------
uint voxelray_encodeSortKey(uint binLocalPos, VoxelRay ray) {
    // Octahedral encode to [0,1]², quantise to 11-bit grid (2×11 = 22 bits)
    vec2  oct     = coords_octEncode01(ray.worldRayDir);
    uvec2 q       = uvec2(clamp(oct * 2048.0, vec2(0.0), vec2(2047.0)));
    uint  hilbert = hilbert2D_encode(q, 11u);
    return (hilbert << 10u) | (binLocalPos & 0x3FFu);
}

uint voxelray_decodeBinLocalIndex(uint sortKey) {
    return sortKey & 0x3FFu;
}

#endif // INCLUDE_techniques_voxel_VoxelRayState_glsl

