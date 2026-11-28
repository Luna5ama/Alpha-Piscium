#ifndef INCLUDE_techniques_voxel_VoxelFaceTexcoords_glsl
#define INCLUDE_techniques_voxel_VoxelFaceTexcoords_glsl a

// ---------------------------------------------------------------------------
// Per-material per-face atlas texcoord storage
//
// Layout: 16384 materials × 6 faces, each entry is vec4(minU, minV, maxU, maxV).
// Written by ShadowPass vertex shader during voxelization (binding = 3).
// Read by any pass that needs to sample the block colour atlas by material+face.
//
// Face index convention (matches voxel_faceIndexFromNormal):
//   0 = +X,  1 = -X,  2 = +Y,  3 = -Y,  4 = +Z,  5 = -Z
// ---------------------------------------------------------------------------

#define VOXEL_FACE_TEXCOORD_MATERIALS 16384
#define VOXEL_FACE_TEXCOORD_COUNT (VOXEL_FACE_TEXCOORD_MATERIALS * 6)

#ifndef VOXEL_FACE_TEXCOORD_MODIFIER
#define VOXEL_FACE_TEXCOORD_MODIFIER restrict readonly buffer
#endif

layout(std430, binding = 3) VOXEL_FACE_TEXCOORD_MODIFIER VoxelFaceTexcoordData {
    // Each face has packed UV bounds, a sprite selection key and the float bits of its alpha threshold.
    uvec2 voxel_faceTexcoords[VOXEL_FACE_TEXCOORD_COUNT];
    uint voxel_faceKeys[VOXEL_FACE_TEXCOORD_COUNT];
    uint voxel_faceAlpha[VOXEL_FACE_TEXCOORD_COUNT];
    // Local model boxes for each direction octant. The top two bits of x select the split quarter (zero for one box).
    // z is the directional block-model header. The first six w lanes cache the pre-shadow face alpha bits.
    uvec4 voxel_modelBounds[VOXEL_FACE_TEXCOORD_MATERIALS * 8];
};

// Flat index into voxel_faceTexcoords[].
uint voxel_faceTexcoordIndex(uint materialID, uint faceIdx) {
    return materialID * 6u + faceIdx;
}
// Map a world-space surface normal to a face index 0..5.
// (+X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5)
uint voxel_faceIndexFromNormal(vec3 worldNormal) {
    vec3 a = abs(worldNormal);
    if (a.x >= a.y && a.x >= a.z) {
        return worldNormal.x >= 0.0 ? 0u : 1u;
    } else if (a.y >= a.z) {
        return worldNormal.y >= 0.0 ? 2u : 3u;
    } else {
        return worldNormal.z >= 0.0 ? 4u : 5u;
    }
}

vec2 voxel_faceLocalUV(uint faceIdx, vec3 hitPos) {
    vec3 f = fract(hitPos);
    uint faceAxis = faceIdx >> 1u;
    bool positiveFace = (faceIdx & 1u) == 0u;
    if (faceAxis == 0u) {
        return vec2(positiveFace ? 1.0 - f.z : f.z, f.y);
    } else if (faceAxis == 1u) {
        return vec2(f.x, positiveFace ? 1.0 - f.z : f.z);
    }
    return vec2(positiveFace ? f.x : 1.0 - f.x, f.y);
}

#endif // INCLUDE_techniques_voxel_VoxelFaceTexcoords_glsl
