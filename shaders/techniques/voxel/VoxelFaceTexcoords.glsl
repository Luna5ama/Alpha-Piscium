#ifndef INCLUDE_techniques_voxel_VoxelFaceTexcoords_glsl
#define INCLUDE_techniques_voxel_VoxelFaceTexcoords_glsl

// ---------------------------------------------------------------------------
// Per-material per-face atlas texcoord storage
//
// Layout: 2048 materials × 6 faces, each entry is vec4(minU, minV, maxU, maxV).
// Written by ShadowPass vertex shader during voxelization (binding = 9).
// Read by any pass that needs to sample the block colour atlas by material+face.
//
// Face index convention (matches voxel_faceIndexFromNormal):
//   0 = +X,  1 = -X,  2 = +Y,  3 = -Y,  4 = +Z,  5 = -Z
// ---------------------------------------------------------------------------

#define VOXEL_FACE_TEXCOORD_MATERIALS 2048
#define VOXEL_FACE_TEXCOORD_COUNT (VOXEL_FACE_TEXCOORD_MATERIALS * 6)

#ifndef VOXEL_FACE_TEXCOORD_MODIFIER
#define VOXEL_FACE_TEXCOORD_MODIFIER restrict readonly buffer
#endif

layout(std430, binding = 9) VOXEL_FACE_TEXCOORD_MODIFIER VoxelFaceTexcoordData {
    vec4 voxel_faceTexcoords[]; // VOXEL_FACE_TEXCOORD_COUNT entries
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

#endif // INCLUDE_techniques_voxel_VoxelFaceTexcoords_glsl

