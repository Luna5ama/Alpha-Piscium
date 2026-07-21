// VoxelHit.glsl
// Result of a voxel ray trace.

#ifndef INCLUDE_techniques_voxel_VoxelHit_glsl
#define INCLUDE_techniques_voxel_VoxelHit_glsl a

struct VoxelHit {
    bool hit;
    uint materialID;   // material written by the shadow pass; 0 = miss
    vec3 hitPos;       // world-space entry point of the hit block
    vec3 normal;       // outward face normal of the hit surface
#if VOXEL_TRACE_DEBUG_COUNTERS
    ivec4 debugCounters;
#endif
};

#endif // INCLUDE_techniques_voxel_VoxelHit_glsl
