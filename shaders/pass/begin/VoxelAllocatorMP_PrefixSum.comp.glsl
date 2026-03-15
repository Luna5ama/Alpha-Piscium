// Multi-pass Brick Allocator — Pass 3/4: Prefix Sum
//
// Converts voxel_bucketCounts[] from per-bucket brick counts (written by
// the Remap pass) into exclusive prefix sums (starting alloc IDs per bucket)
// using a 2-level subgroup-based scan matching the GetWarp.comp.glsl pattern.
// Also resets voxel_brickAllocCounter to 0 for the Assign pass.
//
// Single workgroup of NUM_DIST_BUCKETS = 1024 threads.
// Only active when SETTING_VOXEL_GRID_SIZE > 16 (begin9.csh).

#extension GL_KHR_shader_subgroup_arithmetic : enable

#define VOXEL_BRICK_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 1024) in;
const ivec3 workGroups = ivec3(1, 1, 1);

// One slot per subgroup. With local_size_x=1024 and minimum subgroup size=32,
// we have at most 32 subgroups.
shared uint shared_prefixBuffer[32];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tValue = voxel_bucketCounts[tid];

    // Level 1: inclusive prefix within each subgroup
    uint prefix = subgroupInclusiveAdd(tValue);
    if (gl_SubgroupInvocationID == gl_SubgroupSize - 1) {
        shared_prefixBuffer[gl_SubgroupID] = prefix;
    }
    barrier();

    // Level 2: all threads load their subgroup's total; subgroup 0 scans them
    // (matches GetWarp.comp.glsl pattern)
    uint tValue2 = shared_prefixBuffer[gl_LocalInvocationID.x];
    barrier();
    if (gl_SubgroupID == 0) {
        uint prefix2 = subgroupInclusiveAdd(tValue2);
        shared_prefixBuffer[gl_LocalInvocationID.x] = prefix2;
    }
    barrier();

    // Combine: add inclusive sum of all previous subgroups, then subtract own
    // value to convert from inclusive to exclusive prefix
    prefix += (gl_SubgroupID == 0) ? 0u : shared_prefixBuffer[gl_SubgroupID - 1];
    voxel_bucketCounts[tid] = prefix - tValue;

    if (tid == 0u) {
        voxel_brickAllocCounter = 0u;
    }
}

