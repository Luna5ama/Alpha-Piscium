// 64-Tree Propagator — Pass 1 of 2 for grid=64 (shadowcomp4).
// Disabled for grid=16/32; those use a single pass (shadowcomp3).
//
// Builds Level 3 from Level 2.
// Dispatch: 4096 workgroups × 64 threads — one workgroup per L3 node,
// one thread per L2 child.  Shared-memory atomics reduce the 64 children
// into a single 64-bit mask.
// Pass 2 (shadowcomp5) then builds L4 → L5.

#define VOXEL_TREE_DATA_MODIFIER buffer
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 64) in;
// One workgroup per L3 node — 16^3 = 4096 L3 nodes for a 64-brick grid
const ivec3 workGroups = ivec3(4096, 1, 1);

shared uint shLo;
shared uint shHi;

void main() {
    uint tid    = gl_LocalInvocationID.x;  // L2 child index within this L3 node (0..63)
    uint l3Node = gl_WorkGroupID.x;        // L3 node index (0..4095)

    if (tid == 0u) {
        shLo = 0u;
        shHi = 0u;
    }
    barrier();

    // Each thread reads one L2 child and atomically sets its bit
    uvec2 child = voxel_tree[uint(VOXEL_TREE_OFFSET_L2) + l3Node * 64u + tid];
    if ((child.x | child.y) != 0u) {
        if (tid < 32u) atomicOr(shLo, 1u << tid);
        else           atomicOr(shHi, 1u << (tid - 32u));
    }
    barrier();

    if (tid == 0u) {
        voxel_tree[uint(VOXEL_TREE_OFFSET_L3) + l3Node] = uvec2(shLo, shHi);
    }
}

