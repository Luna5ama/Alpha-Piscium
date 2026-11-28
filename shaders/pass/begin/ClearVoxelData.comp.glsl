// Setup clears the pool. Until the allocator runs, the old tree identifies every written material quad.
#ifdef VOXEL_CLEAR_ALL
#define VOXEL_BRICK_DATA_MODIFIER buffer
#endif
#define VOXEL_MATERIAL_DATA_MODIFIER buffer
#define VOXEL_MATERIAL_VEC4 a
#include "/techniques/voxel/Voxelization.glsl"

#ifndef VOXEL_CLEAR_ALL
#define VOXEL_FACE_TEXCOORD_MODIFIER buffer
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"

layout(std430, binding = 8) restrict readonly buffer VoxelTreeData {
    uint previousTree[];
};
#endif

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1024, 1, 1);

void main() {
    #ifdef VOXEL_CLEAR_ALL
    uint clearCount = uint(VOXEL_POOL_SIZE) * 1024u;
    if (gl_GlobalInvocationID.x == 0u) {
        voxel_brickAllocCounter = 0u;
    }
    for (uint index = gl_GlobalInvocationID.x; index < clearCount; index += 262144u) {
        voxel_materials_v4[index] = uvec4(0u);
    }
    #else
    // The rotating face clear has completed. This lower bound stays valid through every shadow draw.
    uint index = gl_GlobalInvocationID.x;
    uint face = index & 7u;
    if (index < VOXEL_FACE_TEXCOORD_MATERIALS * 8u && face < 6u) {
        uint bound = voxel_faceAlpha[voxel_faceTexcoordIndex(index >> 3u, face)];
        voxel_modelBounds[index].w = bound;
    }

    uint clearCount = voxel_brickAllocCounter * 1024u;
    for (uint index = gl_GlobalInvocationID.x; index < clearCount; index += 262144u) {
        uint brick = voxel_allocatedBricks[index >> 10u];
        uint leaf = (index >> 4u) & 63u;
        uint rootWord = previousTree[(uint(VOXEL_TREE_OFFSET_L2) + brick) * 2u + (leaf >> 5u)];
        if ((rootWord & (1u << (leaf & 31u))) != 0u) {
            uint quad = index & 15u;
            uint word = previousTree[(uint(VOXEL_TREE_OFFSET_L1) + brick * 64u + leaf) * 2u + (quad >> 3u)];
            if ((word & (15u << ((quad & 7u) * 4u))) != 0u) {
                voxel_materials_v4[index] = uvec4(0u);
            }
        }
    }
    #endif
}
