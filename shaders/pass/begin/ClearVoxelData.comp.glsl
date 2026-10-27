// Setup clears the full pool; begin clears every slot written by the previous frame.
#ifdef VOXEL_CLEAR_ALL
#define VOXEL_BRICK_DATA_MODIFIER buffer
#endif
#define VOXEL_MATERIAL_DATA_MODIFIER buffer
#define VOXEL_MATERIAL_VEC4 a
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(1024, 1, 1);

void main() {
    #ifdef VOXEL_CLEAR_ALL
    uint clearCount = uint(VOXEL_POOL_SIZE) * 1024u;
    if (gl_GlobalInvocationID.x == 0u) {
        voxel_brickAllocCounter = 0u;
    }
    #else
    uint clearCount = voxel_brickAllocCounter * 1024u;
    #endif
    for (uint index = gl_GlobalInvocationID.x; index < clearCount; index += 262144u) {
        voxel_materials_v4[index] = uvec4(0u);
    }
}
