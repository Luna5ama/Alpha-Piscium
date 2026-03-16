// Clears the 3D voxel tree image (uimg_voxelTree) every frame.
//
// Uses a 3D dispatch: workgroup (x, y, z) covers the full image
// (VOXEL_TREE_XY x VOXEL_TREE_XY x VOXEL_TREE_Z_TOTAL).
// Each 16x16x1 workgroup clears a 16x16 slice of one Z layer.
// Each invocation clears exactly one texel.
//
// Grid=16: ivec3( 4,  4,  85) workgroups ->  64x 64x 85 texels
// Grid=32: ivec3( 8,  8, 171) workgroups -> 128x128x171 texels
// Grid=64: ivec3(16, 16, 341) workgroups -> 256x256x341 texels

#define VOXEL_TREE_IMG_QUALIFIER restrict writeonly
#define VOXEL_TREE_WRITE_ONLY
#include "/techniques/voxel/Voxelization.glsl"

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

#if VOXEL_GRID_SIZE == 64
const ivec3 workGroups = ivec3(16, 16, 341);
#elif VOXEL_GRID_SIZE == 32
const ivec3 workGroups = ivec3(8, 8, 171);
#else // VOXEL_GRID_SIZE == 16
const ivec3 workGroups = ivec3(4, 4, 85);
#endif

void main() {
    imageStore(uimg_voxelTree, ivec3(gl_GlobalInvocationID), uvec4(0u));
}

