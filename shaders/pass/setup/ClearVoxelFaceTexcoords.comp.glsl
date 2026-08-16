// Clears SSBO 9 (per-material per-face atlas texcoords) every frame so the
// shadow voxelization pass repopulates it from scratch.
//
// Dispatch: 768 workgroups x 256 threads = 196608 entries.

#define VOXEL_FACE_TEXCOORD_MODIFIER buffer
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(768, 1, 1);

void main() {
    voxel_faceTexcoords[gl_GlobalInvocationID.x] = uvec2(0);
}
