// Initialize the face atlas table and directional model bounds when the shader pack loads.
#include "/Base.glsl"
#define VOXEL_FACE_TEXCOORD_MODIFIER buffer
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(768, 1, 1);

void main() {
    uint index = gl_GlobalInvocationID.x;
    voxel_faceTexcoords[index] = uvec2(0);
    uint material = index >> 3u;
    if (material >= uint(min(VOXEL_FACE_TEXCOORD_MATERIALS, textureSize(usam_pbrLUT2, 0).y))) return;
    uint octant = index & 7u;
    uint faceMask = 42u - (octant & 1u) - ((octant & 2u) << 1u) - ((octant & 4u) << 2u);
    uint model = texelFetch(usam_pbrLUT2, ivec2(faceMask, material), 0).x;
    if ((model & 0x80000000u) == 0u) return;

    ivec2 quad = ivec2(bitfieldExtract(model, 9, 14) * 2u, bitfieldExtract(model, 23, 2));
    uint count = bitfieldExtract(model, 25, 6);
    vec3 boundsMin = vec3(1.0);
    vec3 boundsMax = vec3(0.0);
    for (uint i = 0u; i < count; ++i) {
        vec4 centerAxis = texelFetch(usam_blockModelQuads, quad, 0);
        vec2 halfSize = texelFetch(usam_blockModelQuads, quad + ivec2(1, 0), 0).zw;
        vec3 extent = centerAxis.w < 253.5 / 255.0 ? vec3(0.0, halfSize) :
            (centerAxis.w < 254.5 / 255.0 ? vec3(halfSize.x, 0.0, halfSize.y) : vec3(halfSize, 0.0));
        boundsMin = min(boundsMin, centerAxis.xyz - extent);
        boundsMax = max(boundsMax, centerAxis.xyz + extent);
        quad.x += 2;
    }

    // Round outwards, including the tolerance used by the quad intersection.
    uvec3 lo = uvec3(clamp(floor((boundsMin - 1e-5) * 31.0), 0.0, 31.0));
    uvec3 hi = uvec3(clamp(ceil((boundsMax + 1e-5) * 31.0), 0.0, 31.0));
    voxel_modelBounds[index] = lo.x | (lo.y << 5u) | (lo.z << 10u) |
        (hi.x << 15u) | (hi.y << 20u) | (hi.z << 25u);
}
