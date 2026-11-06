// Initialize the face atlas table and directional model bounds when the shader pack loads.
#include "/Base.glsl"
#define VOXEL_FACE_TEXCOORD_MODIFIER buffer
#include "/techniques/voxel/VoxelFaceTexcoords.glsl"

layout(local_size_x = 256) in;
const ivec3 workGroups = ivec3(768, 1, 1);

struct ModelBounds {
    vec3 lo;
    vec3 hi;
};

ModelBounds mergeBounds(ModelBounds a, ModelBounds b) {
    return ModelBounds(min(a.lo, b.lo), max(a.hi, b.hi));
}

float halfSurfaceArea(ModelBounds bounds) {
    vec3 size = max(bounds.hi - bounds.lo, vec3(0.0));
    return dot(size, size.yzx);
}

uint packBounds(ModelBounds bounds) {
    // Round outwards, including the tolerance used by the quad intersection.
    uvec3 lo = uvec3(clamp(floor((bounds.lo - 1e-5) * 31.0), 0.0, 31.0));
    uvec3 hi = uvec3(clamp(ceil((bounds.hi + 1e-5) * 31.0), 0.0, 31.0));
    return lo.x | (lo.y << 5u) | (lo.z << 10u) |
        (hi.x << 15u) | (hi.y << 20u) | (hi.z << 25u);
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    voxel_faceTexcoords[index] = uvec2(0);
    uint material = index >> 3u;
    if (material >= uint(min(VOXEL_FACE_TEXCOORD_MATERIALS, textureSize(usam_pbrLUT2, 0).y))) return;
    uint octant = index & 7u;
    uint faceMask = 42u - (octant & 1u) - ((octant & 2u) << 1u) - ((octant & 4u) << 2u);
    uint model = texelFetch(usam_pbrLUT2, ivec2(faceMask, material), 0).x;
    if (model == 0u) return;
    bool axisAligned = (model & 0x80000000u) != 0u;

    ivec2 quad = ivec2(bitfieldExtract(model, 9, 11) * 2u, bitfieldExtract(model, 20, 5));
    uint count = bitfieldExtract(model, 25, 6);
    ModelBounds first = ModelBounds(vec3(1.0), vec3(0.0));
    ModelBounds second = first;
    ModelBounds third = first;
    ModelBounds fourth = first;
    for (uint i = 0u; i < count; ++i) {
        vec4 centerAxis = texelFetch(usam_blockModelQuads, quad, 0);
        vec4 normalYZHalfSize = texelFetch(usam_blockModelQuads, quad + ivec2(1, 0), 0);
        vec2 halfSize = normalYZHalfSize.zw;
        vec3 extent;
        if (axisAligned) {
            extent = centerAxis.w < 253.5 / 255.0 ? vec3(0.0, halfSize) :
                (centerAxis.w < 254.5 / 255.0 ? vec3(halfSize.x, 0.0, halfSize.y) : vec3(halfSize, 0.0));
        } else {
            vec3 normal = vec3(centerAxis.w, normalYZHalfSize.xy) * (255.0 / 126.0) - 1.0;
            vec3 axis = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
            vec3 u = cross(axis, normal);
            vec3 v = cross(normal, u);
            // The quad test projects onto orthogonal, unnormalized axes.
            extent = abs(u) * ((halfSize.x + 1e-6) / dot(u, u)) +
                abs(v) * ((halfSize.y + 1e-6) / dot(v, v));
        }
        ModelBounds quadBounds = ModelBounds(centerAxis.xyz - extent, centerAxis.xyz + extent);
        if (i < (count >> 2u)) {
            first = mergeBounds(first, quadBounds);
        } else if (i < (count >> 1u)) {
            second = mergeBounds(second, quadBounds);
        } else if (i < ((count * 3u) >> 2u)) {
            third = mergeBounds(third, quadBounds);
        } else {
            fourth = mergeBounds(fourth, quadBounds);
        }
        quad.x += 2;
    }

    ModelBounds leftHalf = mergeBounds(first, second);
    ModelBounds rightHalf = mergeBounds(third, fourth);
    ModelBounds whole = mergeBounds(leftHalf, rightHalf);
    uvec2 packedBounds = uvec2(packBounds(whole), 0u);
    if (count >= 6u) {
        // A split adds a second box test. Keep it only when the area-weighted quad count falls enough.
        float bestCost = 0.78 * halfSurfaceArea(whole) * float(count);
        for (uint split = 1u; split <= 3u; ++split) {
            ModelBounds left = leftHalf;
            ModelBounds right = rightHalf;
            if (split == 1u) {
                left = first;
                right = mergeBounds(second, rightHalf);
            } else if (split == 3u) {
                left = mergeBounds(leftHalf, third);
                right = fourth;
            }
            uint leftCount = (count * split) >> 2u;
            float cost = halfSurfaceArea(left) * float(leftCount) + halfSurfaceArea(right) * float(count - leftCount);
            if (cost < bestCost) {
                bestCost = cost;
                packedBounds = uvec2(packBounds(left) | (split << 30u), packBounds(right));
            }
        }
    }
    voxel_modelBounds[index] = packedBounds;
}
