#include "/util/Coords.glsl"
#include "/util/Math.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba8) uniform writeonly restrict image2D uimg_rgba8;

// Shared memory with padding for 3x3 tap
// Each work group is 16x16, need +2 padding on each side for 3x3 taps
shared float shared_edgeMask[18][18];

uvec2 groupOriginTexelPos = gl_WorkGroupID.xy << 4u;

void loadSharedData(uint index) {
    if (index < 324) { // 18 * 18 = 324
        uvec2 sharedXY = uvec2(index % 18, index / 18);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 1;
        srcXY = clamp(srcXY, ivec2(0), ivec2(uval_mainImageSize - 1));

        float edgeMask = transient_edgeMaskTemp_fetch(srcXY).r;
        shared_edgeMask[sharedXY.y][sharedXY.x] = edgeMask;
    }
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

    // Load shared data using flattened index (18*18 = 324 elements, 256 threads)
    loadSharedData(gl_LocalInvocationIndex);
    loadSharedData(gl_LocalInvocationIndex + 256);
    barrier();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        // Local position in shared memory (with +1 offset for padding)
        ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 1;

        float minEdgeMask = 1.0;

        // Apply 3x3 min kernel
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                ivec2 sampleLocalPos = localPos + ivec2(dx, dy);
                float sampleEdgeMask = shared_edgeMask[sampleLocalPos.y][sampleLocalPos.x];
                minEdgeMask = min(minEdgeMask, sampleEdgeMask);
            }
        }

        transient_edgeMask_store(texelPos, vec4(minEdgeMask));
    }
}

