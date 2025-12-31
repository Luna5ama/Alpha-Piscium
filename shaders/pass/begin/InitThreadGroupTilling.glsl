#include "/util/Morton.glsl"

layout(std430, binding = 7) writeonly buffer ThreadGroupTilingData {
    uvec2 ssbo_threadGroupTiling[];
};

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0625, 0.0625);

void main() {
    uvec2 actualNumGroups = uvec2((uval_mainImageSizeI + 15) / 16);
    uint actualTotalGroups = actualNumGroups.x * actualNumGroups.y;

    uint globalThreadIdx = gl_GlobalInvocationID.x + (gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    // Linear work group index
    uint workGroupIdx = globalThreadIdx;

    if (workGroupIdx >= actualTotalGroups) {
        ssbo_threadGroupTiling[workGroupIdx] = uvec2(0xFFFFFFFFu, 0xFFFFFFFFu);
        return;
    }

    // Find the smallest power of 2 that covers both dimensions
    uvec2 size = actualNumGroups;
    uint maxDim = max(size.x, size.y);
    uint pot = 1u;
    while (pot < maxDim) pot <<= 1u;

    // Map linear index to Z-order coordinate, skipping out-of-bounds positions
    // We need to find the nth valid Z-order position where n = workGroupIdx
    uvec2 swizzledWGCoord;

    // Binary search / iterative approach to find the morton index that gives us
    // the workGroupIdx-th valid position
    // For efficiency, we iterate through morton codes and count valid ones

    // Since this could be expensive, we use a smarter approach:
    // Recursively traverse the Z-order quadtree, counting valid cells in each quadrant

    // Calculate how many valid work groups exist in a quadrant of given size at given offset
    // Then use this to navigate directly to the target index

    swizzledWGCoord = uvec2(0u);
    uint targetIdx = workGroupIdx;
    uint currentSize = pot;

    while (currentSize > 1u) {
        currentSize >>= 1u;

        // Four quadrants in Z-order: (0,0), (1,0), (0,1), (1,1)
        const uvec2 quadrantOffsets[4] = uvec2[4](
            uvec2(0u, 0u),
            uvec2(currentSize, 0u),
            uvec2(0u, currentSize),
            uvec2(currentSize, currentSize)
        );

        for (int q = 0; q < 4; q++) {
            uvec2 qStart = swizzledWGCoord + quadrantOffsets[q];
            uvec2 qEnd = qStart + uvec2(currentSize);

            // Count valid cells in this quadrant (cells within bounds)
            uvec2 clampedStart = min(qStart, size);
            uvec2 clampedEnd = min(qEnd, size);

            uint validInQuadrant = 0u;
            if (clampedEnd.x > clampedStart.x && clampedEnd.y > clampedStart.y) {
                validInQuadrant = (clampedEnd.x - clampedStart.x) * (clampedEnd.y - clampedStart.y);
            }

            if (targetIdx < validInQuadrant) {
                // Target is in this quadrant, descend into it
                swizzledWGCoord = qStart;
                break;
            } else {
                // Target is not in this quadrant, skip it
                targetIdx -= validInQuadrant;
            }
        }
    }

    ssbo_threadGroupTiling[workGroupIdx] = swizzledWGCoord;
}