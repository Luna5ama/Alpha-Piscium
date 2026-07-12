#include "Morton.glsl"

layout(std430, binding = 7) readonly buffer ThreadGroupTilingData {
    uvec2 ssbo_threadGroupTiling[];
};

bool threadGroupTiling_isWorkGroupValid(uint workGroupIdx) {
    uvec2 numGroups = uvec2((uval_mainImageSizeI + 15) / 16);
    return workGroupIdx < numGroups.x * numGroups.y;
}
