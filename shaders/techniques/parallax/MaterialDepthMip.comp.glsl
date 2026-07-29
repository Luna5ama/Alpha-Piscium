#include "/Base.glsl"
#include "/techniques/parallax/Common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(MATERIAL_DEPTH_MIP_WORK_GROUPS, MATERIAL_DEPTH_MIP_WORK_GROUPS, 1);

uniform sampler2D usam_blocksNormal;
layout(r8) uniform restrict image2D uimg_materialDepthMip;

float loadMaterialDepth(ivec2 texelPos) {
#if MATERIAL_DEPTH_MIP_LEVEL == 1
    return texelFetch(usam_blocksNormal, texelPos, 0).a;
#else
    ivec2 baseSize = textureSize(usam_blocksNormal, 0);
    return imageLoad(uimg_materialDepthMip, parallax_mipPackedOffset(baseSize, MATERIAL_DEPTH_MIP_LEVEL - 1) + texelPos).r;
#endif
}

void main() {
    ivec2 baseSize = textureSize(usam_blocksNormal, 0);
    ivec2 outputSize = parallax_mipPackedSize(baseSize, MATERIAL_DEPTH_MIP_LEVEL);
    ivec2 outputTexel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(outputTexel, outputSize))) {
        return;
    }

    ivec2 sourceSize = parallax_mipPackedSize(baseSize, MATERIAL_DEPTH_MIP_LEVEL - 1);
    float maxDepth = 0.0;
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            ivec2 sourceTexel = min(outputTexel * 2 + ivec2(x, y), sourceSize - 1);
            maxDepth = max(maxDepth, loadMaterialDepth(sourceTexel));
        }
    }

    ivec2 storeTexel = parallax_mipPackedOffset(baseSize, MATERIAL_DEPTH_MIP_LEVEL) + outputTexel;
    if (all(lessThan(storeTexel, imageSize(uimg_materialDepthMip)))) {
        imageStore(uimg_materialDepthMip, storeTexel, vec4(maxDepth));
    }
}
