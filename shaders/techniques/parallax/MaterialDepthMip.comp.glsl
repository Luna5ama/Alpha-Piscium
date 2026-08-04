#define GLOBAL_DATA_MODIFIER buffer
#include "/Base.glsl"
#include "/techniques/parallax/Common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(MATERIAL_DEPTH_MIP_WORK_GROUPS, MATERIAL_DEPTH_MIP_WORK_GROUPS, 1);

uniform sampler2D usam_blocksNormal;
layout(r8) uniform restrict image2D uimg_materialDepthMip;

void main() {
    ivec2 baseSize = textureSize(usam_blocksNormal, 0);
    ivec2 outputSize = parallax_mipPackedSize(baseSize, MATERIAL_DEPTH_MIP_LEVEL);
    ivec2 outputTexel = ivec2(gl_GlobalInvocationID.xy);
#if MATERIAL_DEPTH_MIP_LEVEL == 0
    if (all(equal(gl_GlobalInvocationID.xy, uvec2(0)))) {
        for (int level = 0; level < 15; level++) {
            global_parallaxMipPackedData[level] = ivec4(
                parallax_mipPackedSize(baseSize, level),
                parallax_mipPackedOffset(baseSize, level)
            );
        }
    }
#endif
    if (any(greaterThanEqual(outputTexel, outputSize))) {
        return;
    }

    float maxDepth;
#if MATERIAL_DEPTH_MIP_LEVEL == 0
    maxDepth = texelFetch(usam_blocksNormal, outputTexel, 0).a;
#else
    ivec2 sourceSize = parallax_mipPackedSize(baseSize, MATERIAL_DEPTH_MIP_LEVEL - 1);
    maxDepth = 0.0;
#if SETTING_PARALLAX_MODE == 4
#if MATERIAL_DEPTH_MIP_LEVEL == 1
    for (int y = -2; y <= 3; y++) {
        for (int x = -2; x <= 3; x++) {
#else
    for (int y = -1; y <= 2; y++) {
        for (int x = -1; x <= 2; x++) {
#endif
#elif SETTING_PARALLAX_MODE > 1
#if MATERIAL_DEPTH_MIP_LEVEL == 1
    for (int y = 0; y <= 3; y++) {
        for (int x = 0; x <= 3; x++) {
#else
    for (int y = 0; y <= 2; y++) {
        for (int x = 0; x <= 2; x++) {
#endif
#else
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
#endif
            ivec2 sourceTexel = clamp(outputTexel * 2 + ivec2(x, y), ivec2(0), sourceSize - 1);
            ivec2 sourceStoreTexel = parallax_mipPackedOffset(baseSize, MATERIAL_DEPTH_MIP_LEVEL - 1) + sourceTexel;
            maxDepth = max(maxDepth, imageLoad(uimg_materialDepthMip, sourceStoreTexel).r);
        }
    }
#endif

    ivec2 storeTexel = parallax_mipPackedOffset(baseSize, MATERIAL_DEPTH_MIP_LEVEL) + outputTexel;
    if (all(lessThan(storeTexel, imageSize(uimg_materialDepthMip)))) {
        imageStore(uimg_materialDepthMip, storeTexel, vec4(maxDepth));
    }
}
