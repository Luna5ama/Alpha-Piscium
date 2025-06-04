#version 460 compatibility

#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable

#include "/atmosphere/Common.glsl"
#include "/general/Lighting.glsl"
#include "/util/Morton.glsl"
#include "/util/Hash.glsl"
#include "/util/FullScreenComp.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform usampler2D usam_gbufferData32UI;
uniform sampler2D usam_gbufferData8UN;
uniform sampler2D usam_gbufferViewZ;

layout(rgba8) uniform writeonly image2D uimg_temp5;

vec2 texel2Screen(ivec2 texelPos) {
    return (vec2(texelPos) + 0.5) * global_mainImageSizeRcp;
}

float searchBlocker(vec3 shadowTexCoord) {
    #define BLOCKER_SEARCH_LOD SETTING_PCSS_BLOCKER_SEARCH_LOD
    #define BLOCKER_SEARCH_N SETTING_PCSS_BLOCKER_SEARCH_COUNT

    float blockerSearchRange = 0.1;
    uint idxB = frameCounter * BLOCKER_SEARCH_N + (hash_31_q3(floatBitsToUint(lighting_viewCoord.xyz)) & 1023u);

    float blockerDepth = 0.0f;
    int n = 0;

    shadowTexCoord.z = rtwsm_linearDepth(shadowTexCoord.z);
    shadowTexCoord.z += 0.5;
    float originalZ = shadowTexCoord.z;
    shadowTexCoord.z = rtwsm_linearDepthInverse(shadowTexCoord.z);

    for (int i = 0; i < BLOCKER_SEARCH_N; i++) {
        vec2 randomOffset = (rand_r2Seq2(idxB) * 2.0 - 1.0);
        vec3 sampleTexCoord = shadowTexCoord;
        sampleTexCoord.xy += randomOffset * blockerSearchRange * vec2(shadowProjection[0][0], shadowProjection[1][1]);
        sampleTexCoord.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, sampleTexCoord.xy);
        float depth = rtwsm_sampleShadowDepth(shadowtex1, sampleTexCoord, BLOCKER_SEARCH_LOD).r;
        bool isBlocker = sampleTexCoord.z > depth;
        blockerDepth += float(isBlocker) * depth;
        n += int(isBlocker);
        idxB++;
    }
    blockerDepth /= float(max(n, 1));
    blockerDepth = mix(shadowTexCoord.z, blockerDepth, float(n != 0));

    return rtwsm_linearDepth(blockerDepth) - originalZ;
}

vec3 calcShadow(Material material, bool isHand) {
    float sssFactor = material.sss;
    uint skipFlag = uint(dot(gData.normal, uval_upDirView) < -0.99);
    skipFlag &= uint(sssFactor < 0.001);
    if (bool(skipFlag)) {
        return vec3(1.0);
    }

    vec3 viewCoord = lighting_viewCoord;
    float distnaceSq = dot(viewCoord, viewCoord);

    float normalOffset = 0.03;

    float viewNormalDot = 1.0 - abs(dot(gData.normal, lighting_viewDir));
    #define NORMAL_OFFSET_DISTANCE_FACTOR1 2048.0
    float normalOffset1 = 1.0 - (NORMAL_OFFSET_DISTANCE_FACTOR1 / (NORMAL_OFFSET_DISTANCE_FACTOR1 + distnaceSq));
    normalOffset += saturate(normalOffset1 * viewNormalDot) * 0.2;

    float lightNormalDot = 1.0 - abs(dot(uval_shadowLightDirView, gData.normal));
    #define NORMAL_OFFSET_DISTANCE_FACTOR2 512.0
    float normalOffset2 = 1.0 - (NORMAL_OFFSET_DISTANCE_FACTOR2 / (NORMAL_OFFSET_DISTANCE_FACTOR2 + distnaceSq));
    normalOffset += saturate(normalOffset2 * lightNormalDot) * 0.2;

    viewCoord = mix(viewCoord + gData.normal * normalOffset, viewCoord, isHand);

    vec4 worldCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);

    vec4 shadowTexCoordCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * worldCoord;
    shadowTexCoordCS /= shadowTexCoordCS.w;

    vec3 shadowTexCoord = shadowTexCoordCS.xyz * 0.5 + 0.5;
    float blockerDistance = searchBlocker(shadowTexCoord);

    shadowTexCoord.z = rtwsm_linearDepth(shadowTexCoord.z);

    float ssRange = 0.0;
    #if SETTING_PCSS_BPF > 0
    ssRange += exp2(SETTING_PCSS_BPF - 10.0);
    ssRange = mix(ssRange, ssRange + 0.05, isHand);
    #endif
    ssRange += SUN_ANGULAR_RADIUS * 2.0 * SETTING_PCSS_VPF * blockerDistance;
    ssRange = saturate(ssRange);
    ssRange += sssFactor * SETTING_SSS_DIFFUSE_RANGE;

    const float ssRangeMul = 0.5;

    ssRange *= ssRangeMul;

    vec3 shadow = vec3(0.0);
    uint idxSS = 0;

    #define DEPTH_BIAS_DISTANCE_FACTOR 1024.0
    float dbfDistanceCoeff = (DEPTH_BIAS_DISTANCE_FACTOR / (DEPTH_BIAS_DISTANCE_FACTOR + max(distnaceSq, 1.0)));
    float depthBiasFactor = 0.001 + lightNormalDot * 0.001;
    depthBiasFactor += mix(0.005 + lightNormalDot * 0.005, -0.001, dbfDistanceCoeff);

    float sampleCountMul = sqrt(ssRange);
    uint sampleCount = 1u;

    for (int i = 0; i < sampleCount; i++) {
        ivec2 r2Offset = ivec2(rand_r2Seq2(idxSS) * vec2(128, 128));

        float jitterR = rand_stbnVec1(lighting_texelPos + r2Offset, frameCounter);
        vec2 dir = rand_stbnUnitVec211(lighting_texelPos + r2Offset, frameCounter);
        float sqrtJitterR = sqrt(jitterR);
        float r = sqrtJitterR * ssRange;

        vec3 sampleTexCoord = shadowTexCoord;
        sampleTexCoord.xy += r * dir * vec2(shadowProjection[0][0], shadowProjection[1][1]);

        sampleTexCoord.z += jitterR * min(sssFactor * SETTING_SSS_DEPTH_RANGE, SETTING_SSS_MAX_DEPTH_RANGE);
        sampleTexCoord.z = rtwsm_linearDepthInverse(sampleTexCoord.z);
        vec2 texelSize;
        sampleTexCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_imap, sampleTexCoord.xy, texelSize);
        float depthBias = SHADOW_MAP_SIZE.y * depthBiasFactor / length(texelSize);
        depthBias = min(depthBias, 0.001);
        sampleTexCoord.z -= depthBias;

        float sampleShadow0 = rtwsm_sampleShadowDepth(shadowtex0HW, sampleTexCoord, 0.0);
        float sampleShadow1 = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
        vec4 sampleColor = rtwsm_sampleShadowColor(shadowcolor0, sampleTexCoord.xy, 0.0);
        sampleColor.rgb = mix(vec3(1.0), sampleColor.rgb * sampleColor.rgb, float(sampleShadow0 < 1.0));

        shadow += min(sampleColor.rgb, sampleShadow1.rrr);
        idxSS++;
    }
    shadow /= float(sampleCount);

    float shadowRangeBlend = linearStep(shadowDistance - 8.0, shadowDistance, length(worldCoord.xz));
    return mix(vec3(shadow), vec3(1.0), shadowRangeBlend);
}

vec4 compShadow(ivec2 texelPos, float viewZ) {
    vec2 screenPos = texel2Screen(texelPos);
    gbufferData1_unpack(texelFetch(usam_gbufferData32UI, texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferData8UN, texelPos, 0), gData);
    Material material = material_decode(gData);
    lighting_init(coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse), texelPos);
    return vec4(calcShadow(material, gData.isHand), 1.0);
}

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;

    ivec2 texelPos = ivec2(mortonGlobalPosU);
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    vec4 outputColor = compShadow(texelPos, viewZ);
    imageStore(uimg_temp5, texelPos, outputColor);
}