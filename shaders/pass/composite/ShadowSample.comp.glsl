#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_clustered : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#define HIZ_SUBGROUP_CHECK a
#define GLOBAL_DATA_MODIFIER buffer

#include "/techniques/atmospherics/water/Constants.glsl"
#include "/util/Celestial.glsl"
#include "/util/Material.glsl"
#include "/util/Morton.glsl"
#include "/util/Hash.glsl"
#include "/util/Rand.glsl"
#include "/util/GBufferData.glsl"
#include "/techniques/HiZ.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#ifdef DISTANT_HORIZONS
uniform sampler2D dhDepthTex0;
#endif

layout(rgba16f) uniform restrict image2D uimg_temp3;
layout(r32i) uniform iimage2D uimg_rtwsm_imap;
layout(rgba16f) uniform restrict image2D uimg_translucentColor;

#include "/techniques/rtwsm/Backward.glsl"

ivec2 texelPos = ivec2(0);
GBufferData gData = gbufferData_init();
vec3 viewPos = vec3(0.0);
vec3 viewDir = vec3(0.0);

vec2 texel2Screen(ivec2 texelPos) {
    return (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
}

float searchBlocker(vec3 shadowTexCoord) {
    #define BLOCKER_SEARCH_LOD SETTING_PCSS_BLOCKER_SEARCH_LOD
    #define BLOCKER_SEARCH_N SETTING_PCSS_BLOCKER_SEARCH_COUNT

    float blockerSearchRange = 0.1;
    uint idxB = frameCounter * BLOCKER_SEARCH_N + (hash_31_q3(floatBitsToUint(viewPos.xyz)) & 1023u);

    float blockerDepth = 0.0;
    int n = 0;

    for (int i = 0; i < BLOCKER_SEARCH_N; i++) {
        vec2 randomOffset = (rand_r2Seq2(idxB) * 2.0 - 1.0);
        vec3 sampleTexCoord = shadowTexCoord;
        sampleTexCoord.xy += randomOffset * blockerSearchRange * vec2(global_shadowProjPrev[0][0], global_shadowProjPrev[1][1]);
        sampleTexCoord.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, sampleTexCoord.xy);
        float depth = rtwsm_sampleShadowDepth(shadowtex1, sampleTexCoord, BLOCKER_SEARCH_LOD).r;
        bool isBlocker = sampleTexCoord.z > depth;
        blockerDepth += float(isBlocker) * depth;
        n += int(isBlocker);
        idxB++;
    }
    blockerDepth /= float(max(n, 1));
    blockerDepth = mix(shadowTexCoord.z, blockerDepth, float(n != 0));

    return abs(rtwsm_linearDepth(blockerDepth) - rtwsm_linearDepth(shadowTexCoord.z));
}

float waterSurfaceDistance(vec3 shadowUVPos) {
    shadowUVPos.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, shadowUVPos.xy);
    vec2 ndcCoord = shadowUVPos.xy * 2.0 - 1.0;
    float edgeCoord = max(abs(ndcCoord.x), abs(ndcCoord.y));
    if (edgeCoord > 1.0 - SHADOW_MAP_SIZE.y * 16) {
        return -1.0;
    }
    if (texture(usam_shadow_waterMask, shadowUVPos.xy).r < 0.9) {
        return -1.0;
    }
    float sampleDepth = texture(shadowtex0, shadowUVPos.xy).r;
    if (texture(shadowtex1, shadowUVPos.xy).r <= sampleDepth) {
        return -1.0;
    }
    return abs(rtwsm_linearDepth(shadowUVPos.z) - rtwsm_linearDepth(sampleDepth));
}

vec3 calcShadow(Material material) {
    float sssFactor = material.sss;
    uint skipFlag = uint(dot(gData.normal, uval_upDirView) < -0.99);
    skipFlag &= uint(sssFactor < 0.001);
    if (bool(skipFlag)) {
        return vec3(0.0);
    }

    float cosLightTheta = dot(uval_shadowLightDirView, gData.geomNormal);
    float sideFacingFactor = 1.0 - abs(cosLightTheta);

    vec3 offsetViewPos = viewPos;
    offsetViewPos += gData.geomNormal * mix(0.01, 0.03, sideFacingFactor);
    vec4 scenePos = gbufferModelViewInverse * vec4(offsetViewPos, 1.0);
    vec4 shadowViewPos = global_shadowRotationMatrix * global_shadowView * scenePos;
    vec4 shadowClipPos = global_shadowProjPrev * shadowViewPos;
    vec3 shadowNDCPos = shadowClipPos.xyz / shadowClipPos.w;
    vec3 shadowScreenPos = shadowNDCPos * 0.5 + 0.5;
    float blockerDistance = searchBlocker(shadowScreenPos);

    float ssRange = 0.0;
    #if SETTING_PCSS_BPF > 0
    ssRange += exp2(SETTING_PCSS_BPF - 10.0);
    ssRange = mix(ssRange, ssRange + 0.05, gData.isHand);
    #endif
    ssRange += SUN_ANGULAR_RADIUS * 2.0 * SETTING_PCSS_VPF * blockerDistance;
    ssRange = saturate(ssRange);
    ssRange += sssFactor * SETTING_SSS_DIFFUSE_RANGE;

    const float ssRangeMul = 0.5;
    ssRange *= ssRangeMul;

    float jitterR = rand_stbnVec1(texelPos, frameCounter);
    vec2 dir = rand_stbnUnitVec211(texelPos, frameCounter);
    float sqrtJitterR = sqrt(jitterR);
    float r = sqrtJitterR * ssRange;

    vec3 sampleTexCoord = shadowScreenPos;

    sampleTexCoord.xy += r * dir * vec2(global_shadowProjPrev[0][0], global_shadowProjPrev[1][1]);

    vec2 texelSize;
    sampleTexCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_imap, sampleTexCoord.xy, texelSize);

    float sampleShadowDepthOffset = rtwsm_sampleShadowColor(shadowcolor0, sampleTexCoord.xy, 0.0).x;
    sampleTexCoord.z -= max(sampleShadowDepthOffset, 0.0);

    vec3 worldGeometryNormal = mat3(gbufferModelViewInverse) * gData.geomNormal;
    vec3 sampleNormal = rtwsm_sampleShadowColor(shadowcolor1, sampleTexCoord.xy, 0.0).xyz;
    float shadowNormalCos = dot(worldGeometryNormal, sampleNormal);
    float confidance = pow2(shadowNormalCos * 0.5 + 0.5);
    confidance = subgroupClusteredMax(confidance, 16);

    float depthBiasFactor = 2.0;
    depthBiasFactor += (1.0 - pow2(1.0 - sideFacingFactor)) * 8.0;
    depthBiasFactor *= confidance;
    float depthBiasConstant = 0.02;
    depthBiasConstant += sideFacingFactor * 0.03;
    depthBiasConstant *= pow2(confidance);
    float depthBias = SHADOW_MAP_SIZE.y * depthBiasFactor / min(texelSize.x, texelSize.y);
    sampleTexCoord.z = rtwsm_linearDepth(sampleTexCoord.z);
    sampleTexCoord.z -= jitterR * min(sssFactor * SETTING_SSS_DEPTH_RANGE, SETTING_SSS_MAX_DEPTH_RANGE);
    sampleTexCoord.z -= depthBias;
    sampleTexCoord.z -= depthBiasConstant;
    sampleTexCoord.z = rtwsm_linearDepthInverse(sampleTexCoord.z);

    float sampleShadow0 = rtwsm_sampleShadowDepth(shadowtex0HW, sampleTexCoord, 0.0);
    float sampleShadow1 = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
    vec4 sampleColor = rtwsm_sampleShadowColor(shadowcolor2, sampleTexCoord.xy, 0.0);
    sampleColor.rgb = colors2_material_toWorkSpace(sampleColor.rgb);
    sampleColor.rgb = mix(vec3(1.0), sampleColor.rgb, float(sampleShadow0 < 1.0));

    vec3 shadow = min(sampleColor.rgb, sampleShadow1.rrr);

    if (texture(usam_shadow_waterMask, sampleTexCoord.xy).r > 0.9) {
        float rcpShadowY = rcp(uval_shadowLightDirWorld.y);
        vec4 scenePos = gbufferModelViewInverse * vec4(viewPos, 1.0);
        float worldHeight = scenePos.y + cameraPosition.y;
        float waterDepth = waterSurfaceDistance(shadowScreenPos);
        if (waterDepth == -1.0) {
            max(63.0 - worldHeight, 0.0) * rcpShadowY;
        }
        #ifdef SETTING_WATER_CAUSTICS
        ivec2 readPos = texelPos;
        readPos.y += uval_mainImageSizeI.y;
        float causticsV = texelFetch(usam_causticsPhoton, readPos, 0).r;
        shadow *= mix(1.0, causticsV, exp2(-waterDepth * 0.05));
        #endif
        shadow *= exp(-waterDepth * WATER_EXTINCTION);
    }

    float shadowRangeBlend = linearStep(shadowDistance - 8.0, shadowDistance, length(scenePos.xz));
    return mix(vec3(shadow), vec3(1.0), shadowRangeBlend);
}

vec4 compShadow(ivec2 texelPos, float viewZ) {
    vec2 screenPos = texel2Screen(texelPos);
    Material material = material_decode(gData);
    viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    viewDir = normalize(-viewPos);
    return vec4(calcShadow(material), 1.0);
}

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float viewZ = -65536.0;

        if (hiz_groupGroundCheckSubgroup(gl_WorkGroupID.xy, 4)) {
            viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
        }

        if (viewZ != -65536.0) {
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            vec4 outputColor = compShadow(texelPos, viewZ);
            outputColor = clamp(outputColor, 0.0, FP16_MAX);
            imageStore(uimg_temp3, texelPos, outputColor);

            rtwsm_backward(texelPos, viewZ, gData);
        }
    }
}
