#include "../_Util.glsl"
#include "../rtwsm/RTWSM.glsl"

const bool shadowtex0Mipmap = true;
const bool shadowtex1Mipmap = true;

const bool shadowHardwareFiltering0 = true;
uniform sampler2D shadowtex0;
uniform sampler2DShadow shadowtex0HW;

const bool shadowHardwareFiltering1 = true;
uniform sampler2D shadowtex1;
uniform sampler2DShadow shadowtex1HW;

uniform sampler2D shadowcolor0;
uniform sampler2D usam_rtwsm_imap;

uniform sampler2D usam_transmittanceLUT;
uniform sampler2D usam_skyLUT;

uint coord3Rand[2];
ivec2 texelPos;
GBufferData gData;
vec3 g_viewCoord;
vec3 g_viewDir;

float searchBlocker(vec3 shadowTexCoord) {
    #define BLOCKER_SEARCH_LOD SETTING_PCSS_BLOCKER_SEARCH_LOD

    #define BLOCKER_SEARCH_N SETTING_PCSS_BLOCKER_SEARCH_COUNT

    float blockerSearchRange = 0.2;
    uint idxB = frameCounter * BLOCKER_SEARCH_N + coord3Rand[1];

    float blockerDepth = 0.0f;
    int n = 0;

    for (int i = 0; i < BLOCKER_SEARCH_N; i++) {
        vec2 randomOffset = (rand_r2Seq2(idxB) * 2.0 - 1.0);
        vec3 sampleTexCoord = shadowTexCoord;
        sampleTexCoord.xy += randomOffset * blockerSearchRange * vec2(shadowProjection[0][0], shadowProjection[1][1]);
        sampleTexCoord.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, sampleTexCoord.xy);
        float depth = rtwsm_sampleShadowDepth(shadowtex0, sampleTexCoord, BLOCKER_SEARCH_LOD).r;
        bool isBlocker = sampleTexCoord.z > depth;
        blockerDepth += float(isBlocker) * depth;
        n += int(isBlocker);
        idxB++;
    }
    blockerDepth /= float(max(n, 1));
    blockerDepth = mix(shadowTexCoord.z, blockerDepth, float(n != 0));

    return rtwsm_linearDepth(blockerDepth) - rtwsm_linearDepth(shadowTexCoord.z);
}

vec3 calcShadow(float sssFactor) {
    vec3 viewCoord = g_viewCoord;
    float distnaceSq = dot(viewCoord, viewCoord);

    float normalOffset = 0.03;

    float viewNormalDot = 1.0 - abs(dot(gData.normal, g_viewDir));
    #define NORMAL_OFFSET_DISTANCE_FACTOR1 2048.0
    float normalOffset1 = 1.0 - (NORMAL_OFFSET_DISTANCE_FACTOR1 / (NORMAL_OFFSET_DISTANCE_FACTOR1 + distnaceSq));
    normalOffset += saturate(normalOffset1 * viewNormalDot) * 0.2;

    float lightNormalDot = 1.0 - abs(dot(uval_shadowLightDirView, gData.normal));
    #define NORMAL_OFFSET_DISTANCE_FACTOR2 512.0
    float normalOffset2 = 1.0 - (NORMAL_OFFSET_DISTANCE_FACTOR2 / (NORMAL_OFFSET_DISTANCE_FACTOR2 + distnaceSq));
    normalOffset += saturate(normalOffset2 * lightNormalDot) * 0.2;

    viewCoord += gData.normal * normalOffset;

    vec4 worldCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);

    vec4 shadowTexCoordCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * worldCoord;
    shadowTexCoordCS /= shadowTexCoordCS.w;

    vec3 shadowTexCoord = shadowTexCoordCS.xyz * 0.5 + 0.5;
    float blockerDistance = searchBlocker(shadowTexCoord);

    shadowTexCoord.z = rtwsm_linearDepth(shadowTexCoord.z);

    float ssRange = 0.0;
    ssRange += sssFactor * 0.1;
    #if SETTING_PCSS_BPF > 0
    ssRange += exp2(SETTING_PCSS_BPF - 10.0);
    #endif
    ssRange += uval_sunAngularRadius * 2.0 * SETTING_PCSS_VPF * blockerDistance;
    ssRange = saturate(ssRange);
    ssRange *= 0.4;

    #define SAMPLE_N SETTING_PCSS_SAMPLE_COUNT

    vec3 shadow = vec3(0.0);
    uint idxSS = (frameCounter + coord3Rand[0]) * SAMPLE_N;

    #define DEPTH_BIAS_DISTANCE_FACTOR 1024.0
    float dbfDistanceCoeff = (DEPTH_BIAS_DISTANCE_FACTOR / (DEPTH_BIAS_DISTANCE_FACTOR + max(distnaceSq, 1.0)));
    float depthBiasFactor = 0.001 + lightNormalDot * 0.001;
    depthBiasFactor += mix(0.005 + lightNormalDot * 0.005, -0.001, dbfDistanceCoeff);

    for (int i = 0; i < SAMPLE_N; i++) {
        vec3 randomOffset = rand_r2Seq3(idxSS);
        randomOffset.xy = randomOffset.xy * 2.0 - 1.0;
        vec3 sampleTexCoord = shadowTexCoord;
        sampleTexCoord.xy += ssRange * randomOffset.xy * vec2(shadowProjection[0][0], shadowProjection[1][1]);
        sampleTexCoord.z = rtwsm_linearDepthInverse(sampleTexCoord.z + randomOffset.z * sssFactor * 2.0);
        vec2 texelSize;
        sampleTexCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_imap, sampleTexCoord.xy, texelSize);
        float depthBias = SHADOW_MAP_SIZE.y * depthBiasFactor / length(texelSize);
        depthBias = min(depthBias, 0.001);
        sampleTexCoord.z -= depthBias;

        float sampleShadow0 = rtwsm_sampleShadowDepth(shadowtex0HW, sampleTexCoord, 0.0);
        float sampleShadow1 = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
        vec4 sampleColor = rtwsm_sampleShadowColor(shadowcolor0, sampleTexCoord.xy, 0.0);
        sampleColor.rgb = mix(vec3(1.0), sampleColor.rgb, float(sampleShadow0 < 1.0));

        shadow += min(sampleColor.rgb, sampleShadow1.rrr);
        idxSS++;
    }
    shadow /= float(SAMPLE_N);
    shadow *= shadow;
    float shadowRangeBlend = linearStep(shadowDistance - 8.0, shadowDistance, length(worldCoord.xz));
    return mix(shadow, vec3(1.0), shadowRangeBlend);
}



vec3 calcFresnel(Material material, float dotP) {
    /*
        Hardcoded metals
        https://shaderlabs.org/wiki/LabPBR_Material_Standard
        Metal	    Bit Value	N (R, G, B)	                K (R, G, B)
        Iron	    230	        2.9114,  2.9497,  2.5845	3.0893, 2.9318, 2.7670
        Gold	    231	        0.18299, 0.42108, 1.3734	3.4242, 2.3459, 1.7704
        Aluminum	232	        1.3456,  0.96521, 0.61722	7.4746, 6.3995, 5.3031
        Chrome	    233	        3.1071,  3.1812,  2.3230	3.3314, 3.3291, 3.1350
        Copper	    234	        0.27105, 0.67693, 1.3164	3.6092, 2.6248, 2.2921
        Lead	    235	        1.9100,  1.8300,  1.4400	3.5100, 3.4000, 3.1800
        Platinum	236	        2.3757,  2.0847,  1.8453	4.2655, 3.7153, 3.1365
        Silver	    237	        0.15943, 0.14512, 0.13547	3.9291, 3.1900, 2.3808
    */
    const vec3[] METAL_IOR = vec3[](
        vec3(2.9114, 2.9497, 2.5845),
        vec3(0.18299, 0.42108, 1.3734),
        vec3(1.3456, 0.96521, 0.61722),
        vec3(3.1071, 3.1812, 2.3230),
        vec3(0.27105, 0.67693, 1.3164),
        vec3(1.9100, 1.8300, 1.4400),
        vec3(2.3757, 2.0847, 1.8453),
        vec3(0.15943, 0.14512, 0.13547)
    );

    const vec3[] METAL_K = vec3[](
        vec3(3.0893, 2.9318, 2.7670),
        vec3(3.4242, 2.3459, 1.7704),
        vec3(7.4746, 6.3995, 5.3031),
        vec3(3.3314, 3.3291, 3.1350),
        vec3(3.6092, 2.6248, 2.2921),
        vec3(3.5100, 3.4000, 3.1800),
        vec3(4.2655, 3.7153, 3.1365),
        vec3(3.9291, 3.1900, 2.3808)
    );

    vec3 f = vec3(0.0);
    if (material.f0 < 229.5 / 255.0) {
        f = bsdf_frenel_cook_torrance(dotP, material.f0) * material.albedo.rgb;
    } else if (material.f0 < 237.5 / 255.0) {
        uint metalIdx = clamp(uint(material.f0 * 255.0) - 230u, 0u, 7u);
        vec3 ior = METAL_IOR[metalIdx];
        vec3 k = METAL_K[metalIdx];
        f = bsdf_fresnel_lazanyi(dotP, ior, k);
    } else {
        f = bsdf_frenel_schlick_f0(dotP, material.albedo.rgb);
    }

    return saturate(f);
}

vec3 directLighting(Material material, vec3 shadow, vec3 irradiance, vec3 L, vec3 N, vec3 V, vec3 F) {
    vec3 NSss = normalize(mix(N, L, material.sss * 0.9));
    vec3 H = normalize(L + V);
    float LDotV = dot(L, V);
    float LDotH = dot(L, H);
    float NDotL = dot(NSss, L);
    //    vec3 diffuseV = shadow * irradiance * bsdf_diffuseHammon(NDotL, NDotV, NDotH, LDotV, material.albedo, alpha);
    //    sunDiffuseV *= vec3(1.0) - fresnel;
    vec3 diffuseV = saturate(NDotL) * RCP_PI * shadow * irradiance * material.albedo;
    vec3 result = diffuseV;
    return result;
}