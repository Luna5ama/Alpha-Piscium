#version 460 compatibility

#include "_Util.glsl"
#include "rtwsm/RTWSM.glsl"
#include "atmosphere/Common.glsl"

uniform sampler2D usam_main;
uniform usampler2D usam_gbuffer;
uniform sampler2D usam_viewZ;

uniform sampler2D depthtex0;

const bool generateShadowMipmap = true;
const bool shadowtex0Mipmap = true;
const bool shadowtex1Mipmap = true;

const bool shadowHardwareFiltering0 = true;
uniform sampler2D shadowtex0;
uniform sampler2DShadow shadowtex0HW;

const bool shadowHardwareFiltering1 = true;
uniform sampler2D shadowtex1;
uniform sampler2DShadow shadowtex1HW;

uniform sampler2D shadowcolor0;

uniform sampler2D usam_rtwsm_warpingMap;

uniform sampler2D usam_transmittanceLUT;

in vec2 frag_texCoord;

ivec2 intTexCoord = ivec2(gl_FragCoord.xy);
GBufferData gData;
vec3 g_viewCoord;
vec3 g_viewDir;

uint coord3Rand[2];

/* RENDERTARGETS:0 */
layout(location = 0) out vec4 rt_out;

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
        vec2 texelSize;
        sampleTexCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_warpingMap, sampleTexCoord.xy, texelSize);
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

    float lightNormalDot = 1.0 - abs(dot(shadowLightPosition * 0.01, gData.normal));
    #define NORMAL_OFFSET_DISTANCE_FACTOR2 512.0
    float normalOffset2 = 1.0 - (NORMAL_OFFSET_DISTANCE_FACTOR2 / (NORMAL_OFFSET_DISTANCE_FACTOR2 + distnaceSq));
    normalOffset += saturate(normalOffset2 * lightNormalDot) * 0.2;

    viewCoord += gData.normal * normalOffset;

    vec4 worldCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);
    worldCoord = mix(worldCoord, vec4(0.0, 1.0, 0.0, 1.0), float(gData.materialID == MATERIAL_ID_HAND));

    vec4 shadowTexCoordCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * worldCoord;
    shadowTexCoordCS /= shadowTexCoordCS.w;

    vec3 shadowTexCoord = shadowTexCoordCS.xyz * 0.5 + 0.5;

    float blockerDistance = searchBlocker(shadowTexCoord);

    float ssRange = exp2(SETTING_PCSS_BPF - 8.0);
    ssRange += uval_sunAngularRadius.x * 2.0 * SETTING_PCSS_VPF * blockerDistance;
    ssRange = saturate(ssRange);
    ssRange *= 0.4;

    #define SAMPLE_N SETTING_PCSS_SAMPLE_COUNT

    vec4 totalShadow = vec4(0.0);
    uint idxSS = (frameCounter + coord3Rand[0]) * SAMPLE_N;

    #define DEPTH_BIAS_DISTANCE_FACTOR 1027.0
    float dbfDistanceCoeff = (DEPTH_BIAS_DISTANCE_FACTOR / (DEPTH_BIAS_DISTANCE_FACTOR + max(distnaceSq, 1.0)));
    float depthBiasFactor = 0.001 + lightNormalDot * 0.002;
    depthBiasFactor += mix(0.05 + lightNormalDot * 0.005, -0.001, dbfDistanceCoeff);

    for (int i = 0; i < SAMPLE_N; i++) {
        vec2 randomOffset = (rand_r2Seq2(idxSS) * 2.0 - 1.0);
        vec3 sampleTexCoord = shadowTexCoord;
        sampleTexCoord.xy += ssRange * randomOffset * vec2(shadowProjection[0][0], shadowProjection[1][1]);
        vec2 texelSize;
        sampleTexCoord.xy = rtwsm_warpTexCoordTexelSize(usam_rtwsm_warpingMap, sampleTexCoord.xy, texelSize);
        float depthBias = SHADOW_MAP_SIZE.y * depthBiasFactor / length(texelSize);
        depthBias = min(depthBias, 0.001);
        sampleTexCoord.z -= depthBias;

        float sampleShadow0 = rtwsm_sampleShadowDepth(shadowtex0HW, sampleTexCoord, 0.0);
        float sampleShadow1 = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
        vec4 sampleColor = rtwsm_sampleShadowColor(shadowcolor0, sampleTexCoord.xy, 0.0);
        sampleColor.rgb = mix(vec3(1.0), sampleColor.rgb, float(sampleShadow0 < 1.0));

        totalShadow += vec4(sampleColor.rgb, sampleShadow1);
        idxSS++;
    }
    totalShadow /= float(SAMPLE_N);
    totalShadow.a *= totalShadow.a;
    vec3 result = min(totalShadow.rgb, totalShadow.aaa);
    float shadowRangeBlend = linearStep(shadowDistance - 8.0, shadowDistance, length(worldCoord.xz));
    return mix(result, vec3(1.0), shadowRangeBlend);
}

vec3 calcDirectLighting(Material material, vec3 shadow, vec3 L, vec3 N, vec3 V) {
    vec3 directLight = vec3(0.0);
    float ambient = 0.5;
    directLight += ambient * material.albedo;

    vec3 H = normalize(L + V);
    float NDotL = dot(N, L);
    float NDotV = dot(N, V);
    float NDotH = dot(N, H);
    float LDotV = dot(L, V);

    float alpha = material.roughness;

    vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;

    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(g_viewCoord, 1.0)).xyz;
    vec3 worldPos = feetPlayerPos + cameraPosition;
    float viewAltitude = calcViewAltitude(atmosphere, worldPos);
    float cosSunZenith = calcCosSunZenith(atmosphere, L);
    vec2 transmittanceUV;
    lutTransmittanceParamsToUv(atmosphere, viewAltitude, cosSunZenith, transmittanceUV);
    vec3 transmittance = texture(usam_transmittanceLUT, transmittanceUV).rgb;

    sunRadiance *= transmittance;

//    vec3 diffuseV = bsdfs_diffuseHammon(NDotL, NDotV, NDotH, LDotV, material.albedo, alpha);
    vec3 diffuseV = saturate(NDotL) * material.albedo * RCP_PI_CONST;
    directLight += shadow * diffuseV * sunRadiance;
    directLight += material.emissive * material.albedo * 32.0;

    return directLight;
}

void doStuff() {
    vec3 shadow = calcShadow(0.0);

    Material material = material_decode(gData);

    vec3 directLight = calcDirectLighting(material, shadow, sunPosition * 0.01, gData.normal, g_viewDir);

    rt_out += vec4(directLight, 1.0);
}

void main() {
    rt_out = vec4(0.0);
    float viewZ = texelFetch(usam_viewZ, intTexCoord, 0).r;
    if (viewZ == 1.0) {
        rt_out = texelFetch(usam_main, intTexCoord, 0);
        return;
    }

    gbuffer_unpack(texelFetch(usam_gbuffer, ivec2(gl_FragCoord.xy), 0), gData);
    g_viewCoord = coords_toViewCoord(frag_texCoord, viewZ, gbufferProjectionInverse);
    g_viewDir = normalize(-g_viewCoord);

    coord3Rand[0] = rand_hash31(floatBitsToUint(g_viewCoord.xyz)) & 1023u;
    coord3Rand[1] = rand_hash31(floatBitsToUint(g_viewCoord.xzy)) & 1023u;

    doStuff();
}