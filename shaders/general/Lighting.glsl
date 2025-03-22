#include "/util/Rand.glsl"
#include "/util/Material.glsl"
#include "/util/GBufferData.glsl"
#include "/util/BSDF.glsl"
#include "/rtwsm/RTWSM.glsl"

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

uniform sampler2D usam_skyLUT;

GBufferData gData;
vec3 lighting_viewCoord;
vec3 lighting_viewDir;
ivec2 lighting_texelPos;

void lighting_init(vec3 viewCoord, ivec2 texelPos) {
    lighting_viewCoord = viewCoord;
    lighting_viewDir = normalize(-viewCoord);
    lighting_texelPos = texelPos;
}

float searchBlocker(vec3 shadowTexCoord) {
    #define BLOCKER_SEARCH_LOD SETTING_PCSS_BLOCKER_SEARCH_LOD

    #define BLOCKER_SEARCH_N SETTING_PCSS_BLOCKER_SEARCH_COUNT

    float blockerSearchRange = 0.1;
    uint idxB = frameCounter * BLOCKER_SEARCH_N + (rand_hash31(floatBitsToUint(lighting_viewCoord.xyz)) & 1023u);

    float blockerDepth = 0.0f;
    int n = 0;

    shadowTexCoord.z = rtwsm_linearDepth(shadowTexCoord.z);
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

vec3 calcShadow(float sssFactor) {
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

    viewCoord += gData.normal * normalOffset;

    vec4 worldCoord = gbufferModelViewInverse * vec4(viewCoord, 1.0);

    vec4 shadowTexCoordCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * worldCoord;
    shadowTexCoordCS /= shadowTexCoordCS.w;

    vec3 shadowTexCoord = shadowTexCoordCS.xyz * 0.5 + 0.5;
    float blockerDistance = searchBlocker(shadowTexCoord);

    shadowTexCoord.z = rtwsm_linearDepth(shadowTexCoord.z);

    float ssRange = 0.0;
    #if SETTING_PCSS_BPF > 0
    ssRange += exp2(SETTING_PCSS_BPF - 10.0);
    #endif
    ssRange += uval_sunAngularRadius * 2.0 * SETTING_PCSS_VPF * blockerDistance;
    ssRange = saturate(ssRange);

    const float ssRangeMul = 0.5;

    ssRange *= ssRangeMul;

    #define SAMPLE_N SETTING_PCSS_SAMPLE_COUNT

    vec3 shadow = vec3(0.0);
    uint idxSS = frameCounter * SAMPLE_N;

    uint hashss = rand_hash31(floatBitsToUint(viewCoord.xyz)) & 1023u;

    #define DEPTH_BIAS_DISTANCE_FACTOR 1024.0
    float dbfDistanceCoeff = (DEPTH_BIAS_DISTANCE_FACTOR / (DEPTH_BIAS_DISTANCE_FACTOR + max(distnaceSq, 1.0)));
    float depthBiasFactor = 0.001 + lightNormalDot * 0.001;
    depthBiasFactor += mix(0.005 + lightNormalDot * 0.005, -0.001, dbfDistanceCoeff);

    for (int i = 0; i < SAMPLE_N; i++) {
        vec2 randomOffset;
        randomOffset.x = rand_IGN(lighting_texelPos, idxSS);
        randomOffset.y = rand_r2Seq1(idxSS + hashss);
        vec3 sampleTexCoord = shadowTexCoord;

        float theta = randomOffset.x * PI_2;
        float r = sqrt(randomOffset.y) * ssRange;
        r += randomOffset.y * sssFactor * ssRangeMul * SETTING_SSS_DIFFUSE_RANGE;
        sampleTexCoord.xy += r * vec2(cos(theta), sin(theta)) * vec2(shadowProjection[0][0], shadowProjection[1][1]);

        sampleTexCoord.z += randomOffset.y * min(sssFactor * SETTING_SSS_DEPTH_RANGE, SETTING_SSS_MAX_DEPTH_RANGE);
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
    shadow /= float(SAMPLE_N);
    float shadowRangeBlend = linearStep(shadowDistance - 8.0, shadowDistance, length(worldCoord.xz));
    return mix(shadow, vec3(1.0), shadowRangeBlend);
}

struct LightingResult {
    vec3 diffuse;
    vec3 diffuseLambertian;
    vec3 specular;
    vec3 sss;
};

LightingResult lightingResult_add(LightingResult a, LightingResult b) {
    LightingResult result;
    result.diffuse = a.diffuse + b.diffuse;
    result.diffuseLambertian = a.diffuseLambertian + b.diffuseLambertian;
    result.specular = a.specular + b.specular;
    result.sss = a.sss + b.sss;
    return result;
}

vec3 skyReflection(Material material, float lmCoordSky, vec3 N) {
    vec3 V = lighting_viewDir;
    float NDotV = dot(N, V);
    vec3 fresnelReflection = bsdf_fresnel(material, saturate(NDotV));

    vec3 reflectDirView = reflect(-V, N);
    vec3 reflectDir = normalize(mat3(gbufferModelViewInverse) * reflectDirView);
    vec2 reflectLUTUV = coords_octEncode01(reflectDir);
    vec3 reflectRadiance = texture(usam_skyLUT, reflectLUTUV).rgb;

    vec3 result = fresnelReflection;
    result *= material.albedo;
    result *= linearStep(1.5 / 16.0, 15.5 / 16.0, lmCoordSky);
    result *= texture(usam_skyLUT, reflectLUTUV).rgb;

    return result;
}

LightingResult directLighting(Material material, vec4 irradiance, vec3 L, vec3 N) {
    vec3 V = lighting_viewDir;
    vec3 H = normalize(L + V);
    float LDotV = dot(L, V);
    float LDotH = dot(L, H);
    float NDotL = dot(N, L);
    float NDotV = dot(N, V);
    float NDotH = dot(N, H);

    vec3 fresnel = bsdf_fresnel(material, saturate(LDotH));

    LightingResult result;

    float diffuseBase = 1.0 - material.metallic;

    result.diffuse = diffuseBase * irradiance.rgb * (vec3(1.0) - fresnel) * material.albedo;
    result.diffuse *= bsdf_disneyDiffuse(material, NDotL, NDotV, LDotH);

    float diffuseV = diffuseBase * saturate(NDotL) * RCP_PI;
    result.diffuseLambertian = diffuseV * (vec3(1.0) - fresnel) * irradiance.rgb * material.albedo;

    float shadowPow = saturate(1.0 - irradiance.a);
    shadowPow = (1.0 - SETTING_SSS_HIGHLIGHT * 0.5) + pow(shadowPow, 8.0) * material.sss * SETTING_SSS_SCTR_FACTOR;

    float backDot = saturate(NDotL * - 0.5 + 0.5);
    float sssV = material.sss * RCP_PI * backDot * 0.5 * SETTING_SSS_STRENGTH;
    result.sss = sssV * pow(material.albedo, vec3(shadowPow)) * irradiance.rgb;

    result.specular = irradiance.rgb * material.albedo * fresnel;
    result.specular *= bsdf_ggx(material, NDotL, NDotV, NDotH);

    return result;
}