#include "/techniques/textile/CSR32F.glsl"
#include "/util/Colors.glsl"
#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"
#include "/util/MaterialIDConst.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(RENDER_SCALE_FACTOR, RENDER_SCALE_FACTOR);

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;

shared vec3 shared_refraction[18][18];
shared vec3 shared_reflection[18][18];
shared float shared_viewZ[18][18];

struct ColorStats {
    vec3 minValue;
    vec3 maxValue;
    vec3 moment1;
    vec3 moment2;
    float count;
};

float translucentFrontViewZ(ivec2 texelPos) {
    float waterViewZ = -texelFetch(usam_csr32f, csr32f_tile1_texelToTexel(texelPos), 0).r;
    float translucentViewZ = -texelFetch(usam_csr32f, csr32f_tile3_texelToTexel(texelPos), 0).r;
    return max(waterViewZ, translucentViewZ);
}

void loadSharedData(uvec2 workGroupOrigin, uint index) {
    if (index >= 324u) return;

    ivec2 sharedPos = ivec2(index % 18u, index / 18u);
    ivec2 sourcePos = ivec2(workGroupOrigin) + sharedPos - 1;
    sourcePos = clamp(sourcePos, ivec2(0), uval_mainImageSizeI - 1);

    shared_refraction[sharedPos.y][sharedPos.x] = colors_RGBToYCoCg(
        max(transient_translucentRefraction_fetch(sourcePos).rgb, 0.0)
    );
    shared_reflection[sharedPos.y][sharedPos.x] = colors_RGBToYCoCg(
        max(transient_translucentReflection_fetch(sourcePos).rgb, 0.0)
    );
    shared_viewZ[sharedPos.y][sharedPos.x] = translucentFrontViewZ(sourcePos);
}

ColorStats colorStats_init() {
    ColorStats stats;
    stats.minValue = vec3(FLT_MAX);
    stats.maxValue = vec3(-FLT_MAX);
    stats.moment1 = vec3(0.0);
    stats.moment2 = vec3(0.0);
    stats.count = 0.0;
    return stats;
}

void colorStats_add(vec3 value, inout ColorStats stats) {
    stats.minValue = min(stats.minValue, value);
    stats.maxValue = max(stats.maxValue, value);
    stats.moment1 += value;
    stats.moment2 += value * value;
    stats.count += 1.0;
}

vec3 clipAABB(vec3 center, vec3 halfSize, vec3 value) {
    vec3 offset = value - center;
    vec3 unitOffset = abs(offset / max(halfSize, vec3(1e-6)));
    return center + offset / max(mmax3(unitOffset), 1.0);
}

vec3 clampHistory(vec3 history, ColorStats stats) {
    vec3 mean = stats.moment1 / stats.count;
    vec3 variance = max(stats.moment2 / stats.count - mean * mean, vec3(0.0));
    vec3 halfSize = max(sqrt(variance) * 2.0, vec3(1e-4));
    vec3 minValue = max(stats.minValue, mean - halfSize);
    vec3 maxValue = min(stats.maxValue, mean + halfSize);
    vec3 historyYCoCg = colors_RGBToYCoCg(max(history, 0.0));
    historyYCoCg = clipAABB((minValue + maxValue) * 0.5, (maxValue - minValue) * 0.5, historyYCoCg);
    return max(colors_YCoCgToRGB(historyYCoCg), 0.0);
}

void main() {
    uvec2 workGroupOrigin = gl_WorkGroupID.xy << 4u;
    loadSharedData(workGroupOrigin, gl_LocalInvocationIndex);
    loadSharedData(workGroupOrigin, gl_LocalInvocationIndex + 256u);
    barrier();

    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(texelPos, uval_mainImageSizeI))) return;

    ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 1;
    float viewZ = shared_viewZ[localPos.y][localPos.x];
    if (viewZ <= -65536.0) {
        transient_translucentRefractionResolved_store(texelPos, vec4(0.0));
        transient_translucentReflectionResolved_store(texelPos, vec4(0.0));
        return;
    }

    vec4 currentRefraction = transient_translucentRefraction_fetch(texelPos);
    vec4 currentReflection = transient_translucentReflection_fetch(texelPos);

    ColorStats refractionStats = colorStats_init();
    ColorStats reflectionStats = colorStats_init();
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 samplePos = localPos + ivec2(x, y);
            if (shared_viewZ[samplePos.y][samplePos.x] > -65536.0) {
                colorStats_add(shared_refraction[samplePos.y][samplePos.x], refractionStats);
                colorStats_add(shared_reflection[samplePos.y][samplePos.x], reflectionStats);
            }
        }
    }

    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferTranslucentData1, texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferTranslucentData2, texelPos, 0), gData);

    vec2 screenUv = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
    vec2 currentUv = screenUv - uval_taaJitterUV;
    vec3 currentViewPos = coords_toViewCoord(currentUv, viewZ, global_camProjInverse);
    vec4 previousViewPos = coord_viewCurrToPrev(vec4(currentViewPos, 1.0), gData.isHand);
    vec4 previousClip = global_prevCamProj * previousViewPos;

    bool validHistory = frameCounter > 1 && global_taaResetFactor.z >= 0.5;
    validHistory = validHistory && previousClip.w > 0.0 && previousClip.z > 0.0;

    vec2 historyUv = vec2(-1.0);
    if (validHistory) {
        vec2 previousNdc = previousClip.xy / previousClip.w;
        validHistory = all(lessThan(abs(previousNdc), vec2(1.0)));
        historyUv = previousNdc * 0.5 + 0.5 + uval_prevTaaJitterUV;
        validHistory = validHistory && all(greaterThanEqual(historyUv, vec2(0.0)));
        validHistory = validHistory && all(lessThan(historyUv, vec2(1.0)));
    }

    vec4 previousRefraction = vec4(0.0);
    vec4 previousReflection = vec4(0.0);
    if (validHistory) {
        ivec2 historyTexel = clamp(ivec2(historyUv * uval_mainImageSize), ivec2(0), uval_mainImageSizeI - 1);
        vec4 refractionMetadata = history_translucentRefraction_fetch(historyTexel);
        vec4 reflectionMetadata = history_translucentReflection_fetch(historyTexel);

        bool isWater = gData.materialID == MATERIAL_ID_WATER;
        float surfaceSign = isWater ? -1.0 : 1.0;
        float previousDistance = abs(refractionMetadata.a);
        float expectedDistance = max(-previousViewPos.z, 0.0);
        float depthTolerance = max(0.1, expectedDistance * 0.02);
        validHistory = reflectionMetadata.a > 0.0;
        validHistory = validHistory && refractionMetadata.a * surfaceSign > 0.0;
        validHistory = validHistory && abs(previousDistance - expectedDistance) <= depthTolerance;

        if (validHistory) {
            previousRefraction = history_translucentRefraction_sample(historyUv);
            previousReflection = history_translucentReflection_sample(historyUv);
            previousRefraction.rgb = clampHistory(previousRefraction.rgb, refractionStats);
            previousReflection.rgb = clampHistory(previousReflection.rgb, reflectionStats);
            previousReflection.a = reflectionMetadata.a;
        }
    }

    float maxAccumulation = mix(2.0, 4.0, pow3(global_motionFactor.w));
    if (gData.isHand) {
        maxAccumulation = min(maxAccumulation, 2.0);
    }

    float accumulation = validHistory ? min(previousReflection.a + 1.0, maxAccumulation) : 1.0;
    float currentWeight = rcp(accumulation);
    vec3 resolvedRefraction = mix(previousRefraction.rgb, max(currentRefraction.rgb, 0.0), currentWeight);
    vec3 resolvedReflection = mix(previousReflection.rgb, max(currentReflection.rgb, 0.0), currentWeight);

    // FSR3 replaces the full-frame TAA pass, so stochastic rough translucent SST would otherwise reach it unfiltered.
    // Resolve only the two SST signals here; filtering the composed frame would double-filter solid pixels and mix background parallax.
    float surfaceSign = gData.materialID == MATERIAL_ID_WATER ? -1.0 : 1.0;
    transient_translucentRefractionResolved_store(texelPos, vec4(resolvedRefraction, surfaceSign * -viewZ));
    transient_translucentReflectionResolved_store(texelPos, vec4(resolvedReflection, accumulation));
}
