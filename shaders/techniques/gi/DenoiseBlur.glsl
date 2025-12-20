/*
    References:
        [ZHD20] Zhdan, Dmitry. "Fast Denoising With Self-Stabilizing Recurrent Blurs". GDC 2020.
            https://www.gdcvault.com/play/1026701/Fast-Denoising-With-Self-Stabilizing
            https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf
        [ZHD21] Zhdan, Dmitry. "ReBLUR: A Hierarchical Recurrent Denoiser". Ray Tracing Gems II. 2021.
            https://link.springer.com/content/pdf/10.1007/978-1-4842-7185-8_49.pdf
*/

#include "Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Mat2.glsl"

struct GeomData {
    vec3 geomNormal;
    vec3 normal;
    vec3 viewPos;
};

GeomData _gi_readGeomData(ivec2 texelPos, vec2 screenPos) {
    GeomData geomData;
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    geomData.viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    geomData.geomNormal = normalize(transient_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
    geomData.normal = normalize(transient_viewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
    return geomData;
}

vec4 _gi_readDiff(ivec2 texelPos) {
    #if GI_DENOISE_PASS == 1
    return transient_gi_blurDiff1_fetch(texelPos);
    #elif GI_DENOISE_PASS == 2
    return transient_gi_blurDiff2_fetch(texelPos);
    #endif
}

vec4 _gi_readSpec(ivec2 texelPos) {
    #if GI_DENOISE_PASS == 1
    return transient_gi_blurSpec1_fetch(texelPos);
    #elif GI_DENOISE_PASS == 2
    return transient_gi_blurSpec2_fetch(texelPos);
    #endif
}

vec2 _gi_mirrorUV(vec2 uv) {
    return 1.0 - abs(1.0 - (fract(uv * 0.5) * 2.0));
}

float gaussianKernel(float x) {
    return exp(-1.0 * pow2(x));
}

float geomNormalWeight(vec3 norA, vec3 norB) {
    return pow8(saturate(dot(norA, norB)));
}

float normalWeight(vec3 norA, vec3 norB) {
    return pow2(saturate(dot(norA, norB)));
}

float planeDistanceWeight(vec3 posA, vec3 normalA, vec3 posB, vec3 normalB) {
    float planeDistance = gi_planeDistance(posA, normalA, posB, normalB);
    return exp2(-256.0 * pow2(planeDistance));
}

void gi_blur(ivec2 texelPos, float kernelRadius, vec2 blurJitter) {
    vec4 diffResult = _gi_readDiff(texelPos);
    vec4 specResult = _gi_readSpec(texelPos);

    float angle = blurJitter.x * PI_2;
    vec2 dir = vec2(cos(angle), sin(angle));
    float rcpSamples = 1.0 / float(GI_DENOISE_SAMPLES);
    vec2 centerScreenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);

    GeomData centerGeomData = _gi_readGeomData(texelPos, centerScreenPos);

    vec2 centerTexelPos = vec2(texelPos) + vec2(0.5);
    float weightSum = 1.0;

    for (uint i = 0u; i < GI_DENOISE_SAMPLES; ++i) {
        dir *= GOLDEN_ANGLE;
        float baseRadius = sqrt((float(i) + blurJitter.y) * rcpSamples);
        vec2 offset = dir * (baseRadius * kernelRadius);
        vec2 sampleTexelPosF = centerTexelPos + offset;
        vec2 sampleUV = sampleTexelPosF * uval_mainImageSizeRcp;
        float weight = gaussianKernel(baseRadius);
        if (saturate(sampleUV) != sampleUV) {
            sampleUV = _gi_mirrorUV(sampleUV);
            weight = 1.0;
        }
        ivec2 sampleTexelPos = ivec2(sampleUV * uval_mainImageSize);

        // TODO: geom weights
        GeomData geomData = _gi_readGeomData(sampleTexelPos, sampleUV);

        vec4 diffSample = _gi_readDiff(sampleTexelPos);
        vec4 specSample = _gi_readSpec(sampleTexelPos);

        weight *= geomNormalWeight(centerGeomData.geomNormal, geomData.geomNormal);
        weight *= planeDistanceWeight(
            centerGeomData.viewPos,
            centerGeomData.geomNormal,
            geomData.viewPos,
            geomData.geomNormal
        );
        weight *= normalWeight(centerGeomData.normal, geomData.normal);

        diffResult += diffSample * weight;
        specResult += specSample * weight;

        weightSum += weight;
    }

    float rcpWeightSum = 1.0 / weightSum;

    diffResult *= rcpWeightSum;
    specResult *= rcpWeightSum;

    #if GI_DENOISE_PASS == 1
    transient_gi_blurDiff2_store(texelPos, diffResult);
    transient_gi_blurSpec2_store(texelPos, specResult);
    #elif GI_DENOISE_PASS == 2
    transient_gi_blurDiff1_store(texelPos, diffResult);
    transient_gi_blurSpec1_store(texelPos, specResult);
    #endif
}