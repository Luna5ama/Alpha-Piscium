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

float gaussianKernel(float x, float sigma) {
    return exp(-sigma * pow2(x));
}
float normalWeight(vec3 norA, vec3 norB, float factor) {
    return pow(saturate(dot(norA, norB)), factor);
}

float planeDistanceWeight(vec3 posA, vec3 normalA, vec3 posB, vec3 normalB, float factor) {
    float planeDistance = gi_planeDistance(posA, normalA, posB, normalB);
    return exp2(factor * pow2(planeDistance));
}

void gi_blur(ivec2 texelPos, vec4 baseKernelRadius, GIHistoryData historyData, vec2 blurJitter) {
    vec2 centerScreenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    GeomData centerGeomData = _gi_readGeomData(texelPos, centerScreenPos);

    if (centerGeomData.viewPos.z > -65536.0) {
        vec4 centerDiff = _gi_readDiff(texelPos);
        vec4 centerSpec = _gi_readSpec(texelPos);

        float historyLength = historyData.realHistoryLength * REAL_HISTORY_LENGTH;
        float accumFactor = (1.0 / (1.0 + historyLength));
        float invAccumFactor = saturate(1.0 - accumFactor); // Increases as history accumulates

        float hitDistFactor = linearStep(0.0, 4.0, historyData.diffuseHitDistance);
        #if GI_DENOISE_PASS == 1
        hitDistFactor = pow(hitDistFactor, invAccumFactor * 1.0);
        #else
        hitDistFactor = pow(hitDistFactor, invAccumFactor * 0.5);
        #endif

        float kernelRadius = baseKernelRadius.x;
        kernelRadius *= sqrt(accumFactor);

        float varianceFactor = 0.0;

        // TODO: optimize with shared memory
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                varianceFactor += _gi_readDiff(texelPos + ivec2(dx, dy)).w;
            }
        }

        varianceFactor /= 25.0;

        kernelRadius *= 1.0 + varianceFactor * baseKernelRadius.y;
        kernelRadius *= hitDistFactor;
        kernelRadius = clamp(kernelRadius, baseKernelRadius.z, baseKernelRadius.w);

        float hitDistColorWeightHistoryDecay = historyData.realHistoryLength * 0.5 + 0.001;
        float hitDistColorWeightFactor = saturate(hitDistColorWeightHistoryDecay / (hitDistColorWeightHistoryDecay + historyData.diffuseHitDistance));
        #if GI_DENOISE_PASS == 1
        float baseColorWeight = -mix(16.0, 512.0, hitDistColorWeightFactor);
        #else GI_DENOISE_PASS == 2
        float baseColorWeight = -mix(4.0, 128.0, hitDistColorWeightFactor);
        #endif

        baseColorWeight *= pow2(invAccumFactor);
        float baseGeomNormalWeight = invAccumFactor * 8.0;
        float baseNormalWeight = invAccumFactor * 4.0;
        float basePlaneDistWeight = invAccumFactor * -512.0;

        vec4 diffResult = centerDiff;
        vec4 specResult = centerSpec;

        float angle = blurJitter.x * PI_2;
        vec2 dir = vec2(cos(angle), sin(angle));
        float rcpSamples = 1.0 / float(GI_DENOISE_SAMPLES);

        vec2 centerTexelPos = vec2(texelPos) + vec2(0.5);
        float weightSum = 1.0;
        float edgeWeightSum = 0.0;
        //    float lastEdgeWeight = 1.0;

        float centerLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, centerDiff.rgb);
        float moment1 = centerLuma;
        float moment2 = pow2(centerLuma);

        float sigma = 0.69;
        sigma += kernelRadius * 4.0 * pow2(1.0 - saturate(hitDistFactor));

        #if ENABLE_DENOISER
        for (uint i = 0u; i < GI_DENOISE_SAMPLES; ++i) {
            dir *= MAT2_GOLDEN_ANGLE;
            float baseRadius = sqrt((float(i) + blurJitter.y) * rcpSamples);
            //        float baseRadius = ((float(i) + blurJitter.y) * rcpSamples);
            vec2 offset = dir * (baseRadius * kernelRadius);
            vec2 sampleTexelPosF = centerTexelPos + offset;
            vec2 sampleUV = sampleTexelPosF * uval_mainImageSizeRcp;
            float kernelWeight = gaussianKernel(baseRadius, sigma);
            if (saturate(sampleUV) != sampleUV) {
                sampleUV = _gi_mirrorUV(sampleUV);
            }
            ivec2 sampleTexelPos = ivec2(sampleUV * uval_mainImageSize);

            GeomData geomData = _gi_readGeomData(sampleTexelPos, sampleUV);

            vec4 diffSample = _gi_readDiff(sampleTexelPos);

            float edgeWeight = 1.0;
            edgeWeight *= normalWeight(centerGeomData.geomNormal, geomData.geomNormal, baseGeomNormalWeight);
            edgeWeight *= planeDistanceWeight(
                centerGeomData.viewPos,
                centerGeomData.geomNormal,
                geomData.viewPos,
                geomData.geomNormal,
                basePlaneDistWeight
            );
            edgeWeight *= normalWeight(centerGeomData.normal, geomData.normal, baseNormalWeight);

            //        edgeWeight *= sqrt(lastEdgeWeight);
            //        lastEdgeWeight = edgeWeight;

            vec3 colorDiff = abs(centerDiff.rgb - diffSample.rgb);
            float lumaDiff = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, colorDiff);
            float colorWeight = exp2(baseColorWeight * lumaDiff);
            //        edgeWeight *= colorWeight;
            edgeWeight = smoothstep(0.0, 1.0, edgeWeight);


            float totalWeight = kernelWeight * edgeWeight;

            float sampleLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, diffSample.rgb);
            moment1 += sampleLuma * edgeWeight;
            moment2 += pow2(sampleLuma) * edgeWeight;

            diffResult += diffSample * totalWeight;
            weightSum += totalWeight;
            edgeWeightSum += edgeWeight;
        }
        #endif

        float rcpWeightSum = 1.0 / weightSum;

        diffResult *= rcpWeightSum;
        specResult *= rcpWeightSum;

        float rcpEdgeWeightSum = 1.0 / (edgeWeightSum + 1.0);
        moment1 *= rcpEdgeWeightSum;
        moment2 *= rcpEdgeWeightSum;

        float variance = max(0.0, moment2 - pow2(moment1));
        float stddev = sqrt(variance);

        #if GI_DENOISE_PASS == 1
        diffResult.w = pow(safeDiv(variance, centerLuma) * 8.0, 1.0 / 2.0);
        transient_gi_blurDiff2_store(texelPos, diffResult);
        transient_gi_blurSpec2_store(texelPos, specResult);
        #elif GI_DENOISE_PASS == 2
        transient_gi_blurDiff1_store(texelPos, diffResult);
        transient_gi_blurSpec1_store(texelPos, specResult);
        #endif
    }
}