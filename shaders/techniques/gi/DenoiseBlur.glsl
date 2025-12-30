/*
    References:
        [ZHD20] Zhdan, Dmitry. "Fast Denoising With Self-Stabilizing Recurrent Blurs". GDC 2020.
            https://www.gdcvault.com/play/1026701/Fast-Denoising-With-Self-Stabilizing
            https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf
        [ZHD21] Zhdan, Dmitry. "ReBLUR: A Hierarchical Recurrent Denoiser". Ray Tracing Gems II. 2021.
            https://link.springer.com/content/pdf/10.1007/978-1-4842-7185-8_49.pdf
*/

#include "Common.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/Rand.glsl"
#include "/util/Coords.glsl"
#include "/util/Mat2.glsl"
#include "/util/Rand.glsl"
#include "/util/Dither.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba16f) uniform writeonly image2D uimg_temp3;

uvec2 groupOriginTexelPos = gl_WorkGroupID.xy << 4u;

// Shared memory with padding for 5x5 tap (-2 to +2)
// Each work group is 16x16, need +2 padding on each side for 5x5 taps
shared float shared_varianceData[20][20];

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
    return transient_gi_blurDiff2_fetch(texelPos);
    #elif GI_DENOISE_PASS == 2
    return transient_gi_blurDiff1_fetch(texelPos);
    #endif
}

vec4 _gi_readSpec(ivec2 texelPos) {
    #if GI_DENOISE_PASS == 1
    return transient_gi_blurSpec2_fetch(texelPos);
    #elif GI_DENOISE_PASS == 2
    return transient_gi_blurSpec1_fetch(texelPos);
    #endif
}

void loadSharedVarianceData(uint index) {
    if (index < 400u) { // 20 * 20 = 400
        uvec2 sharedXY = uvec2(index % 20u, index / 20u);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 2;
        srcXY = clamp(srcXY, ivec2(0), ivec2(uval_mainImageSize - 1));

        float variance = _gi_readDiff(srcXY).w;
        shared_varianceData[sharedXY.y][sharedXY.x] = variance;
    }
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

void main() {
    if (hiz_groupGroundCheckSubgroup(gl_WorkGroupID.xy, 4)) {
        // Load shared memory for variance filtering (20x20 = 400 elements)
        loadSharedVarianceData(gl_LocalInvocationIndex);
        loadSharedVarianceData(gl_LocalInvocationIndex + 256u);

        ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
        vec4 baseKernelRadius = GI_DENOISE_BLUR_RADIUS;
        vec2 blurJitter = rand_stbnVec2(texelPos + GI_DENOISE_RAND_NOISE_OFFSET, frameCounter);

        vec2 centerScreenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        GeomData centerGeomData = _gi_readGeomData(texelPos, centerScreenPos);

        barrier();

        if (centerGeomData.viewPos.z > -65536.0) {
            GIHistoryData historyData = gi_historyData_init();
            gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));

            vec2 localMinHitDistance = transient_gi_filteredHitDistances_fetch(texelPos).xy;

            vec4 centerDiff = _gi_readDiff(texelPos);
            vec4 centerSpec = _gi_readSpec(texelPos);

            float historyLength = historyData.historyLength * TOTAL_HISTORY_LENGTH;
            float sqrtRealHistoryLength = sqrt(historyData.historyLength);
            float accumFactor = (1.0 / (1.0 + historyLength));
            float invAccumFactor = saturate(1.0 - accumFactor); // Increases as history accumulates

            float hitDistFactor = linearStep(0.0, 4.0, localMinHitDistance.x);
            #if GI_DENOISE_PASS == 1
            hitDistFactor = pow(hitDistFactor, sqrtRealHistoryLength * 2.0);
            #else
            hitDistFactor = pow(hitDistFactor, sqrtRealHistoryLength * 1.0);
            #endif

            float kernelRadius = baseKernelRadius.x;
            kernelRadius *= pow(accumFactor, 0.25);

            float varianceFactor = 0.0;

            // Optimized variance calculation using shared memory
            ivec2 localPos = ivec2(gl_LocalInvocationID.xy) + 2; // +2 for padding
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    ivec2 samplePos = localPos + ivec2(dx, dy);
                    varianceFactor += shared_varianceData[samplePos.y][samplePos.x];
                }
            }

            varianceFactor /= 25.0;

            kernelRadius *= 1.0 + varianceFactor * baseKernelRadius.y;
            kernelRadius *= hitDistFactor;
            kernelRadius = clamp(kernelRadius, baseKernelRadius.z, baseKernelRadius.w);

//            float hitDistColorWeightHistoryDecay = historyData.realHistoryLength * 0.5 + 0.001;
//            float hitDistColorWeightFactor = saturate(hitDistColorWeightHistoryDecay / (hitDistColorWeightHistoryDecay + localMinHitDistance.x));
//            #if GI_DENOISE_PASS == 1
//            float baseColorWeight = -mix(16.0, 512.0, hitDistColorWeightFactor);
//            #else GI_DENOISE_PASS == 2
//            float baseColorWeight = -mix(4.0, 128.0, hitDistColorWeightFactor);
//            #endif
//
//            baseColorWeight *= pow2(invAccumFactor);
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
            float extraVarianceFactor = 0.0;
            float edgeWeightSum = 0.0;

            float centerLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, centerDiff.rgb);
            float moment1 = centerLuma;
            float moment2 = pow2(centerLuma);

            float sigma = 0.69;
            sigma += kernelRadius * 2.0 * pow2(1.0 - saturate(hitDistFactor));

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


                float totalWeight = kernelWeight * smoothstep(0.0, 1.0, edgeWeight);

                float sampleLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, diffSample.rgb);
                moment1 += sampleLuma * edgeWeight;
                moment2 += pow2(sampleLuma) * edgeWeight;
                extraVarianceFactor += diffSample.w * edgeWeight;

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
            extraVarianceFactor *= rcpEdgeWeightSum;

            float variance = max(0.0, moment2 - pow2(moment1));
            float stddev = sqrt(variance);

            float ditherNoise = rand_stbnVec1(texelPos, frameCounter + GI_DENOISE_PASS);
            diffResult.rgb = dither_fp16(diffResult.rgb, ditherNoise);
            specResult.rgb = dither_fp16(specResult.rgb, ditherNoise);

            #if GI_DENOISE_PASS == 1
            vec4 localLumaData = transient_gi_localLuma_fetch(texelPos);
            float newVarianceFactor = stddev * safeRcp(localLumaData.x + localLumaData.y);
            diffResult.w = varianceFactor + extraVarianceFactor + newVarianceFactor;

            transient_gi_blurDiff1_store(texelPos, diffResult);
            transient_gi_blurSpec1_store(texelPos, specResult);
            #elif GI_DENOISE_PASS == 2
            #if SETTING_DEBUG_OUTPUT
            if (RANDOM_FRAME < MAX_FRAMES){
                //            imageStore(uimg_temp3, texelPos, vec4(linearStep(baseKernelRadius.z, baseKernelRadius.w, kernelRadius)));
//                imageStore(uimg_temp3, texelPos, vec4(varianceFactor));
            }
            #endif
            transient_gi_blurDiff2_store(texelPos, diffResult);
            transient_gi_blurSpec2_store(texelPos, specResult);
            #endif
        }
    }
}