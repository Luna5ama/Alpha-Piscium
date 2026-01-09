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
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
layout(rgba8) uniform writeonly image2D uimg_rgba8;
layout(rgba16f) uniform writeonly image2D uimg_temp3;

// Shared memory with padding for 5x5 tap (-2 to +2)
// Each work group is 16x16, need +2 padding on each side for 5x5 taps
shared vec2 shared_varianceData[20][20];

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

vec2 _gi_readVariance(ivec2 texelPos) {
    #if GI_DENOISE_PASS == 1
    return transient_gi_denoiseVariance1_fetch(texelPos).xy;
    #elif GI_DENOISE_PASS == 2
    return transient_gi_denoiseVariance2_fetch(texelPos).xy;
    #endif
}

void loadSharedVarianceData(uvec2 groupOriginTexelPos, uint index) {
    if (index < 400u) { // 20 * 20 = 400
        uvec2 sharedXY = uvec2(index % 20u, index / 20u);
        ivec2 srcXY = ivec2(groupOriginTexelPos) + ivec2(sharedXY) - 2;
        srcXY = clamp(srcXY, ivec2(0), ivec2(uval_mainImageSize - 1));
        // x: diffuse variance
        // y: specular variance
        // z: history length
        // w: real history length
        shared_varianceData[sharedXY.y][sharedXY.x] = _gi_readVariance(srcXY);
    }
}

float gaussianKernel(float x, float sigma) {
    return exp(-sigma * pow2(x));
}

float normalWeight(GeomData a, GeomData b, float factor) {
    float geomNormalDot = saturate(dot(a.geomNormal, b.geomNormal));
    float normalDot = saturate(dot(a.normal, b.normal));
    return pow(pow4(geomNormalDot) * normalDot, factor);
}

float planeDistanceWeight(vec3 posA, vec3 normalA, vec3 posB, vec3 normalB, float factor) {
    float planeDistance = gi_planeDistance(posA, normalA, posB, normalB);
    return exp2(factor * pow2(planeDistance));
}

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (hiz_groupGroundCheckSubgroup(swizzledWGPos, 4)) {
        vec2 centerScreenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        GeomData centerGeomData = _gi_readGeomData(texelPos, centerScreenPos);
        // Load shared memory for variance filtering (20x20 = 400 elements)
        loadSharedVarianceData(workGroupOrigin, gl_LocalInvocationIndex);
        loadSharedVarianceData(workGroupOrigin, gl_LocalInvocationIndex + 256u);

        vec4 baseKernelRadius = GI_DENOISE_BLUR_RADIUS;
        vec2 blurJitter = rand_stbnVec2(texelPos + GI_DENOISE_RAND_NOISE_OFFSET, frameCounter);

        barrier();

        if (centerGeomData.viewPos.z > -65536.0) {
            GIHistoryData historyData = gi_historyData_init();
            gi_historyData_unpack5(historyData, transient_gi5Reprojected_fetch(texelPos));
            vec2 hitDistanceFactors = transient_gi_hitDistanceFactors_fetch(texelPos).xy;
            vec4 centerDiff = _gi_readDiff(texelPos);
            vec4 centerSpec = _gi_readSpec(texelPos);

            vec2 filteredInputVariance = vec2(0.0);

            // Optimized variance calculation using shared memory
            ivec2 localPos = ivec2(mortonPos) + 2; // +2 for padding
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    ivec2 samplePos = localPos + ivec2(dx, dy);
                    filteredInputVariance += shared_varianceData[samplePos.y][samplePos.x];
                }
            }

            filteredInputVariance /= 25.0;

            float historyLength = historyData.historyLength * TOTAL_HISTORY_LENGTH;
            float sqrtRealHistoryLength = sqrt(historyData.historyLength);
            float accumFactor = (1.0 / (1.0 + historyLength));
            float invAccumFactor = saturate(1.0 - accumFactor); // Increases as history accumulates

            float hitDistFactor = hitDistanceFactors.x;
            #if GI_DENOISE_PASS == 1
            hitDistFactor = saturate(pow(hitDistFactor, sqrtRealHistoryLength * 4.0));
            #else
            hitDistFactor = saturate(pow(hitDistFactor, sqrtRealHistoryLength * 2.0));
            #endif

            float kernelRadius = baseKernelRadius.x;
            kernelRadius *= pow(accumFactor, 0.5);

            kernelRadius *= 1.0 + filteredInputVariance.x * baseKernelRadius.y;
            kernelRadius *= hitDistFactor + 0.1;
            kernelRadius = clamp(kernelRadius, baseKernelRadius.z, baseKernelRadius.w);

            //            float baseColorWeight = hitDistFactor * 0.5 + 0.5;

            float baseNormalWeight = invAccumFactor * 16.0;
            float basePlaneDistWeight = invAccumFactor * -512.0;

            vec4 diffResult = centerDiff;
            vec4 specResult = centerSpec;

            float weightSum = 1.0;
            float centerLuma = diffResult.w;

            #if GI_DENOISE_PASS == 1
            float edgeWeightSum = 0.0;
            float moment1 = centerLuma;
            float moment2 = pow2(centerLuma);
            #endif

            float sigma = 0.69;
            sigma += kernelRadius * 2.0 * (1.0 - saturate(hitDistFactor));
            sigma *= 1.0 - filteredInputVariance.x;

            // Stretch kernel based on view angle and normal
            vec2 stretchFactor = vec2(1.0);
            vec3 V = normalize(-centerGeomData.viewPos);
            float NoV = abs(dot(centerGeomData.geomNormal, V));
            stretchFactor = mix(1.0 - abs(centerGeomData.geomNormal.xy), stretchFactor, NoV);

            vec2 kernelRadius2 = kernelRadius * stretchFactor;

            float angle = blurJitter.x * PI_2;
            vec2 dir = vec2(cos(angle), sin(angle));
            float rcpSamples = 1.0 / float(GI_DENOISE_SAMPLES);

            #ifdef SETTING_DENOISER_SPATIAL
            for (uint i = 0u; i < GI_DENOISE_SAMPLES; ++i) {
                dir *= MAT2_GOLDEN_ANGLE;
                float baseRadius = sqrt((float(i) + blurJitter.y) * rcpSamples);
                vec2 offsetTexel = dir * (baseRadius * kernelRadius2);
                vec2 offsetUV = offsetTexel * uval_mainImageSizeRcp;
                vec2 sampleUV = centerScreenPos + offsetUV;
                if (saturate(sampleUV) != sampleUV) {
                    sampleUV = _gi_mirrorUV(sampleUV);
                }
                float kernelWeight = gaussianKernel(baseRadius, sigma);
                ivec2 sampleTexelPos = ivec2(sampleUV * uval_mainImageSize);

                vec4 diffSample = _gi_readDiff(sampleTexelPos);
                GeomData geomData = _gi_readGeomData(sampleTexelPos, sampleUV);
                float edgeWeight = 1.0;
                edgeWeight *= planeDistanceWeight(
                    centerGeomData.viewPos,
                    centerGeomData.geomNormal,
                    geomData.viewPos,
                    geomData.geomNormal,
                    basePlaneDistWeight
                );
                edgeWeight *= normalWeight(centerGeomData, geomData, baseNormalWeight);

                float sampleLuma = diffSample.a;
                // float lumaDiff = pow2(centerLuma - sampleLuma);
                // float colorWeight = smoothstep(baseColorWeight, 0.0, lumaDiff);
                #if GI_DENOISE_PASS == 1
                moment1 += sampleLuma * edgeWeight;
                moment2 += pow2(sampleLuma) * edgeWeight;
                edgeWeightSum += edgeWeight;
                #endif


                float totalWeight = kernelWeight * smoothstep(0.0, 1.0, edgeWeight);

                diffResult += diffSample * totalWeight;
                weightSum += totalWeight;
            }
            #endif

            float rcpWeightSum = 1.0 / weightSum;
            diffResult *= rcpWeightSum;
            specResult *= rcpWeightSum;

            float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 5u + GI_DENOISE_PASS), frameCounter);
            diffResult = dither_fp16(diffResult, ditherNoise);
            specResult = dither_fp16(specResult, ditherNoise);

            #if GI_DENOISE_PASS == 1
            float rcpEdgeWeightSum = 1.0 / (edgeWeightSum + 1.0);
            moment1 *= rcpEdgeWeightSum;
            moment2 *= rcpEdgeWeightSum;
            float variance = max(0.0, moment2 - pow2(moment1));
            vec4 newVariance = vec4(filteredInputVariance + vec2(variance), 0.0, 0.0);
            transient_gi_denoiseVariance2_store(texelPos, newVariance);


            #if SETTING_DEBUG_OUTPUT
            if (RANDOM_FRAME < MAX_FRAMES){
                // imageStore(uimg_temp3, texelPos, vec4(linearStep(baseKernelRadius.z, baseKernelRadius.w, kernelRadius)));
                imageStore(uimg_temp3, texelPos, hitDistanceFactors.xxxx);
                // imageStore(uimg_temp3, texelPos, sigma.xxxx);
            }
            #endif
            transient_gi_blurDiff1_store(texelPos, diffResult);
            transient_gi_blurSpec1_store(texelPos, specResult);
            #elif GI_DENOISE_PASS == 2
            transient_gi_blurDiff2_store(texelPos, diffResult);
            transient_gi_blurSpec2_store(texelPos, specResult);

            gi_historyData_unpack1(historyData, transient_gi1Reprojected_fetch(texelPos));
            gi_historyData_unpack2(historyData, transient_gi2Reprojected_fetch(texelPos));
            gi_historyData_unpack3(historyData, transient_gi3Reprojected_fetch(texelPos));
            gi_historyData_unpack4(historyData, transient_gi4Reprojected_fetch(texelPos));

            historyData.diffuseColor = diffResult.rgb;
            historyData.specularColor = specResult.rgb;

            vec4 packedData1 = clamp(gi_historyData_pack1(historyData), 0.0, FP16_MAX);
            packedData1 = dither_fp16(packedData1, ditherNoise);
            vec4 packedData2 = clamp(gi_historyData_pack2(historyData), 0.0, FP16_MAX);
            packedData2 = dither_fp16(packedData2, ditherNoise);
            vec4 packedData3 = clamp(gi_historyData_pack3(historyData), 0.0, FP16_MAX);
            packedData3 = dither_fp16(packedData3, ditherNoise);
            vec4 packedData4 = clamp(gi_historyData_pack4(historyData), 0.0, FP16_MAX);
            packedData4 = dither_fp16(packedData4, ditherNoise);
            vec4 packedData5 = gi_historyData_pack5(historyData);
            packedData5 = dither_u8(packedData5, ditherNoise);

            history_gi1_store(texelPos, packedData1);
            history_gi2_store(texelPos, packedData2);
            history_gi3_store(texelPos, packedData3);
            history_gi4_store(texelPos, packedData4);
            history_gi5_store(texelPos, packedData5);
            #endif
        }
    }
}