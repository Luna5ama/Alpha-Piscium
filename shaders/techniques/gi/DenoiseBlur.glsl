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
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"
#include "/util/Rand.glsl"
#include "/util/Coords.glsl"
#include "/util/Mat2.glsl"
#include "/util/Rand.glsl"
#include "/util/Dither.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;
layout(rgba8) uniform restrict writeonly image2D uimg_rgba8;
layout(rgba16f) uniform restrict writeonly image2D uimg_temp3;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;

// Shared memory with padding for 5x5 tap (-2 to +2)
// Each work group is 16x16, need +2 padding on each side for 5x5 taps
shared vec2 shared_varianceData[20][20];

struct GeomData {
    vec3 geomNormal;
    vec3 normal;
    vec3 viewPos;
    float roughness;
};

GeomData _gi_readGeomData(ivec2 texelPos, vec2 screenPos) {
    GeomData geomData;
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    geomData.viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    geomData.geomNormal = transient_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0;
    geomData.normal = transient_viewNormal_fetch(texelPos).xyz * 2.0 - 1.0;
    geomData.roughness = pow2(transient_specularPBRData_fetch(texelPos).r);
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

vec3 getSpecularDominantDirection(vec3 N, vec3 V, float roughness) {
    float f = (1.0 - roughness) * (sqrt(1.0 - roughness) + roughness);
    vec3 R = reflect(-V, N);
    return normalize(mix(N, R, f));
}

void getSpecularKernelBasis(vec3 viewPos, vec3 N, float roughness, float worldRadius,
                            out vec3 T, out vec3 B) {
    vec3 V = -normalize(viewPos);
    vec3 D = getSpecularDominantDirection(N, V, roughness);
    vec3 R = reflect(-D, N);
    T = normalize(cross(N, R));
    if (length(T) < 0.001) {
        T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    }
    B = cross(R, T);
    T *= worldRadius;
    B *= worldRadius;
    float NoV = abs(dot(N, V));
    float angle = saturate(acos(NoV) / (PI * 0.5));
    float skewFactor = mix(1.0, roughness, angle);
    T *= skewFactor;
}

float getRoughnessWeight(float roughness0, float roughness) {
    float norm = roughness0 * roughness0 * 0.99 + 0.01;
    float w = abs(roughness0 - roughness) / norm;
    return saturate(1.0 - w);
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
        history_viewZ_store(texelPos, vec4(centerGeomData.viewPos.z));
        // Load shared memory for variance filtering (20x20 = 400 elements)
        loadSharedVarianceData(workGroupOrigin, gl_LocalInvocationIndex);
        loadSharedVarianceData(workGroupOrigin, gl_LocalInvocationIndex + 256u);

        vec4 baseKernelRadius = GI_DENOISE_BLUR_RADIUS;
        vec2 blurJitter = rand_stbnVec2(texelPos + GI_DENOISE_RAND_NOISE_OFFSET, frameCounter);

        barrier();

        if (centerGeomData.viewPos.z > -65536.0) {
            history_geomViewNormal_store(texelPos, vec4(centerGeomData.geomNormal * 0.5 + 0.5, 0.0));
            history_viewNormal_store(texelPos, vec4(centerGeomData.normal * 0.5 + 0.5, 0.0));

            vec2 hitDistanceFactors = transient_gi_hitDistanceFactors_fetch(texelPos).xy;
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
            f16vec2 filteredInputVarianceFP16 = f16vec2(filteredInputVariance);

            // Just load it again later to save registers
            float historyLength = transient_gi5Reprojected_fetch(texelPos).x * TOTAL_HISTORY_LENGTH;
            float accumFactor = rcp(1.0 + pow2(0.1 * historyLength));
            float invAccumFactor = saturate(1.0 - accumFactor); // Increases as history accumulates

            float hitDistFactor = hitDistanceFactors.x;
            #if GI_DENOISE_PASS == 2
            hitDistFactor = pow2(hitDistFactor);
            #endif
            hitDistFactor = hitDistFactor * 0.9 + 0.1;

            float kernelRadius = baseKernelRadius.x;
            kernelRadius *= accumFactor;

            kernelRadius *= 1.0 + filteredInputVariance.x * baseKernelRadius.y;
            kernelRadius *= hitDistFactor;
            kernelRadius = clamp(kernelRadius, baseKernelRadius.z, baseKernelRadius.w);

            vec4 centerDiff = _gi_readDiff(texelPos);
            vec4 centerSpec = _gi_readSpec(texelPos);
            f16vec4 diffResultFP16 = f16vec4(centerDiff);
            f16vec4 specResultFP16 = f16vec4(centerSpec);

            float16_t weightSumFP16 = float16_t(1.0);
            float16_t specWeightSumFP16 = float16_t(1.0);
            float16_t centerLuma = diffResultFP16.w;

            float worldRadius = kernelRadius * abs(centerGeomData.viewPos.z) * uval_mainImageSizeRcp.y;

            vec3 specTFP32, specBFP32;
            getSpecularKernelBasis(
                centerGeomData.viewPos,
                centerGeomData.normal,
                centerGeomData.roughness,
                worldRadius,
                specTFP32,
                specBFP32
            );

            f16vec3 specT = f16vec3(specTFP32);
            f16vec3 specB = f16vec3(specBFP32);

            #if GI_DENOISE_PASS == 1
            float16_t edgeWeightSumFP16 = float16_t(0.0);
            float16_t moment1FP16 = centerLuma;
            float16_t moment2FP16 = centerLuma * centerLuma;
            #endif

            float sigmaFP32 = 0.69;
            sigmaFP32 += kernelRadius * 2.0 * (1.0 - saturate(hitDistFactor));
            sigmaFP32 *= 1.0 - filteredInputVariance.x;

            float16_t sigma = float16_t(-sigmaFP32);
            float16_t jitterR = float16_t(blurJitter.y);

            float angle = blurJitter.x * PI_2;
            f16vec2 dir = f16vec2(cos(angle), sin(angle));
            float16_t rcpSamples = float16_t(1.0 / float(GI_DENOISE_SAMPLES));

            #ifdef SETTING_DENOISER_SPATIAL
            for (uint i = 0u; i < GI_DENOISE_SAMPLES; ++i) {
                f16vec2 tempDir = dir;
                dir.x = dot(tempDir, f16vec2(-0.737368878, -0.675490294));
                dir.y = dot(tempDir, f16vec2(0.675490294, -0.737368878));
                float16_t baseRadius = sqrt((float16_t(i) + jitterR) * rcpSamples);

                vec3 sampleView = centerGeomData.viewPos + vec3(specT * dir.x + specB * dir.y) * float(baseRadius);
                vec4 sampleClip = global_camProj * vec4(sampleView, 1.0);
                vec2 sampleUV = sampleClip.xy / sampleClip.w * 0.5 + 0.5;

                if (saturate(sampleUV) != sampleUV) {
                    sampleUV = _gi_mirrorUV(sampleUV);
                }
                float16_t kernelWeight = exp2(sigma * pow2(baseRadius));
                ivec2 sampleTexelPos = ivec2(sampleUV * uval_mainImageSize);

                GeomData geomData = _gi_readGeomData(sampleTexelPos, sampleUV);

                // Cheap enough to just recompute it to save 1 extra register
                float baseNormalWeight = invAccumFactor * 256.0;
                float basePlaneDistWeight = invAccumFactor * -512.0;
                float edgeWeightFP32 = normalWeight(centerGeomData, geomData, baseNormalWeight);
                edgeWeightFP32 *= planeDistanceWeight(
                    centerGeomData.viewPos,
                    centerGeomData.geomNormal,
                    geomData.viewPos,
                    geomData.geomNormal,
                    basePlaneDistWeight
                );

                float16_t edgeWeight = float16_t(edgeWeightFP32);

                f16vec4 diffSample = f16vec4(_gi_readDiff(sampleTexelPos));
                f16vec4 specSample = f16vec4(_gi_readSpec(sampleTexelPos));
                float16_t sampleLuma = diffSample.a;
                #if GI_DENOISE_PASS == 1
                moment1FP16 += sampleLuma * edgeWeight;
                moment2FP16 += pow2(sampleLuma) * edgeWeight;
                edgeWeightSumFP16 += edgeWeight;
                #endif

                float16_t totalWeight = float16_t(kernelWeight * smoothstep(0.0, 1.0, edgeWeight));

                float roughnessW = getRoughnessWeight(centerGeomData.roughness, geomData.roughness);
                float16_t specTotalWeight = totalWeight * float16_t(roughnessW);

                diffResultFP16 += diffSample * totalWeight;
                weightSumFP16 += totalWeight;

                specResultFP16 += specSample * specTotalWeight;
                specWeightSumFP16 += specTotalWeight;
            }
            #endif

            vec4 diffResult = vec4(diffResultFP16);
            vec4 specResult = vec4(specResultFP16);
            float weightSum = float(weightSumFP16);
            float specWeightSum = float(specWeightSumFP16);
            #if GI_DENOISE_PASS == 1
            float edgeWeightSum = float(edgeWeightSumFP16);
            float moment1 = float(moment1FP16);
            float moment2 = float(moment2FP16);
            #endif

            diffResult *= 1.0 / weightSum;
            specResult *= 1.0 / specWeightSum;

            float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 5u + GI_DENOISE_PASS), frameCounter);
            diffResult = dither_fp16(diffResult, ditherNoise);
            specResult = dither_fp16(specResult, ditherNoise);

            #if GI_DENOISE_PASS == 1
            transient_gi_blurDiff1_store(texelPos, diffResult);
            transient_gi_blurSpec1_store(texelPos, specResult);
            float rcpEdgeWeightSum = 1.0 / (edgeWeightSum + 1.0);
            moment1 *= rcpEdgeWeightSum;
            moment2 *= rcpEdgeWeightSum;
            float variance = max(0.0, moment2 - pow2(moment1));
            vec4 newVariance = vec4(vec2(filteredInputVarianceFP16) + vec2(variance), 0.0, 0.0);
            transient_gi_denoiseVariance2_store(texelPos, newVariance);


            #if SETTING_DEBUG_OUTPUT
            if (RANDOM_FRAME < MAX_FRAMES){
                //                imageStore(uimg_temp3, texelPos, vec4(linearStep(baseKernelRadius.z, baseKernelRadius.w, kernelRadius)));
//                imageStore(uimg_temp3, texelPos, hitDistanceFactors.xxxx);
                // imageStore(uimg_temp3, texelPos, sigma.xxxx);
            }
            #endif
            #elif GI_DENOISE_PASS == 2
            transient_gi_diffShadingOutput_store(texelPos, diffResult);
            transient_gi_specShadingOutput_store(texelPos, specResult);

            vec4 packedData1 = transient_gi1Reprojected_fetch(texelPos);
            packedData1.rgb = diffResult.rgb;
            packedData1 = dither_fp16(packedData1, ditherNoise);
            packedData1 = clamp(packedData1, 0.0, FP16_MAX);
            history_gi1_store(texelPos, packedData1);

            vec4 packedData3 = transient_gi3Reprojected_fetch(texelPos);
            packedData3.rgb = specResult.rgb;
            packedData3 = dither_fp16(packedData3, ditherNoise);
            packedData3 = clamp(packedData3, 0.0, FP16_MAX);
            history_gi3_store(texelPos, packedData3);

            vec4 packedData5 = transient_gi5Reprojected_fetch(texelPos);
            history_gi5_store(texelPos, packedData5);
            #endif

            return;
        }
    }

    history_geomViewNormal_store(texelPos, vec4(0.0));
    history_viewNormal_store(texelPos, vec4(0.0));
}