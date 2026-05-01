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
    float viewZ = texelFetch(usam_gbufferSolidViewZ, texelPos, 0).r;
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

void getSpecularKernelBasis(
    vec3 viewPos, vec3 N, float roughness, float worldRadius,
    float hitDistFactor, float accumFactor,
    out vec3 T, out vec3 B
) {
    vec3 V = -normalize(viewPos);
    vec3 D = getSpecularDominantDirection(N, V, roughness);

    float bentFactor = sqrt(hitDistFactor);
    vec3 bentD = normalize(mix(N, D, bentFactor));

    vec3 R = reflect(-bentD, N);
    T = normalize(cross(N, R));
    if (length(T) < 0.001) {
        T = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    }
    B = cross(R, T);

    float NoD = saturate(dot(N, D));
    float skewFactor = mix(0.25 + 0.75 * roughness, 1.0, NoD);
    skewFactor = mix(skewFactor, 1.0, accumFactor);
    skewFactor = mix(1.0, skewFactor, bentFactor);

    T *= worldRadius * skewFactor;
    B *= worldRadius / skewFactor;
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
        #if GI_DENOISE_PASS == 2
        history_viewZ_store(texelPos, vec4(centerGeomData.viewPos.z));
        history_roughness_store(texelPos, vec4(centerGeomData.roughness, 0.0, 0.0, 0.0));
        #endif
        // Load shared memory for variance filtering (20x20 = 400 elements)
        loadSharedVarianceData(workGroupOrigin, gl_LocalInvocationIndex);
        loadSharedVarianceData(workGroupOrigin, gl_LocalInvocationIndex + 256u);

        vec4 baseKernelRadius = GI_DENOISE_BLUR_RADIUS;
        vec2 blurJitter = rand_stbnVec2(texelPos + GI_DENOISE_RAND_NOISE_OFFSET, frameCounter);

        barrier();

        if (centerGeomData.viewPos.z > -65536.0) {
            #if GI_DENOISE_PASS == 2
            history_geomViewNormal_store(texelPos, vec4(centerGeomData.geomNormal * 0.5 + 0.5, 0.0));
            history_viewNormal_store(texelPos, vec4(centerGeomData.normal * 0.5 + 0.5, 0.0));
            #endif

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

            vec2 hitDistFactor = hitDistanceFactors;
            #if GI_DENOISE_PASS == 2
            hitDistFactor = pow2(hitDistFactor);
            #endif
            hitDistFactor = hitDistFactor * 0.9 + 0.1;

            float16_t jitterR = float16_t(blurJitter.y);
            float angle = blurJitter.x * PI_2;
            float16_t rcpSamples = float16_t(1.0 / float(GI_DENOISE_SAMPLES));

            GBufferData centerGData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), centerGData);
            gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), centerGData);
            Material material = material_decode(centerGData);

            // --- Diffuse loop: screen-space kernel with view-angle stretch ---
            if (material.dielectric > 0.0) {
                float kernelRadius = baseKernelRadius.x;
                kernelRadius *= accumFactor;
                kernelRadius *= 1.0 + filteredInputVariance.x * baseKernelRadius.y;
                kernelRadius *= hitDistFactor.x;
                kernelRadius = clamp(kernelRadius, baseKernelRadius.z, baseKernelRadius.w);

                vec3 V = normalize(-centerGeomData.viewPos);
                float NoV = abs(dot(centerGeomData.geomNormal, V));
                vec2 stretchFactor = mix(1.0 - abs(centerGeomData.geomNormal.xy), vec2(1.0), NoV);
                f16vec2 kernelRadius2 = f16vec2(kernelRadius * stretchFactor);

                float sigmaFP32 = 0.69;
                // sigmaFP32 += 1.0 - saturate(hitDistFactor.x);
                sigmaFP32 *= 1.0 - filteredInputVariance.x;
                float16_t sigma = float16_t(-sigmaFP32);

                vec4 centerDiff = _gi_readDiff(texelPos);
                f16vec4 diffSumFP16 = f16vec4(centerDiff);
                float16_t weightSumFP16 = float16_t(1.0);
                float16_t centerLuma = diffSumFP16.w;

                #if GI_DENOISE_PASS == 1
                float16_t edgeWeightSumFP16 = float16_t(0.0);
                float16_t moment1FP16 = centerLuma;
                float16_t moment2FP16 = centerLuma * centerLuma;
                #endif

                f16vec2 dir = f16vec2(cos(angle), sin(angle));
                for (uint i = 0u; i < GI_DENOISE_SAMPLES; ++i) {
                    f16vec2 tempDir = dir;
                    dir.x = dot(tempDir, f16vec2(-0.737368878, -0.675490294));
                    dir.y = dot(tempDir, f16vec2(0.675490294, -0.737368878));
                    float16_t baseRadius = sqrt((float16_t(i) + jitterR) * rcpSamples);
                    f16vec2 offsetTexel = dir * (baseRadius * kernelRadius2);
                    vec2 sampleUV = centerScreenPos + vec2(offsetTexel) * uval_mainImageSizeRcp;
                    if (saturate(sampleUV) != sampleUV) {
                        sampleUV = _gi_mirrorUV(sampleUV);
                    }
                    float16_t kernelWeight = exp2(sigma * pow2(baseRadius));
                    ivec2 sampleTexelPos = ivec2(sampleUV * uval_mainImageSize);

                    GeomData geomData = _gi_readGeomData(sampleTexelPos, sampleUV);

                    float baseNormalWeight = invAccumFactor * 64.0 + 32.0;
                    float basePlaneDistWeight = invAccumFactor * -256.0 - 128.0;
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
                    #if GI_DENOISE_PASS == 1
                    float16_t sampleLuma = diffSample.a;
                    moment1FP16 += sampleLuma * edgeWeight;
                    moment2FP16 += pow2(sampleLuma) * edgeWeight;
                    edgeWeightSumFP16 += edgeWeight;
                    #endif

                    float16_t totalWeight = float16_t(kernelWeight * smoothstep(0.0, 1.0, edgeWeight));
                    diffSumFP16 += diffSample * totalWeight;
                    weightSumFP16 += totalWeight;
                }

                {
                    vec4 diffResult = vec4(diffSumFP16);
                    float weightSum = float(weightSumFP16);

                    diffResult *= rcp(weightSum);

                    float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 5u + GI_DENOISE_PASS), frameCounter);
                    diffResult = dither_fp16(diffResult, ditherNoise);

                    #if GI_DENOISE_PASS == 1
                    transient_gi_blurDiff1_store(texelPos, diffResult);

                    float edgeWeightSum = float(edgeWeightSumFP16);
                    float moment1 = float(moment1FP16);
                    float moment2 = float(moment2FP16);
                    float rcpEdgeWeightSum = rcp(edgeWeightSum + 1.0);
                    moment1 *= rcpEdgeWeightSum;
                    moment2 *= rcpEdgeWeightSum;
                    float variance = max(0.0, moment2 - pow2(moment1));
                    filteredInputVarianceFP16.x += float16_t(variance);;

                    #elif GI_DENOISE_PASS == 2
                    transient_gi_diffShadingOutput_store(texelPos, diffResult);

                    vec4 packedData1 = transient_gi1Reprojected_fetch(texelPos);
                    packedData1.rgb = diffResult.rgb;
                    packedData1 = dither_fp16(packedData1, ditherNoise);
                    packedData1 = clamp(packedData1, 0.0, FP16_MAX);
                    history_gi1_store(texelPos, packedData1);
                    #endif
                }
            } else {
                #if GI_DENOISE_PASS == 1
                transient_gi_blurDiff1_store(texelPos, vec4(0.0));
                #elif GI_DENOISE_PASS == 2
                transient_gi_diffShadingOutput_store(texelPos, vec4(0.0));

                vec4 packedData1 = transient_gi1Reprojected_fetch(texelPos);
                packedData1.rgb = vec3(0.0);
                history_gi1_store(texelPos, packedData1);
                #endif
            }

            // --- Specular loop: world-space specular lobe kernel ---
            {
                float kernelRadius = baseKernelRadius.x;
                kernelRadius *= accumFactor;
                kernelRadius *= 1.0 + filteredInputVariance.y * baseKernelRadius.y;
                //kernelRadius *= hitDistFactor.y;
                kernelRadius = clamp(kernelRadius, baseKernelRadius.z, baseKernelRadius.w);
                float worldRadius = kernelRadius * abs(centerGeomData.viewPos.z) * uval_mainImageSizeRcp.y;
                vec3 specTFP32, specBFP32;
                getSpecularKernelBasis(
                    centerGeomData.viewPos,
                    centerGeomData.normal,
                    centerGeomData.roughness,
                    worldRadius,
                    hitDistFactor.y,
                    accumFactor,
                    specTFP32,
                    specBFP32
                );
                f16vec3 specT = f16vec3(specTFP32);
                f16vec3 specB = f16vec3(specBFP32);

                float sigmaFP32 = 0.69;
                // sigmaFP32 += 1.0 - saturate(hitDistFactor.y);
                sigmaFP32 *= 1.0 - filteredInputVariance.y;
                float16_t sigma = float16_t(-sigmaFP32);

                vec4 centerSpec = _gi_readSpec(texelPos);
                f16vec4 spedSumFP16 = f16vec4(centerSpec);
                float16_t weightSumFP16 = float16_t(1.0);
                float16_t centerLuma = spedSumFP16.w;

                #if GI_DENOISE_PASS == 1
                float16_t edgeWeightSumFP16 = float16_t(0.0);
                float16_t moment1FP16 = centerLuma;
                float16_t moment2FP16 = centerLuma * centerLuma;
                #endif

                f16vec2 dir = f16vec2(cos(angle), sin(angle));
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

                    float baseNormalWeight = invAccumFactor * 96.0 + 48.0;
                    float basePlaneDistWeight = invAccumFactor * -384.0 - 192.0;
                    float edgeWeightFP32 = normalWeight(centerGeomData, geomData, baseNormalWeight);
                    edgeWeightFP32 *= planeDistanceWeight(
                        centerGeomData.viewPos,
                        centerGeomData.geomNormal,
                        geomData.viewPos,
                        geomData.geomNormal,
                        basePlaneDistWeight
                    );
                    edgeWeightFP32 *= gi_roughnessWeight(centerGeomData.roughness, geomData.roughness);
                    float16_t edgeWeight = float16_t(edgeWeightFP32);

                    f16vec4 specSample = f16vec4(_gi_readSpec(sampleTexelPos));
                    #if GI_DENOISE_PASS == 1
                    float16_t sampleLuma = specSample.a;
                    moment1FP16 += sampleLuma * edgeWeight;
                    moment2FP16 += pow2(sampleLuma) * edgeWeight;
                    edgeWeightSumFP16 += edgeWeight;
                    #endif

                    float16_t totalWeight = float16_t(kernelWeight * smoothstep(0.0, 1.0, edgeWeight));
                    spedSumFP16 += specSample * totalWeight;
                    weightSumFP16 += totalWeight;
                }

                {
                    vec4 specResult = vec4(spedSumFP16);
                    float weightSum = float(weightSumFP16);

                    specResult *= rcp(weightSum);

                    float ditherNoise = rand_stbnVec1(rand_newStbnPos(texelPos, 7u + GI_DENOISE_PASS), frameCounter);
                    specResult = dither_fp16(specResult, ditherNoise);

                    #if GI_DENOISE_PASS == 1
                    transient_gi_blurSpec1_store(texelPos, specResult);

                    float edgeWeightSum = float(edgeWeightSumFP16);
                    float moment1 = float(moment1FP16);
                    float moment2 = float(moment2FP16);
                    float rcpEdgeWeightSum = rcp(edgeWeightSum + 1.0);
                    moment1 *= rcpEdgeWeightSum;
                    moment2 *= rcpEdgeWeightSum;
                    float variance = max(0.0, moment2 - pow2(moment1));
                    filteredInputVarianceFP16.y += float16_t(variance);;

                    #elif GI_DENOISE_PASS == 2
                    transient_gi_specShadingOutput_store(texelPos, specResult);

                    vec4 packedData3 = transient_gi3Reprojected_fetch(texelPos);
                    packedData3.rgb = specResult.rgb;
                    packedData3 = dither_fp16(packedData3, ditherNoise);
                    packedData3 = clamp(packedData3, 0.0, FP16_MAX);
                    history_gi3_store(texelPos, packedData3);
                    #endif
                }
            }

            #if GI_DENOISE_PASS == 1
            vec4 newVariance = vec4(vec2(filteredInputVarianceFP16), 0.0, 0.0);
            transient_gi_denoiseVariance2_store(texelPos, newVariance);
            #elif GI_DENOISE_PASS == 2
            vec4 packedData5 = transient_gi5Reprojected_fetch(texelPos);
            history_gi5_store(texelPos, packedData5);
            #endif

            return;
        }
    }

    #if GI_DENOISE_PASS == 2
    history_geomViewNormal_store(texelPos, vec4(0.0));
    history_viewNormal_store(texelPos, vec4(0.0));
    #endif
}