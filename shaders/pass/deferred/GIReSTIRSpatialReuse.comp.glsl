/*
    References:
        [BIT22] Bitterli, Benedikt. "Correlations and Reuse for Fast and Accurate Physically Based Light Transport".
            Dartmouth College Ph.D Dissertations 2022.
            https://digitalcommons.dartmouth.edu/cgi/viewcontent.cgi?article=1080&context=dissertations
        [WYM23] Wyman, Chris, et al. "A Gentle Introduction to ReSTIR". SIGGRAPH 2023.
            https://intro-to-restir.cwyman.org/
        [ANA23] Anagnostou, Kostas. "A Gentler Introduction to ReSTIR". Interplay of Light. 2023.
            https://interplayoflight.wordpress.com/2023/12/17/a-gentler-introduction-to-restir/
        [ALE22] Alegruz. "Screen-Space-ReSTIR-GI". GitHub. 2022.
            https://github.com/Alegruz/Screen-Space-ReSTIR-GI
            BSD 3-Clause License. Copyright (c) 2022, Alegruz.

        You can find full license texts in /licenses

    Other Credits:
        - Belmu (https://github.com/BelmuTM) - Advice on ReSTIR.
*/
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/techniques/SST2.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/util/BSDF.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(std430, binding = 5) buffer RayData {
    uvec4 ssbo_rayData[];
};

layout(std430, binding = 6) buffer RayDataIndices {
    uint ssbo_rayDataIndices[];
};

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

shared uint shared_rayCount[16];

// Evaluate combined diffuse + specular BRDF
vec3 evalBRDF(vec3 normal, vec3 lightDir, vec3 viewDir, Material material) {
    vec3 H = normalize(lightDir + viewDir);
    float NdotL = saturate(dot(normal, lightDir));
    float NdotV = saturate(dot(normal, viewDir));
    float NdotH = saturate(dot(normal, H));
    float LdotH = saturate(dot(lightDir, H));

    vec3 fresnel = fresnel_evalMaterial(material, LdotH);
    float diffuseBRDF = (1.0 - material.metallic) * NdotL * RCP_PI;
    float specularBRDF = bsdf_ggx(material, NdotL, NdotV, NdotH);

    return ((1.0 - fresnel) * diffuseBRDF + fresnel * specularBRDF);
}

#if USE_REFERENCE || !defined(SETTING_GI_SPATIAL_REUSE)
void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        SpatialSampleData sampleData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));
        history_restir_prevSample_store(texelPos, sampleData.sampleValue);
        history_restir_prevHitNormal_store(texelPos, vec4(sampleData.hitNormal * 0.5 + 0.5, 0.0));
    }
}
#else
void main() {
    sst_init(SETTING_GI_SST_THICKNESS);
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    uvec2 binId = swizzledWGPos >> 1u;
    uint numBinX = (uval_mainImageSizeI.x + 31) >> 5; // 32x32 bin
    uint binIdx = binId.y * numBinX + binId.x;
    ivec2 binLocalPos = texelPos & 31; // 32x32 bin
    uint binLocalIndex = sst2_encodeBinLocalIndex(binLocalPos);
    uint binWriteBaseIndex = binIdx * 1024;
    uint dataIndex = binWriteBaseIndex + binLocalIndex;
    uint rayIndex = 0xFFFFFFFFu;

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        SpatialSampleData centerSampleData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));
        history_restir_prevSample_store(texelPos, centerSampleData.sampleValue);
        history_restir_prevHitNormal_store(texelPos, vec4(centerSampleData.hitNormal * 0.5 + 0.5, 0.0));
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);
        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
            vec3 V = normalize(-viewPos);
            float NDotV = saturate(dot(centerSampleData.normal, V));

            uvec4 reprojectedData;
            if (bool(frameCounter & 1)) {
                reprojectedData = history_restir_reservoirTemporal1_fetch(texelPos);
            } else {
                reprojectedData = history_restir_reservoirTemporal2_fetch(texelPos);
            }
            ReSTIRReservoir spatialReservoir = restir_reservoir_unpack(reprojectedData);

            #ifdef SETTING_GI_SPATIAL_REUSE_COUNT_DYNAMIC
            const uint reuseCount = uint(mix(float(SETTING_GI_SPATIAL_REUSE_COUNT), 1.0, sqrt(linearStep(0.0, 0.5, transient_gi5Reprojected_fetch(texelPos).y))));
            #else
            const uint reuseCount = uint(SETTING_GI_SPATIAL_REUSE_COUNT);
            #endif
            vec2 texelPosF = vec2(texelPos) + vec2(0.5);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            Material material = material_decode(gData);

            // Compute specular bounce probability to modulate kernel size
            vec3 fresnelV = fresnel_evalMaterial(material, NDotV);
            float pSpec;
            if (material.metallic > 0.5) {
                pSpec = 1.0;
            } else {
                vec3 fresnelT = vec3(1.0) - fresnelV;
                vec3 totalEnergy = material.albedo * fresnelT + fresnelV;
                pSpec = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, fresnelV / max(totalEnergy, vec3(1e-5)));
            }
            pSpec = saturate(pSpec);
            // Shrink kernel for specular surfaces: pSpec=1 + low roughness = small kernel
            // Keep kernel large for diffuse surfaces: pSpec=0 or high roughness = large kernel
            float kernelModulator = mix(1.0, material.roughness, pSpec);
            float reuseRadiusFP32 = float(SETTING_GI_SPATIAL_REUSE_RADIUS) * kernelModulator;
            reuseRadiusFP32 = max(reuseRadiusFP32, 2.0);
            float16_t reuseRadius = float16_t(reuseRadiusFP32);

            float pHatMe = centerSampleData.sampleValue.w;
            vec4 originalSample = vec4(centerSampleData.sampleValue.xyz, pHatMe);
            float rcAvgWY = max(spatialReservoir.avgWY, 0.0);
            float rcM = spatialReservoir.m;
            float rcMDivK = rcM / max(float(reuseCount), 1.0); // rc.M / k [BIT22 Algo.8]

            // Pairwise MIS state [BIT22 Algo.8]
            float spatialWSum = 0.0;
            float mc = 1.0; // Canonical MIS weight accumulator, starts at 1 [line 3]
            uint numValidNeighbors = 0u;
            vec4 rcY = spatialReservoir.Y; // Canonical sample before loop updates it

            // Precompute for pi(rc.y) Jacobian evaluation
            vec3 centerHitViewPos = viewPos + rcY.xyz * rcY.w;
            float cosPhiB_canon = -dot(rcY.xyz, centerSampleData.hitNormal);
            float RB2_canon = rcY.w * rcY.w;

            vec4 selectedSampleF = originalSample;

            vec2 noise2 = rand_stbnVec2(texelPos, RANDOM_FRAME);
            float16_t jitterR = float16_t(noise2.y);
            float angle = noise2.x * PI_2;
            f16vec2 dir = f16vec2(cos(angle), sin(angle));
            float16_t rcpSamples = float16_t(1.0 / float(reuseCount));

            for (uint i = 0u; i < reuseCount; ++i) {
                f16vec2 tempDir = dir;
                dir.x = dot(tempDir, f16vec2(-0.737368878, -0.675490294));
                dir.y = dot(tempDir, f16vec2(0.675490294, -0.737368878));
                float16_t baseRadius = sqrt((float16_t(i) + jitterR) * rcpSamples) * reuseRadius;
                f16vec2 offset = dir * baseRadius;

                vec2 sampleTexelPosF = texelPosF + vec2(offset);
                if (clamp(sampleTexelPosF, vec2(0.0), uval_mainImageSizeI - 1.0) != sampleTexelPosF) {
                    continue;
                }
                ivec2 sampleTexelPos = ivec2(sampleTexelPosF);

                if (sampleTexelPos == texelPos) {
                    continue;
                }

                SpatialSampleData neighborData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(sampleTexelPos));

                if (dot(centerSampleData.geomNormal, neighborData.geomNormal) < 0.99) {
                    continue;
                }
                float neighborViewZ = texelFetch(usam_gbufferViewZ, sampleTexelPos, 0).x;
                uvec4 neighborReservoirData;
                if (bool(frameCounter & 1)) {
                    neighborReservoirData = history_restir_reservoirTemporal1_fetch(sampleTexelPos);
                } else {
                    neighborReservoirData = history_restir_reservoirTemporal2_fetch(sampleTexelPos);
                }
                ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(neighborReservoirData);

                if (restir_isReservoirValid(neighborReservoir)) {
                    vec2 neighborScreenPos = sampleTexelPosF * uval_mainImageSizeRcp;
                    vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_camProjInverse);

                    vec3 neighborHitViewPos = neighborViewPos + neighborReservoir.Y.xyz * neighborReservoir.Y.w;
                    vec3 hitDiff = neighborHitViewPos - viewPos;
                    float hitDist2 = dot(hitDiff, hitDiff);

                    // Safety check: Avoid singularity if reuse sample is at the exact same position
                    if (hitDist2 < 1e-6) continue;

                    float neighborSampleHitDistance = sqrt(hitDist2);
                    vec3 neighborSampleDirView = hitDiff / neighborSampleHitDistance;

                    vec3 hitRadiance = neighborData.sampleValue.xyz;
                    vec3 brdfLoopValue = evalBRDF(centerSampleData.normal, neighborSampleDirView, V, material);
                    vec3 neighborSample = hitRadiance * brdfLoopValue;
                    float neighborPHat = length(neighborSample);

                    // offsetB = neighborReservoir.Y.xyz * Y.w, which is already a scaled unit vector
                    // RB2 = dot(offsetB, offsetB) = Y.w^2  (Y.xyz is a unit direction)
                    float RB2 = neighborReservoir.Y.w * neighborReservoir.Y.w;
                    if (RB2 < 1e-6) continue;

                    // normalize(offsetB) == neighborReservoir.Y.xyz (already unit)
                    float cosB = dot(neighborData.normal, neighborReservoir.Y.xyz);
                    float cosPhiB = -dot(neighborReservoir.Y.xyz, neighborData.hitNormal);
                    if (cosB <= 0.0 || cosPhiB <= 0.0) continue;

                    float cosPhiA = -dot(neighborSampleDirView, neighborData.hitNormal);
                    // cosPhiA <= 0 or neighborPHat <= 0 both zero out m, making the reservoir update a no-op
                    if (cosPhiA <= 0.0 || neighborPHat <= 0.0) continue;

                    // All denominator terms are verified positive at this point
                    float jacobian = clamp((RB2 * cosPhiA) / (hitDist2 * cosPhiB), 0.0, 100.0);

                    float pcRiY = neighborPHat * jacobian; // p̂_c(ri.y): center target for neighbor's sample
                    float piRiY = neighborData.sampleValue.w; // p̂_i(ri.y): neighbor's stored target

                    // Pairwise MIS weight mi [BIT22 Algo.8 line 5]
                    float MiPiRiY = neighborReservoir.m * piRiY;
                    float mi = MiPiRiY / max(MiPiRiY + rcMDivK * pcRiY, 1e-10);

                    // Accumulate mc: need pi(rc.y) = center sample evaluated at neighbor domain [BIT22 Algo.8 line 6]
                    {
                        vec3 cHitDiff = centerHitViewPos - neighborViewPos;
                        float cHitDist2 = dot(cHitDiff, cHitDiff);
                        if (cHitDist2 >= 1e-6 && RB2_canon >= 1e-6 && cosPhiB_canon > 0.0) {
                            float cHitDist = sqrt(cHitDist2);
                            vec3 cDirAtNbr = cHitDiff / cHitDist;
                            float cCosPhiA = -dot(cDirAtNbr, centerSampleData.hitNormal);
                            if (cCosPhiA > 0.0) {
                                float jacCn = clamp((RB2_canon * cCosPhiA) / (cHitDist2 * cosPhiB_canon), 0.0, 100.0);
                                // Evaluate BRDF at neighbor for center's sample direction (center material as approx)
                                vec3 VNeighbor = -normalize(neighborViewPos);
                                vec3 cBrdfValue = evalBRDF(neighborData.normal, cDirAtNbr, VNeighbor, material);
                                float piRcY = length(originalSample.xyz * cBrdfValue) * jacCn;
                                float MiPiRcY = neighborReservoir.m * piRcY;
                                mc += 1.0 - MiPiRcY / max(MiPiRcY + rcMDivK * pHatMe, 1e-10);
                            } else {
                                mc += 1.0; // center hit is behind hit surface from neighbor's POV
                            }
                        } else {
                            mc += 1.0; // geometry degenerate, treat pi(rc.y) = 0
                        }
                    }

                    numValidNeighbors++;
                    float neighborWi = pcRiY * max(neighborReservoir.avgWY, 0.0) * mi;
                    float neighborRand = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 4u + i), RANDOM_FRAME);

                    if (restir_updateReservoir(
                        spatialReservoir,
                        spatialWSum,
                        vec4(neighborSampleDirView, neighborSampleHitDistance),
                        neighborWi,
                        neighborReservoir.m,
                        neighborRand
                    )) {
                        selectedSampleF = vec4(hitRadiance, neighborPHat);
                    }
                }
            }

            // Canonical update [BIT22 Algo.8 line 8]: s.update(rc.y, p̂_c(rc.y) * rc.W(rc.y) * mc)
            float canonicalWi = pHatMe * rcAvgWY * mc;
            float canonicalRand = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 4u + reuseCount), RANDOM_FRAME);
            if (restir_updateReservoir(
                spatialReservoir,
                spatialWSum,
                rcY,
                canonicalWi,
                0.0, // M already counted in spatialReservoir.m = rcM
                canonicalRand
            )) {
                selectedSampleF = originalSample;
            }

            vec4 ssgiOut = vec4(0.0, 0.0, 0.0, -1.0);
            vec4 ssgiSpecOut = vec4(0.0, 0.0, 0.0, -1.0);
            ReSTIRReservoir resultReservoir = spatialReservoir;
            // Pairwise MIS final weight [BIT22 Algo.8 line 10]: W = 1/p̂ * 1/(1+k) * wsum
            float avgWY = spatialWSum * safeRcp(selectedSampleF.w) * safeRcp(float(numValidNeighbors + 1u));
            resultReservoir.avgWY = avgWY;

            vec3 winL_out = resultReservoir.Y.xyz;
            float winHitDist = resultReservoir.Y.w;
            vec3 H_out = normalize(winL_out + V);

            float outNDotL = saturate(dot(centerSampleData.normal, winL_out));
            float outNDotH = saturate(dot(centerSampleData.normal, H_out));
            float outLDotH = saturate(dot(winL_out, H_out));

            vec3 outFresnel = fresnel_evalMaterial(material, outLDotH);
            float lambertianBRDF = outNDotL * RCP_PI;
            float ggxBRDF = bsdf_ggx(material, outNDotL, NDotV, outNDotH);

            vec3 diffuseWeight = (1.0 - material.metallic) * (1.0 - outFresnel) * lambertianBRDF;
            vec3 specularWeight = outFresnel * ggxBRDF;
            vec3 fullBRDF = diffuseWeight + specularWeight;
            vec3 diffRatio3 = diffuseWeight * safeRcp(fullBRDF);

            vec3 totalOutput = selectedSampleF.xyz * fullBRDF * avgWY;
            ssgiOut = vec4(totalOutput * diffRatio3, winHitDist);
            ssgiSpecOut = vec4(totalOutput * (vec3(1.0) - diffRatio3), winHitDist);

            #if SETTING_DEBUG_OUTPUT
            vec4 vvv = vec4(0.0);
            #endif
            if (any(notEqual(selectedSampleF, originalSample))) {
                #if SETTING_DEBUG_OUTPUT
                vvv = vec4(0.0, 1.0, 0.0, 0.0);
                #endif

                SSTRay sstRay;
                if (spatialReservoir.Y.w > 0.0) {
                    vec3 expectHitViewPos = viewPos + spatialReservoir.Y.xyz * spatialReservoir.Y.w;
                    vec3 rayOrigin = coords_viewToScreen(viewPos, global_camProj);
                    vec3 rayEnd = coords_viewToScreen(expectHitViewPos, global_camProj);
                    vec4 rayDirLen = normalizeAndLength(rayEnd - rayOrigin);
                    sstRay = sstray_setup(texelPos, rayOrigin, rayDirLen.xyz, rayDirLen.w);
                } else {
                    sstRay = sstray_setup(texelPos, viewPos, spatialReservoir.Y.xyz);
                }
                sst_trace(sstRay, 4);
                if (sstRay.currT > 0.0) {
                    uvec4 packedData = sstray_pack(sstRay);
                    ssbo_rayData[dataIndex] = packedData;
                    rayIndex = sst2_encodeRayIndexBits(binLocalIndex, sstRay);
                } else {
                    bool discardSptialReuse = true;
                    if (sstRay.currT < -0.99) {
                        discardSptialReuse = false;
                    }

                    if (discardSptialReuse) {
                        resultReservoir = restir_initReservoir();
                        ssgiOut = vec4(0.0);
                        ssgiSpecOut = vec4(0.0);
                        #if SETTING_DEBUG_OUTPUT
                        imageStore(uimg_temp5, texelPos, vec4(0.0, 0.0, 1.0, 0.0));
                        #endif
                    }
                }
            }
            #if SETTING_DEBUG_OUTPUT
            imageStore(uimg_temp5, texelPos, vvv);
            #endif

            ssgiOut.rgb = clamp(ssgiOut.rgb, 0.0, FP16_MAX);
            ssgiSpecOut.rgb = clamp(ssgiSpecOut.rgb, 0.0, FP16_MAX);
            transient_ssgiOut_store(texelPos, ssgiOut);
            transient_ssgiSpecOut_store(texelPos, ssgiSpecOut);
        }
    }
    ssbo_rayDataIndices[dataIndex] = rayIndex;
    uvec4 subgroupRayCountBalllot = subgroupBallot(rayIndex < 0xFFFFFFFFu);
    if (subgroupElect()) {
        shared_rayCount[gl_SubgroupID] = subgroupBallotBitCount(subgroupRayCountBalllot);
    }
    barrier();
    if (gl_SubgroupID == 0u) {
        uint partialRayCount = gl_SubgroupInvocationID < gl_NumSubgroups ? shared_rayCount[gl_SubgroupInvocationID] : 0u;
        uint totalRayCount = subgroupAdd(partialRayCount);
        if (subgroupElect()) {
            transient_spatialReuseRayCount_store(ivec2(swizzledWGPos), vec4(float(totalRayCount)));
        }
    }
}
#endif