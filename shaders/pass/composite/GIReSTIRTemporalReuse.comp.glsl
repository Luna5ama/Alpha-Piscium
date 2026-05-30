/*
    References:
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

#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/InitialSample.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Sampling.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/ThreadGroupTiling.glsl"
#include "/util/BSDF.glsl"
#include "/techniques/gi/PairwiseMIS.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;


shared mat3 shared_prevViewToCurrView;
shared vec3 shared_prevViewToCurrViewTrans;

void sampleTemporalNeighbor(
    ivec2 texelPos,
    ivec2 neighborTexelPos,
    float combinedWeight,
    uint randSeed,
    vec3 viewPos,
    vec3 V,
    vec3 centerNormal,
    ResampleMaterial material,
    bool oddFrame,
    inout ReSTIRReservoir reservoir,
    inout float wSum,
    inout vec4 finalSample,
    inout vec3 finalHitNormal
) {
    if (combinedWeight > 0.0) {
        uvec4 prevTemporalReservoirData = oddFrame
        ? history_restir_reservoirTemporal2_fetch(neighborTexelPos)
        : history_restir_reservoirTemporal1_fetch(neighborTexelPos);
        ReSTIRReservoir neighborReservoir = restir_reservoir_unpack(prevTemporalReservoirData);
        if (restir_isReservoirValid(neighborReservoir)) {

            vec3 neighborHitNormal = vec3(0.0);

            bool valid = true;
            if (neighborReservoir.Y.w > 0.0) {
                vec2 neighborScreenPos = coords_texelToUV(neighborTexelPos, uval_mainImageSizeRcp);
                float neighborViewZ = history_viewZ_fetch(neighborTexelPos).x;
                vec3 neighborViewPos = coords_toViewCoord(neighborScreenPos, neighborViewZ, global_prevCamProjInverse);
                // Save original offset in prev-view space for Jacobian before Y overwrite
                vec3 origOffsetPrevView = neighborReservoir.Y.xyz * neighborReservoir.Y.w;
                vec3 prev2CurrHitViewPos = shared_prevViewToCurrView * (neighborViewPos + origOffsetPrevView) + shared_prevViewToCurrViewTrans;
                vec3 hitDiff = prev2CurrHitViewPos - viewPos;
                float hitDist2 = dot(hitDiff, hitDiff);
                float rcpHitDist = inversesqrt(hitDist2);
                neighborReservoir.Y.xyz = hitDiff * rcpHitDist;
                neighborReservoir.Y.w = hitDist2 * rcpHitDist;

                vec4 prev2CurrHitClipPos = global_camProj * vec4(prev2CurrHitViewPos, 1.0);
                uint clipFlag = uint(prev2CurrHitClipPos.z > 0.0);
                clipFlag &= uint(all(lessThan(abs(prev2CurrHitClipPos.xy), prev2CurrHitClipPos.ww)));
                vec3 prev2CurrHitScreenPos = vec3(prev2CurrHitClipPos.xy / prev2CurrHitClipPos.w * 0.5 + 0.5, prev2CurrHitClipPos.z / prev2CurrHitClipPos.w);
                clipFlag &= uint(saturate(prev2CurrHitScreenPos) == prev2CurrHitScreenPos);

                if (!bool(clipFlag)) {
                    valid = false;
                } else {
                    vec3 neighborHitNormalRaw = history_restir_prevHitNormal_fetch(neighborTexelPos).xyz;
                    neighborHitNormal = normalize(shared_prevViewToCurrView * (neighborHitNormalRaw * 2.0 - 1.0));
                    // offsetB in current view = M * origOffset (translation cancels in subtraction)
                    vec3 offsetB = shared_prevViewToCurrView * origOffsetPrevView;
                    vec3 dirA = neighborReservoir.Y.xyz;
                    float RB2 = dot(offsetB, offsetB);
                    vec3 dirB = offsetB * inversesqrt(max(RB2, 1e-12));
                    float cosPhiA = -dot(dirA, neighborHitNormal);
                    float cosPhiB = -dot(dirB, neighborHitNormal);
                    float jacobian = 1.0;
                    if (cosPhiA <= 0.0 || dot(centerNormal, dirA) <= 0.0) {
                        jacobian = 0.0;
                    } else if (cosPhiB > 5e-2) {
                        jacobian = min((RB2 * cosPhiA) / (hitDist2 * cosPhiB), 256.0);
                    }
                    neighborReservoir.avgWY *= jacobian;
                }
            } else {
                neighborReservoir.Y.xyz = normalize(shared_prevViewToCurrView * neighborReservoir.Y.xyz);
            }

            if (valid) {
                vec4 neighborSample = history_restir_prevSample_fetch(neighborTexelPos);
                float neighborPHat = evalTargetFunction(neighborSample.rgb, centerNormal, neighborReservoir.Y.xyz, V, material);

                neighborReservoir.m *= combinedWeight;
                // Reduces weight further if the target function is much diff from the hisotry footprint
                // Using rcp sqrt instead of rcp to reduce the impact
                float ratio = max(neighborPHat * safeRcp(neighborSample.w), neighborSample.w * safeRcp(neighborPHat));
                neighborReservoir.m *= inversesqrt(max(ratio, 1.0));
                float wi = max(0.0, neighborReservoir.avgWY) * neighborReservoir.m * neighborPHat;
                float neighborRand = restir_updateRand(texelPos, randSeed);

                if (restir_updateReservoir(reservoir, wSum, neighborReservoir.Y, wi, neighborReservoir.m, neighborRand)) {
                    finalSample = vec4(neighborSample.xyz, neighborPHat);
                    finalHitNormal = neighborHitNormal;
                }
            }
        }
    }
}

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (threadIdx == 0u) {
        // Precompute prevViewToCurrView matrix for the workgroup
        shared_prevViewToCurrView = mat3(gbufferModelView) * mat3(gbufferPrevModelViewInverse);
        shared_prevViewToCurrViewTrans = mat3(gbufferModelView) * (gbufferPrevModelViewInverse[3].xyz - uval_cameraDelta) + gbufferModelView[3].xyz;
    }
    barrier();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        ReSTIRReservoir temporalReservoir = restir_initReservoir();
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos.xy, 4, texelPos);
        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            vec3 V = normalize(-viewPos);

            float wSum = 0.0;

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);
            Material material = material_decode(gData);
            ResampleMaterial resampleMaterial = resampleMaterial_fromMaterial(material);

            float hitDistance = transient_gi_initialSampleHitDistance_fetch(texelPos).x;
            restir_InitialSampleData initialSample = restir_initalSample_restoreData(texelPos, viewZ, gData.geomNormal, gData.normal, material, hitDistance);
            vec3 hitRadiance = initialSample.hitRadiance;
            vec3 sampleDirView = initialSample.directionAndLength.xyz;
            float newPHat = evalTargetFunction(hitRadiance, gData.normal, sampleDirView, V, resampleMaterial);
            float samplePdf = initialSample.pdf;
            float newWi = newPHat * safeRcp(samplePdf);

            vec4 finalSample = vec4(0.0);
            vec3 finalHitNormal = vec3(0.0);

            if (samplePdf > 0.0) {
                temporalReservoir.Y = vec4(sampleDirView, hitDistance);
                temporalReservoir.m = 1.0;
                wSum = newWi;

                finalSample = vec4(hitRadiance, newPHat);

                vec3 hitViewPos = viewPos + sampleDirView * hitDistance;
                vec3 hitScreenPos = coords_viewToScreen(hitViewPos, global_camProj);
                ivec2 hitTexelPos = ivec2(hitScreenPos.xy * uval_mainImageSize);

                vec4 hitGeomNormalData = transient_geomViewNormal_fetch(hitTexelPos);
                vec3 hitGeomNormal = normalize(hitGeomNormalData.xyz * 2.0 - 1.0);
                float geomNormalDot = dot(hitGeomNormal, gData.geomNormal);

                if (geomNormalDot > 0.99) {
                    transient_gi_initialSampleHitDistance_store(texelPos, vec4(-1.0));
                }

                vec4 hitNormalData = transient_viewNormal_fetch(hitTexelPos);
                finalHitNormal = normalize(hitNormalData.xyz * 2.0 - 1.0);
            }

            uvec4 reprojInfoData = transient_gi_diffuse_reprojInfo_fetch(texelPos);
            ReprojectInfo reprojInfo = reprojectInfo_unpack(reprojInfoData);
            float ageResetRand = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 1u), RANDOM_FRAME);
            if (reprojInfo.historyResetFactor > ageResetRand) {
                vec2 curr2PrevTexelPos = reprojInfo.curr2PrevScreenPos * uval_mainImageSize;
                curr2PrevTexelPos = clamp(curr2PrevTexelPos, vec2(0.5), uval_mainImageSize - 0.5);
                vec2 gatherTexelPos = floor(curr2PrevTexelPos - 0.5) + 1.0;
                vec2 pixelPosFract = fract(curr2PrevTexelPos - 0.5);
                vec2 bilinearWeights2 = pixelPosFract;
                vec4 bilinearWeights4;
                bilinearWeights4.yz = bilinearWeights2.xx;
                bilinearWeights4.xw = 1.0 - bilinearWeights2.xx;
                bilinearWeights4.xy *= bilinearWeights2.yy;
                bilinearWeights4.zw *= 1.0 - bilinearWeights2.yy;

                ivec2 iGatherTexelPos = ivec2(gatherTexelPos);
                bool oddFrame = bool(frameCounter & 1);

                // 4-tap bilinear temporal gather
                // Layout (gather order matches bilinearWeights xyzw):
                //   x = top-left    iGatherTexelPos + (-1,  0)
                //   y = top-right   iGatherTexelPos + ( 0,  0)
                //   z = bottom-right iGatherTexelPos + ( 0, -1)
                //   w = bottom-left  iGatherTexelPos + (-1, -1)
                {
                    float combinedWeight = bilinearWeights4.x * reprojInfo.bilateralWeights.x * reprojInfo.historyResetFactor;
                    sampleTemporalNeighbor(texelPos, iGatherTexelPos + ivec2(-1, 0), combinedWeight, 3331u, viewPos, V, gData.normal, resampleMaterial, oddFrame, temporalReservoir, wSum, finalSample, finalHitNormal);
                }
                {
                    float combinedWeight = bilinearWeights4.y * reprojInfo.bilateralWeights.y * reprojInfo.historyResetFactor;
                    sampleTemporalNeighbor(texelPos, iGatherTexelPos, combinedWeight, 3332u, viewPos, V, gData.normal, resampleMaterial, oddFrame, temporalReservoir, wSum, finalSample, finalHitNormal);
                }
                {
                    float combinedWeight = bilinearWeights4.z * reprojInfo.bilateralWeights.z * reprojInfo.historyResetFactor;
                    sampleTemporalNeighbor(texelPos, iGatherTexelPos + ivec2(0, -1), combinedWeight, 3333u, viewPos, V, gData.normal, resampleMaterial, oddFrame, temporalReservoir, wSum, finalSample, finalHitNormal);
                }
                {
                    float combinedWeight = bilinearWeights4.w * reprojInfo.bilateralWeights.w * reprojInfo.historyResetFactor;
                    sampleTemporalNeighbor(texelPos, iGatherTexelPos + ivec2(-1, -1), combinedWeight, 3334u, viewPos, V, gData.normal, resampleMaterial, oddFrame, temporalReservoir, wSum, finalSample, finalHitNormal);
                }
            }

            if (restir_isReservoirValid(temporalReservoir) && finalSample.w > 0.0 && wSum > 0.0) {
                float avgWSum = wSum * safeRcp(temporalReservoir.m);
                temporalReservoir.avgWY = avgWSum * safeRcp(finalSample.w);
                temporalReservoir.m = clamp(temporalReservoir.m, 0.0, float(SETTING_GI_TEMPORAL_REUSE_LIMIT));
            } else {
                temporalReservoir = restir_initReservoir();
                temporalReservoir.Y.w = -1.0;
                finalSample = vec4(0.0);
                finalHitNormal = vec3(0.0);
            }

            SpatialSampleData spatialSample = spatialSampleData_init();
            spatialSample.sampleValue = finalSample;
            spatialSample.geomNormal = gData.geomNormal;
            spatialSample.normal = gData.normal;
            spatialSample.hitNormal = finalHitNormal;
            transient_restir_spatialInput_store(texelPos, spatialSampleData_pack(spatialSample));

            #if USE_REFERENCE || !defined(SETTING_GI_SPATIAL_REUSE)
            vec4 ssgiDiffOut = vec4(0.0);
            vec4 ssgiSpecOut = vec4(0.0);
            bool outputValid = restir_isReservoirValid(temporalReservoir);
            #if USE_REFERENCE
            outputValid = initialValid;
            #endif
            if (outputValid) {
                #if USE_REFERENCE
                vec3 winL = sampleDirView;
                float winHitDist = hitDistance;
                vec3 winR = hitRadiance * safeRcp(samplePdf);
                #else
                vec3 winL = temporalReservoir.Y.xyz;
                float winHitDist = temporalReservoir.Y.w;
                vec3 winR = finalSample.rgb * temporalReservoir.avgWY;
                #endif
                vec3 H_win = normalize(winL + V);

                float winNDotL = saturate(dot(gData.normal, winL));
                float winNDotV = saturate(dot(gData.normal, V));
                float winNDotH = saturate(dot(gData.normal, H_win));
                float winLDotH = abs(dot(winL, H_win));

                ResampleBRDF winBRDF = resampleMaterial_evalBRDF(resampleMaterial, winNDotL, winNDotV, winNDotH, winLDotH);
                float diffRatio = winBRDF.diffuse * safeRcp(winBRDF.full);

                vec3 totalOutput = winR * winBRDF.full;
                ssgiDiffOut = vec4(totalOutput * diffRatio, winHitDist);

                ssgiSpecOut = vec4(totalOutput * (1.0 - diffRatio), winHitDist);
                vec3 specAlbedo = resampleMaterial_specularAlbedo(resampleMaterial, winNDotV);
                ssgiSpecOut.rgb *= safeRcp(specAlbedo);

                ssgiDiffOut = clamp(ssgiDiffOut, 0.0, FP16_MAX);
                ssgiSpecOut = clamp(ssgiSpecOut, 0.0, FP16_MAX);
            }

            transient_ssgiDiffOut_store(texelPos, ssgiDiffOut);
            transient_ssgiSpecOut_store(texelPos, ssgiSpecOut);
            #endif
        }
        PairwiseMISMetadata meta = pairwiseMISMetadata_init(texelPos);
        if (!restir_isReservoirValid(temporalReservoir)) {
            temporalReservoir.Y.w = -1.0;
        }
        meta.accumM = temporalReservoir.m;
        transient_restir_pairwiseMISMetadata_store(texelPos, pairwiseMISMetadata_pack(meta));
        uvec4 packedReservoir = restir_reservoir_pack(temporalReservoir);
        if (bool(frameCounter & 1)) {
            history_restir_reservoirTemporal1_store(texelPos, packedReservoir);
        } else {
            history_restir_reservoirTemporal2_store(texelPos, packedReservoir);
        }
    }
}
