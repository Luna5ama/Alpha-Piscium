/*
    References:
        [LKW26] Lin, Daqi, et al. "ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing".
            Proceedings of the ACM on Computer Graphics and Interactive Techniques. 9, 1, Article 13 (2026).
            https://doi.org/10.1145/3804494

        You can find full license texts in /licenses
*/
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_vote : enable

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ReservoirSplat.glsl"
#include "/techniques/gi/PairwiseMISMetadata.glsl"

layout(local_size_x = 128) in;

layout(r32ui) uniform restrict writeonly uimage2D uimg_r32ui;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgba16f) uniform restrict writeonly image2D uimg_rgba16f;

/*const*/
#if PASS_INDEX == 0
#define REUSETEX usam_restirReuseTex0
#elif PASS_INDEX == 1
#define REUSETEX usam_restirReuseTex1
#elif PASS_INDEX == 2
#define REUSETEX usam_restirReuseTex2
#elif PASS_INDEX == 3
#define REUSETEX usam_restirReuseTex3
#elif PASS_INDEX == 4
#define REUSETEX usam_restirReuseTex4
#elif PASS_INDEX == 5
#define REUSETEX usam_restirReuseTex5
#elif PASS_INDEX == 6
#define REUSETEX usam_restirReuseTex6
#else
#define REUSETEX usam_restirReuseTex7
#endif
/*const*/

bool restir_updateReservoirM(inout float reservoirM, inout float wSum, float wi, float m, float rand, out bool selected) {
    selected = false;
    float nextWSum = wSum + wi;
    float nextM = reservoirM + m;
    if (!(
        reservoirM >= 0.0
        && m >= 0.0
        && restir_isFinite(nextM)
        && wSum >= 0.0
        && wi > 0.0
        && restir_isFinite(nextWSum)
    )) {
        return false;
    }
    wSum = nextWSum;
    reservoirM = nextM;
    selected = rand < wi / nextWSum;
    return true;
}

float evaluateShiftTargetPHat(
    uint sourceGeometryValid,
    vec3 hitViewPosSRC,
    float sourceJacobianScale,
    vec3 geomNormalDST,
    vec3 normalDST,
    vec3 hitNormalSRC,
    float sourceRadianceMagnitude,
    float sourceTargetPHat,
    vec3 viewPosDST,
    ResampleMaterial materialDST,
    out float reconnectionJacobian,
    out float targetSpecularBRDF
) {
    const float EPSILON = 1e-6;
    float targetPHat = 0.0;
    reconnectionJacobian = 0.0;
    targetSpecularBRDF = 0.0;

    if (bool(sourceGeometryValid)) {
        vec3 diffSRCtoDST = hitViewPosSRC - viewPosDST;
        float dist2 = dot(diffSRCtoDST, diffSRCtoDST);
        if (dist2 > EPSILON) {
            vec3 dirSRCtoDST = diffSRCtoDST * inversesqrt(dist2);
            float cosPhiDST = -dot(dirSRCtoDST, hitNormalSRC);
            float targetGeomCos = dot(geomNormalDST, dirSRCtoDST);
            if (
                cosPhiDST > RESTIR_RECONNECTION_MIN_COSINE
                && targetGeomCos > RESTIR_RECONNECTION_MIN_COSINE
            ) {
                vec3 VDST = normalize(-viewPosDST);
                ResampleBRDF brdf = evalTargetBRDF(
                    geomNormalDST,
                    normalDST,
                    dirSRCtoDST,
                    VDST,
                    materialDST
                );
                targetSpecularBRDF = brdf.specular;
                float pHat = sourceRadianceMagnitude * abs(brdf.full);
                if (pHat > 0.0 && restir_isFinite(pHat)) {
                    // Equivalent to abs(log2(jacobian)) <= RESTIR_RECONNECTION_MAX_LOG2_JACOBIAN.
                    float jacobian = sourceJacobianScale * cosPhiDST / dist2;
                    if (
                        jacobian >= exp2(-RESTIR_RECONNECTION_MAX_LOG2_JACOBIAN)
                        && jacobian <= exp2(RESTIR_RECONNECTION_MAX_LOG2_JACOBIAN)
                    ) {
                        float mappedPHat = pHat * jacobian;
                        if (restir_reconnectionDensityRatioValid(sourceTargetPHat, mappedPHat)) {
                            targetPHat = mappedPHat;
                            reconnectionJacobian = jacobian;
                        }
                    }
                }
            }
        }
    }

    return targetPHat;
}

void accumulateResample(
inout PairwiseMISMetadata metaDST,
inout float selectedReconnectionJacobian,
inout bool selectedTransactionUpdated,
ivec2 texelDST, ivec2 texelSRC,
float canonMDST, float canonMSRC, float canonAvgWYSRC,
float dstPHat,
float sampleValueWSRC,
float srcToDstTargetPHat,
float dstToSrcTargetPHat,
float srcToDstReconnectionJacobian,
uint randSeed
) {
    float rcMDivK_DST = canonMDST * (1.0 / float(SETTING_GI_SPATIAL_REUSE_COUNT));
    float mcIncrement_DST = 1.0;
    if (dstToSrcTargetPHat > 0.0) {
        float sourceTerm = canonMSRC * dstToSrcTargetPHat;
        float centerTerm = rcMDivK_DST * dstPHat;
        float candidateIncrement = centerTerm / (sourceTerm + centerTerm);
        if (restir_isFinite(candidateIncrement)) {
            mcIncrement_DST = candidateIncrement;
        }
    }
    float nextMc = metaDST.mc + mcIncrement_DST;
    if (restir_isFinite(nextMc)) {
        metaDST.mc = nextMc;
    }
    metaDST.numValidNeighbors += 1u;

    if (srcToDstTargetPHat > 0.0) {
        /*const*/
        float neighborRand = restir_updateRand(texelDST, randSeed + PASS_BASE_SAMPLE_INDEX);
        /*const*/

        float MiPiRiY = canonMSRC * sampleValueWSRC;
        float weightedTargetPHat = srcToDstTargetPHat
            * (MiPiRiY / (MiPiRiY + rcMDivK_DST * srcToDstTargetPHat));
        if (!restir_isFinite(weightedTargetPHat) || weightedTargetPHat <= 0.0) {
            return;
        }

        float neighborWi = max(canonAvgWYSRC, 0.0) * weightedTargetPHat;
        bool selected;
        bool accepted = restir_updateReservoirM(
            metaDST.accumM,
            metaDST.spatialWSum,
            neighborWi,
            canonMSRC,
            neighborRand,
            selected
        );
        if (!accepted) {
            return;
        }

        if (selected) {
            metaDST.selectedTexelDelta = texelSRC - texelDST;
            selectedReconnectionJacobian = srcToDstReconnectionJacobian;
            selectedTransactionUpdated = true;
        }
    }
}

ivec2 restir_reuseUnwrapLocal(ivec2 localAnchor, ivec2 localPos) {
    ivec2 localD = localPos - localAnchor;
    localD = ((localD + RESTIR_REUSE_TILE_SIZE_HALF) & RESTIR_REUSE_TILE_MASK) - RESTIR_REUSE_TILE_SIZE_HALF;
    return localAnchor + localD;
}

void processGroupCandidate(
    uint shuffleMask,
    uint randSeed,
    inout PairwiseMISMetadata metaMe,
    inout float selectedReconnectionJacobian,
    inout bool selectedTransactionUpdated,
    ivec2 texelMe,
    vec3 geomNormalMe,
    vec3 normalMe,
    vec3 hitNormalMe,
    uint sourceGeometryValidMe,
    vec3 hitViewPosMe,
    float sourceJacobianScaleMe,
    float sourceRadianceMagnitudeMe,
    vec4 sampleValueMe,
    vec4 canonYMe,
    float canonMMe,
    float canonAvgWYMe,
    vec3 primaryViewPosMe,
    ResampleMaterial materialMe,
    uint reusableMe,
    inout vec4 ratioMe
) {
    ivec2 texelOther = subgroupShuffleXor(texelMe, shuffleMask);
    uint reusableOther = subgroupShuffleXor(reusableMe, shuffleMask);
    float sampleValueWOther = subgroupShuffleXor(sampleValueMe.w, shuffleMask);
    float canonMOther = subgroupShuffleXor(canonMMe, shuffleMask);
    float canonAvgWYOther = subgroupShuffleXor(canonAvgWYMe, shuffleMask);

    uint pairValid = reusableMe & reusableOther & uint(texelMe != texelOther);
    bool pairReusable = false;
    float meToOtherTargetPHat = 0.0;
    vec4 meToOtherRatio = vec4(0.0);
    float meToOtherReconnectionJacobian = 0.0;
    if (bool(pairValid)) {
        vec3 primaryViewPosOther = subgroupShuffleXor(primaryViewPosMe, shuffleMask);
        vec3 geomNormalOther = subgroupShuffleXor(geomNormalMe, shuffleMask);
        if (dot(geomNormalMe, geomNormalOther) > 0.99) {
            vec3 viewPosDelta = primaryViewPosMe - primaryViewPosOther;
            float planeDistance = max(abs(dot(viewPosDelta, geomNormalOther)), abs(dot(viewPosDelta, geomNormalMe)));
            float viewZMin = min(abs(primaryViewPosMe.z), abs(primaryViewPosOther.z));
            pairReusable = planeDistance < viewZMin * 0.01;
        }

        if (pairReusable) {
            vec3 normalOther = subgroupShuffleXor(normalMe, shuffleMask);
            ResampleMaterial materialOther;
            materialOther.f0 = subgroupShuffleXor(materialMe.f0, shuffleMask);
            materialOther.dielectric = subgroupShuffleXor(materialMe.dielectric, shuffleMask);
            materialOther.roughness = subgroupShuffleXor(materialMe.roughness, shuffleMask);

            float otherSpecularTowardMe;
            meToOtherTargetPHat = evaluateShiftTargetPHat(
                sourceGeometryValidMe,
                hitViewPosMe,
                sourceJacobianScaleMe,
                geomNormalOther,
                normalOther,
                hitNormalMe,
                sourceRadianceMagnitudeMe,
                sampleValueMe.w,
                primaryViewPosOther,
                materialOther,
                meToOtherReconnectionJacobian,
                otherSpecularTowardMe
            );

            // BRDF-ratio resolve: my sample's own-frame specular contribution under the other material,
            // weighted by how well it transfers to the other pixel (never boosted).
            float ownFrameSpecular = evalTargetBRDF(
                geomNormalMe,
                normalMe,
                canonYMe.xyz,
                normalize(-primaryViewPosMe),
                materialOther
            ).specular;
            if (otherSpecularTowardMe > 0.0 && ownFrameSpecular > 0.0) {
                float transferWeight = min(otherSpecularTowardMe / ownFrameSpecular, 1.0);
                // Capping by my own target BRDF bounds pairs whose materials differ.
                float sampleTargetBRDF = sampleValueMe.w / sourceRadianceMagnitudeMe;
                vec3 contribution = sampleValueMe.rgb * canonAvgWYMe * min(ownFrameSpecular, sampleTargetBRDF);
                meToOtherRatio = vec4(contribution * transferWeight, transferWeight);
            }
        }
    }
    float otherToMeTargetPHat = subgroupShuffleXor(meToOtherTargetPHat, shuffleMask);
    ratioMe += subgroupShuffleXor(meToOtherRatio, shuffleMask);
    float otherToMeReconnectionJacobian = subgroupShuffleXor(meToOtherReconnectionJacobian, shuffleMask);
    accumulateResample(
        metaMe,
        selectedReconnectionJacobian,
        selectedTransactionUpdated,
        texelMe,
        texelOther,
        canonMMe,
        canonMOther,
        canonAvgWYOther,
        sampleValueMe.w,
        sampleValueWOther,
        otherToMeTargetPHat,
        abs(meToOtherTargetPHat),
        otherToMeReconnectionJacobian,
        randSeed
    );
}

void main() {
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint tileLocalIdx = workGroupIdx & 511u;
    uint tileIdx = workGroupIdx >> 9u;
    uint tileRow = tileLocalIdx & 127u;
    uint subtileIdx = tileLocalIdx >> 7u;
    uint tileCountX = gl_NumWorkGroups.x >> 1u;
    uvec2 tileID = uvec2(tileIdx % tileCountX, tileIdx / tileCountX);
    uvec2 subtileOffset = uvec2(subtileIdx & 1u, subtileIdx >> 1u) << 7u;
    uvec2 globalPos = (tileID << 8u) + subtileOffset + uvec2(gl_LocalInvocationID.x, tileRow);

    uint groupLane = globalPos.x & 7u;
    ivec2 localFetchPos = ivec2(globalPos & uvec2(RESTIR_REUSE_TILE_MASK));
    localFetchPos.x = (localFetchPos.x >> 3) << 3;

    ivec2 tileId = ivec2(globalPos >> RESTIR_REUSE_TILE_BITS);
    ivec2 tileOrigin = tileId * RESTIR_REUSE_TILE_SIZE;
    /*const*/
    uvec2 localAnchorData = texelFetch(REUSETEX, localFetchPos, 0).xy;
    ivec2 localAnchor = ivec2(int(localAnchorData.x), int(localAnchorData.y));
    localFetchPos.x += int(groupLane);
    uvec2 localMeData = texelFetch(REUSETEX, localFetchPos, 0).xy;
    ivec2 localMe = restir_reuseUnwrapLocal(localAnchor, ivec2(int(localMeData.x), int(localMeData.y)));
    /*const*/
    localMe += uval_restirSpatialTileOffset;

    ivec2 texelMe = tileOrigin + localMe;
    uint validMe = uint(all(lessThan(ivec4(texelMe, ivec2(-1)), ivec4(uval_mainImageSizeI, texelMe))));

    float viewZMe = -65536.0;
    vec3 primaryViewPosMe = vec3(0.0);
    vec3 geomNormalMe = vec3(0.0);
    vec3 normalMe = vec3(0.0);
    vec3 hitNormalMe = vec3(0.0);
    vec4 sampleValueMe = vec4(0.0);
    vec4 canonYMe = vec4(0.0, 0.0, 0.0, -1.0);
    float canonMMe = 0.0;
    float canonAvgWYMe = 0.0;
    ResampleMaterial materialMe = resampleMaterial_init();
    PairwiseMISMetadata metaMe = pairwiseMISMetadata_init();
    // Specular BRDF-ratio resolve: xyz = sum(weight * contribution), w = sum(weight).
    vec4 ratioMe = vec4(0.0);
    float selectedReconnectionJacobian = 0.0;
    bool selectedTransactionUpdated = false;

    if (bool(validMe)) {
        viewZMe = texelFetch(usam_gbufferSolidViewZ, texelMe, 0).x;
        validMe &= uint(viewZMe > -65536.0);
        if (bool(validMe)) {
            uint packedGBufferData2 = texelFetch(usam_gbufferSolidData2, texelMe, 0).r;
            validMe &= uint(bitfieldExtract(packedGBufferData2, 24, 1) == 0u);
        }
    }
    if (subgroupAll(validMe == 0u)) {
        return;
    }
    if (bool(validMe)) {
        uint packedPrimaryMe = restir_splatFetchCurrentPrimary(texelMe);
        uvec4 spatialSamplePackedDataMe = transient_restir_spatialInput_fetch(texelMe);
        if (packedPrimaryMe != 0u) {
            primaryViewPosMe = restir_splatUnpackPrimary(
                texelMe,
                packedPrimaryMe,
                global_camProjInverse
            );
        } else {
            vec2 screenPos = coords_texelToUV(texelMe, uval_mainImageSizeRcp)
                - uval_taaJitterUV;
            primaryViewPosMe = coords_toViewCoord(
                screenPos,
                viewZMe,
                global_camProjInverse
            );
        }
        geomNormalMe = nzpacking_unpackNormalOct32(spatialSamplePackedDataMe.x);
        hitNormalMe = nzpacking_unpackNormalOct32(spatialSamplePackedDataMe.y);
        normalMe = normalize(transient_viewNormal_fetch(texelMe).xyz * 2.0 - 1.0);
        sampleValueMe = unpackHalf4x16(spatialSamplePackedDataMe.zw);
        materialMe = resampleMaterial_unpack(transient_restir_resampleMaterial_fetch(texelMe));
        #if PASS_INDEX != 0
        metaMe = pairwiseMISMetadata_unpack(transient_restir_pairwiseMISMetadata_fetch(texelMe));
        vec4 packedRatio = transient_ssgiSpecOut_fetch(texelMe);
        ratioMe = vec4(packedRatio.xyz * packedRatio.w, packedRatio.w);
        #endif

        uvec4 repMe = transient_restir_reservoirTemporal_fetch(texelMe);
        ReSTIRReservoir reservoirMe = restir_reservoir_unpack(repMe);
        canonYMe = reservoirMe.Y;
        canonMMe = reservoirMe.m;
        canonAvgWYMe = reservoirMe.avgWY;
        #if PASS_INDEX == 0
        // Replace the temporal splat-next scratch with pairwise state.
        metaMe.accumM = canonMMe;
        if (canonYMe.w > 0.0) {
            float centerSpecular = evalTargetBRDF(
                geomNormalMe,
                normalMe,
                canonYMe.xyz,
                normalize(-primaryViewPosMe),
                materialMe
            ).specular;
            ratioMe = vec4(sampleValueMe.rgb * canonAvgWYMe * centerSpecular, 1.0);
        }
        #endif
    }

    uint reusableMe = validMe;
    float sourceRadianceMagnitudeMe = length(sampleValueMe.rgb);
    float cosMe = dot(normalMe, canonYMe.xyz);
    float cosPhiMe = -dot(canonYMe.xyz, hitNormalMe);
    uint sourceGeometryValidMe = uint(
        canonYMe.w > 1e-6
        && cosMe > 0.0
        && cosPhiMe > RESTIR_RECONNECTION_MIN_COSINE
        && dot(geomNormalMe, canonYMe.xyz) > RESTIR_RECONNECTION_MIN_COSINE
    );
    vec3 hitViewPosMe = vec3(0.0);
    // Source half of the reconnection Jacobian: squared hit distance over hit cosine.
    float sourceJacobianScaleMe = 0.0;
    if (bool(sourceGeometryValidMe)) {
        hitViewPosMe = primaryViewPosMe + canonYMe.xyz * canonYMe.w;
        sourceJacobianScaleMe = canonYMe.w * canonYMe.w / cosPhiMe;
    }

    processGroupCandidate(1u, 3337u, metaMe, selectedReconnectionJacobian, selectedTransactionUpdated, texelMe, geomNormalMe, normalMe, hitNormalMe, sourceGeometryValidMe, hitViewPosMe, sourceJacobianScaleMe, sourceRadianceMagnitudeMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, primaryViewPosMe, materialMe, reusableMe, ratioMe);
    processGroupCandidate(2u, 3338u, metaMe, selectedReconnectionJacobian, selectedTransactionUpdated, texelMe, geomNormalMe, normalMe, hitNormalMe, sourceGeometryValidMe, hitViewPosMe, sourceJacobianScaleMe, sourceRadianceMagnitudeMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, primaryViewPosMe, materialMe, reusableMe, ratioMe);
    processGroupCandidate(3u, 3339u, metaMe, selectedReconnectionJacobian, selectedTransactionUpdated, texelMe, geomNormalMe, normalMe, hitNormalMe, sourceGeometryValidMe, hitViewPosMe, sourceJacobianScaleMe, sourceRadianceMagnitudeMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, primaryViewPosMe, materialMe, reusableMe, ratioMe);
    processGroupCandidate(4u, 3340u, metaMe, selectedReconnectionJacobian, selectedTransactionUpdated, texelMe, geomNormalMe, normalMe, hitNormalMe, sourceGeometryValidMe, hitViewPosMe, sourceJacobianScaleMe, sourceRadianceMagnitudeMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, primaryViewPosMe, materialMe, reusableMe, ratioMe);
    processGroupCandidate(5u, 3341u, metaMe, selectedReconnectionJacobian, selectedTransactionUpdated, texelMe, geomNormalMe, normalMe, hitNormalMe, sourceGeometryValidMe, hitViewPosMe, sourceJacobianScaleMe, sourceRadianceMagnitudeMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, primaryViewPosMe, materialMe, reusableMe, ratioMe);
    processGroupCandidate(6u, 3342u, metaMe, selectedReconnectionJacobian, selectedTransactionUpdated, texelMe, geomNormalMe, normalMe, hitNormalMe, sourceGeometryValidMe, hitViewPosMe, sourceJacobianScaleMe, sourceRadianceMagnitudeMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, primaryViewPosMe, materialMe, reusableMe, ratioMe);
    processGroupCandidate(7u, 3343u, metaMe, selectedReconnectionJacobian, selectedTransactionUpdated, texelMe, geomNormalMe, normalMe, hitNormalMe, sourceGeometryValidMe, hitViewPosMe, sourceJacobianScaleMe, sourceRadianceMagnitudeMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, primaryViewPosMe, materialMe, reusableMe, ratioMe);

    if (bool(validMe)) {
        transient_restir_pairwiseMISMetadata_store(texelMe, pairwiseMISMetadata_pack(metaMe));
        if (selectedTransactionUpdated) {
            transient_restir_initialCandidateNormal_store(texelMe, uvec4(floatBitsToUint(selectedReconnectionJacobian), 0u, 0u, 0u));
        }
        // ssgiSpecOut is free until GIReSTIRPairedSpatialShade reads this weighted mean and weight sum.
        vec3 ratioMean = ratioMe.w > 0.0 ? ratioMe.xyz / ratioMe.w : vec3(0.0);
        transient_ssgiSpecOut_store(texelMe, vec4(min(ratioMean, FP16_MAX), ratioMe.w));
    }
}
