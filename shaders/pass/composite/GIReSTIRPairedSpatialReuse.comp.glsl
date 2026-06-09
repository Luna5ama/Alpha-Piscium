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

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/PairwiseMISMetadata.glsl"

layout(local_size_x = 128) in;

layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;

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

bool restir_updateReservoirM(inout float reservoirM, inout float wSum, float wi, float m, float rand) {
    wSum += wi;
    reservoirM += m;
    return rand < wi / wSum;
}

float evaluateTargetFunctionWithNdotL(vec3 irradiance, vec3 normal, float rawNdotL, vec3 lightDir, vec3 viewDir, ResampleMaterial material) {
    float rawNdotV = dot(normal, viewDir);
    float LdotV = dot(lightDir, viewDir);
    float invLen = inversesqrt(max(2.0 + 2.0 * LdotV, 1e-5));
    float NdotV = saturate(rawNdotV);
    float NdotH = saturate((rawNdotL + rawNdotV) * invLen);
    float LdotH = (1.0 + LdotV) * invLen;
    ResampleBRDF brdf = resampleMaterial_evalBRDF(material, rawNdotL, NdotV, NdotH, LdotH);
    return length(irradiance * brdf.full);
}

float evaluateShiftTargetPHat(
    ivec2 texelDST,
    vec4 canonYSRC,
    vec3 normalDST,
    vec3 normalSRC,
    vec3 hitNormalSRC,
    vec3 sampleValueSRC,
    vec3 viewPosDST, vec3 viewPosSRC
) {
    const float EPSILON = 1e-6;
    float targetPHat = 0.0;

    if (canonYSRC.w > EPSILON) {
        vec3 hitViewPosSRC = viewPosSRC + canonYSRC.xyz * canonYSRC.w;
        vec3 diffSRCtoDST = hitViewPosSRC - viewPosDST;
        float dist2 = dot(diffSRCtoDST, diffSRCtoDST);
        if (dist2 > EPSILON) {
            vec3 dirSRCtoDST = diffSRCtoDST * inversesqrt(dist2);
            float cosPhiSRC = -dot(canonYSRC.xyz, hitNormalSRC);
            float cosPhiDST = -dot(dirSRCtoDST, hitNormalSRC);
            if (cosPhiSRC > 0.0 && cosPhiDST > 0.0) {
                float rawNdotLDST = dot(normalDST, dirSRCtoDST);
                if (rawNdotLDST > 0.0) {
                    vec4 resampleMaterialDataDST = transient_restir_resampleMaterial_fetch(texelDST);
                    float cosSRC = dot(normalSRC, canonYSRC.xyz);
                    vec3 VDST = normalize(-viewPosDST);
                    ResampleMaterial matDST = resampleMaterial_unpack(resampleMaterialDataDST);
                    float pHat = evaluateTargetFunctionWithNdotL(sampleValueSRC, normalDST, rawNdotLDST, dirSRCtoDST, VDST, matDST);
                    if (pHat > 0.0) {
                        float jacobian_DST = clamp(((canonYSRC.w * canonYSRC.w) * cosPhiDST) / (dist2 * cosPhiSRC), 0.0, 256.0);
                        targetPHat = pHat * jacobian_DST;
                        if (cosSRC <= 0.0) {
                            targetPHat = -targetPHat;
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
ivec2 texelDST, ivec2 texelSRC,
float canonMDST, float canonMSRC, float canonAvgWYSRC,
float dstPHat,
float sampleValueWSRC,
float srcToDstTargetPHat,
float dstToSrcTargetPHat,
uint randSeed
) {
    if (srcToDstTargetPHat > 0.0) {
        /*const*/
        float neighborRand = restir_updateRand(texelDST, randSeed + PASS_BASE_SAMPLE_INDEX);
        /*const*/

        float rcMDivK_DST = canonMDST * (1.0 / float(SETTING_GI_SPATIAL_REUSE_COUNT));
        float MiPiRiY = canonMSRC * sampleValueWSRC;
        float mi_DST = MiPiRiY * safeRcp(MiPiRiY + rcMDivK_DST * srcToDstTargetPHat);

        float mcIncrement_DST = 1.0;
        if (dstToSrcTargetPHat > 0.0) {
            float MiPiRcY = canonMSRC * dstToSrcTargetPHat;
            mcIncrement_DST = 1.0 - MiPiRcY * safeRcp(MiPiRcY + rcMDivK_DST * dstPHat);
        }

        metaDST.mc += mcIncrement_DST;
        metaDST.numValidNeighbors += 1u;

        float neighborWi = srcToDstTargetPHat * max(canonAvgWYSRC, 0.0) * mi_DST;
        if (restir_updateReservoirM(metaDST.accumM, metaDST.spatialWSum, neighborWi, canonMSRC, neighborRand)) {
            metaDST.selectedTexelDelta = texelSRC - texelDST;
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
    ivec2 texelMe,
    vec3 geomNormalMe,
    vec3 normalMe,
    vec3 hitNormalMe,
    vec4 sampleValueMe,
    vec4 canonYMe,
    float canonMMe,
    float canonAvgWYMe,
    vec3 viewPosMe,
    uint reusableMe
) {
    ivec2 texelOther = subgroupShuffleXor(texelMe, shuffleMask);
    uint reusableOther = subgroupShuffleXor(reusableMe, shuffleMask);
    vec3 viewPosOther = subgroupShuffleXor(viewPosMe, shuffleMask);
    vec3 geomNormalOther = subgroupShuffleXor(geomNormalMe, shuffleMask);
    vec3 normalOther = subgroupShuffleXor(normalMe, shuffleMask);
    float sampleValueWOther = subgroupShuffleXor(sampleValueMe.w, shuffleMask);
    float canonMOther = subgroupShuffleXor(canonMMe, shuffleMask);
    float canonAvgWYOther = subgroupShuffleXor(canonAvgWYMe, shuffleMask);

    uint pairValid = reusableMe & reusableOther & uint(texelMe != texelOther);
    bool pairReusable = false;
    if (bool(pairValid)) {
        if (dot(geomNormalMe, geomNormalOther) > 0.99) {
            vec3 viewPosDelta = viewPosMe - viewPosOther;
            float planeDistance = max(abs(dot(viewPosDelta, geomNormalOther)), abs(dot(viewPosDelta, geomNormalMe)));
            float viewZMin = min(abs(viewPosMe.z), abs(viewPosOther.z));
            pairReusable = planeDistance < viewZMin * 0.01;
        }
    }

    float meToOtherTargetPHat = 0.0;
    if (pairReusable) {
        meToOtherTargetPHat = evaluateShiftTargetPHat(
            texelOther,
            canonYMe,
            normalOther,
            normalMe,
            hitNormalMe,
            sampleValueMe.xyz,
            viewPosOther,
            viewPosMe
        );
    }
    float otherToMeTargetPHat = subgroupShuffleXor(meToOtherTargetPHat, shuffleMask);
    if (pairReusable) {
        accumulateResample(
            metaMe,
            texelMe,
            texelOther,
            canonMMe,
            canonMOther,
            canonAvgWYOther,
            sampleValueMe.w,
            sampleValueWOther,
            otherToMeTargetPHat,
            abs(meToOtherTargetPHat),
            randSeed
        );
    }
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
    vec3 viewPosMe = vec3(0.0);
    vec3 geomNormalMe = vec3(0.0);
    vec3 normalMe = vec3(0.0);
    vec3 hitNormalMe = vec3(0.0);
    vec4 sampleValueMe = vec4(0.0);
    vec4 canonYMe = vec4(0.0, 0.0, 0.0, -1.0);
    float canonMMe = 0.0;
    float canonAvgWYMe = 0.0;
    PairwiseMISMetadata metaMe = pairwiseMISMetadata_init();

    if (bool(validMe)) {
        viewZMe = texelFetch(usam_gbufferSolidViewZ, texelMe, 0).x;
        uvec4 spatialSamplePackedDataMe = transient_restir_spatialInput_fetch(texelMe);
        vec2 screenPosMe = coords_texelToUV(texelMe, uval_mainImageSizeRcp);
        viewPosMe = coords_toViewCoord(screenPosMe, viewZMe, global_camProjInverse);
        nzpacking_unpackNormalOct16(spatialSamplePackedDataMe.x, geomNormalMe, hitNormalMe);
        normalMe = nzpacking_unpackNormalOct32(spatialSamplePackedDataMe.y);
        sampleValueMe = unpackHalf4x16(spatialSamplePackedDataMe.zw);
        metaMe = pairwiseMISMetadata_unpack(transient_restir_pairwiseMISMetadata_fetch(texelMe));

        if (viewZMe > -65536.0) {
            uvec4 repMe;
            if (bool(frameCounter & 1)) {
                repMe = history_restir_reservoirTemporal1_fetch(texelMe);
            } else {
                repMe = history_restir_reservoirTemporal2_fetch(texelMe);
            }
            canonYMe.xyz = nzpacking_unpackNormalOct32(repMe.x);
            canonMMe = uintBitsToFloat(repMe.y);
            canonAvgWYMe = uintBitsToFloat(repMe.z);
            canonYMe.w = uintBitsToFloat(repMe.w);
        }
    }

    uint reusableMe = validMe & uint(viewZMe > -65536.0);

    processGroupCandidate(1u, 3337u, metaMe, texelMe, geomNormalMe, normalMe, hitNormalMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, viewPosMe, reusableMe);
    processGroupCandidate(2u, 3338u, metaMe, texelMe, geomNormalMe, normalMe, hitNormalMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, viewPosMe, reusableMe);
    processGroupCandidate(3u, 3339u, metaMe, texelMe, geomNormalMe, normalMe, hitNormalMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, viewPosMe, reusableMe);
    processGroupCandidate(4u, 3340u, metaMe, texelMe, geomNormalMe, normalMe, hitNormalMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, viewPosMe, reusableMe);
    processGroupCandidate(5u, 3341u, metaMe, texelMe, geomNormalMe, normalMe, hitNormalMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, viewPosMe, reusableMe);
    processGroupCandidate(6u, 3342u, metaMe, texelMe, geomNormalMe, normalMe, hitNormalMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, viewPosMe, reusableMe);
    processGroupCandidate(7u, 3343u, metaMe, texelMe, geomNormalMe, normalMe, hitNormalMe, sampleValueMe, canonYMe, canonMMe, canonAvgWYMe, viewPosMe, reusableMe);

    if (bool(validMe)) {
        transient_restir_pairwiseMISMetadata_store(texelMe, pairwiseMISMetadata_pack(metaMe));
    }
}
