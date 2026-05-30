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
#extension GL_KHR_shader_subgroup_clustered : enable

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/PairwiseMIS.glsl"

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



ShiftMapping evaluateShiftMapping(
ivec2 texelDST,
ReSTIRReservoir canonResSRC,
vec3 nomrlaDST, SpatialSampleData sampleSRC,
vec3 viewPosDST, vec3 viewPosSRC
) {
    const float EPSILON = 1e-6;
    ShiftMapping mapping = shiftMapping_init();

    if (canonResSRC.Y.w > EPSILON) {
        vec3 hitViewPosSRC = viewPosSRC + canonResSRC.Y.xyz * canonResSRC.Y.w;
        vec3 diffSRCtoDST = hitViewPosSRC - viewPosDST;
        float dist2 = dot(diffSRCtoDST, diffSRCtoDST);
        if (dist2 > EPSILON) {
            vec3 dirSRCtoDST = diffSRCtoDST * inversesqrt(dist2);
            float cosPhiSRC = -dot(canonResSRC.Y.xyz, sampleSRC.hitNormal);
            float cosPhiDST = -dot(dirSRCtoDST, sampleSRC.hitNormal);
            if (cosPhiSRC > 0.0 && cosPhiDST > 0.0) {
                vec4 resampleMaterialDataDST = transient_restir_resampleMaterial_fetch(texelDST);
                vec3 VDST = normalize(-viewPosDST);
                ResampleMaterial matDST = resampleMaterial_unpack(resampleMaterialDataDST);
                float pHat = evalTargetFunction(sampleSRC.sampleValue.xyz, nomrlaDST, dirSRCtoDST, VDST, matDST);
                if (pHat > 0.0) {
                    float jacobian_DST = clamp(((canonResSRC.Y.w * canonResSRC.Y.w) * cosPhiDST) / (dist2 * cosPhiSRC), 0.0, 256.0);
                    mapping.Y = vec4(dirSRCtoDST, sqrt(dist2));
                    mapping.targetPHat = pHat * jacobian_DST;
                    float cosSRC = dot(sampleSRC.normal, canonResSRC.Y.xyz);
                    if (cosSRC <= 0.0) {
                        mapping.targetPHat = -mapping.targetPHat;
                    }
                }
            }
        }
    }

    return mapping;
}

void doResample(
ivec2 texelDST, ivec2 texelSRC,
float canonMDST, float canonMSRC, float canonAvgWYSRC,
float dstPHat,
SpatialSampleData sampleSRC,
ShiftMapping srcToDst,
float dstToSrcTargetPHat
) {
    if (shiftMapping_isReusable(srcToDst)) {
        uvec4 pairwiseMISMetadataDST = transient_restir_pairwiseMISMetadata_fetch(texelDST);

        float rcMDivK_DST = canonMDST / SETTING_GI_SPATIAL_REUSE_COUNT;
        float MiPiRiY = canonMSRC * sampleSRC.sampleValue.w;
        float mi_DST = MiPiRiY * safeRcp(MiPiRiY + rcMDivK_DST * abs(srcToDst.targetPHat));

        float mcIncrement_DST = 1.0;
        if (dstToSrcTargetPHat > 0.0) {
            float MiPiRcY = canonMSRC * dstToSrcTargetPHat;
            mcIncrement_DST = 1.0 - MiPiRcY * safeRcp(MiPiRcY + rcMDivK_DST * dstPHat);
        }

        PairwiseMISMetadata metaDST = pairwiseMISMetadata_unpack(pairwiseMISMetadataDST);
        metaDST.mc += mcIncrement_DST;
        metaDST.numValidNeighbors += 1u;

        float neighborWi = abs(srcToDst.targetPHat) * max(canonAvgWYSRC, 0.0) * mi_DST;
        float neighborRand = restir_updateRand(texelDST, 3337u + PASS_INDEX);
        if (restir_updateReservoirM(metaDST.accumM, metaDST.spatialWSum, neighborWi, canonMSRC, neighborRand)) {
            metaDST.selectedTexel = texelSRC;
        }
        transient_restir_pairwiseMISMetadata_store(texelDST, pairwiseMISMetadata_pack(metaDST));
    }
}

void main() {
    ivec2 localFetchPos = ivec2(gl_GlobalInvocationID.xy) & RESTIR_REUSE_TILE_MASK;
    localFetchPos.x = localFetchPos.x >> 1;

    ivec2 tileId = ivec2(gl_GlobalInvocationID.xy) >> RESTIR_REUSE_TILE_BITS;
    ivec2 tileOrigin = tileId * RESTIR_REUSE_TILE_SIZE;
    uvec4 pairData = texelFetch(REUSETEX, localFetchPos, 0);
    ivec2 localA = ivec2(pairData.xy);
    ivec2 localB = ivec2(pairData.zw);
    ivec2 localD = localB - localA;
    localD = ((localD + RESTIR_REUSE_TILE_SIZE_HALF) & RESTIR_REUSE_TILE_MASK) - RESTIR_REUSE_TILE_SIZE_HALF;
    localB = localA + localD;
    localA = (localA + uval_restirSpatialTileOffset);
    localB = (localB + uval_restirSpatialTileOffset);
    ivec2 texelA = tileOrigin + localA;
    ivec2 texelB = tileOrigin + localB;
    uint validA = uint(all(lessThan(ivec4(texelA, ivec2(-1)), ivec4(uval_mainImageSizeI, texelA))));
    uint validB = uint(all(lessThan(ivec4(texelB, ivec2(-1)), ivec4(uval_mainImageSizeI, texelB))));

    if (bool(validA & validB & uint(texelA != texelB))) {
        bool flagA = bool(gl_GlobalInvocationID.x & 1u);
        ivec2 texelMe = flagA ? texelA : texelB;
        float viewZMe = texelFetch(usam_gbufferSolidViewZ, texelMe, 0).x;
        vec2 screenPosMe = coords_texelToUV(texelMe, uval_mainImageSizeRcp);
        vec3 viewPosMe = coords_toViewCoord(screenPosMe, viewZMe, global_camProjInverse);
        uint checkMe = uint(viewZMe > -65536.0);
        uint viewZCheck = subgroupClusteredAnd(checkMe, 2);
        if (bool(viewZCheck)) {
            uvec4 spatialSamplePackedDataMe = transient_restir_spatialInput_fetch(texelMe);
            SpatialSampleData sampleMe = spatialSampleData_unpack(spatialSamplePackedDataMe);
            uvec2 packedDataOther = subgroupShuffleXor(spatialSamplePackedDataMe.xy, 1);
            vec4 xyzw = unpackSnorm4x8(packedDataOther.x);
            vec3 geomNormalOther = coords_octDecode11(xyzw.xy);
            vec3 viewPosOther = subgroupShuffleXor(viewPosMe, 1);
            float planeDistance = gi_planeDistance(viewPosMe, sampleMe.geomNormal, viewPosOther, geomNormalOther);
            float viewZMin = min(abs(viewPosMe.z), abs(viewPosOther.z));

            if (dot(sampleMe.geomNormal, geomNormalOther) > 0.99 && planeDistance < viewZMin * 0.01) {
                uvec4 repMe;
                if (bool(frameCounter & 1)) {
                    repMe = history_restir_reservoirTemporal1_fetch(texelMe);
                } else {
                    repMe = history_restir_reservoirTemporal2_fetch(texelMe);
                }
                vec3 normalOther = nzpacking_unpackNormalOct32(packedDataOther.y);
                ivec2 texelOther = subgroupShuffleXor(texelMe, 1);
                ReSTIRReservoir canonResMe = restir_reservoir_unpack(repMe);
                ShiftMapping shiftMeToOther = evaluateShiftMapping(texelOther, canonResMe, normalOther, sampleMe, viewPosOther, viewPosMe);

                float otherToMePHat = subgroupShuffleXor(abs(shiftMeToOther.targetPHat), 1);
                float dstPHat = subgroupShuffleXor(sampleMe.sampleValue.w, 1);
                float canonMOther = subgroupShuffleXor(canonResMe.m, 1);
                doResample(texelOther, texelMe, canonMOther, canonResMe.m, canonResMe.avgWY, dstPHat, sampleMe, shiftMeToOther, otherToMePHat);
            }
        }
    }
}
