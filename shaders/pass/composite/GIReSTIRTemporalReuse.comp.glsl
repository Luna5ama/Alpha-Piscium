/*
    References:
        [LLK25] Liu, Jeffrey, et al. "Reservoir Splatting for Temporal Path Resampling and Motion Blur".
            SIGGRAPH Conference Papers 2025. https://doi.org/10.1145/3721238.3730646
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

layout(local_size_x = 16, local_size_y = 16) in;

#include "/Base.glsl"

layout(rgba16f) uniform writeonly image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_temp2;
layout(rgba16f) uniform writeonly image2D uimg_temp3;
layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(r32f) uniform restrict writeonly image2D uimg_r32f;
layout(r32ui) uniform restrict uimage2D uimg_r32ui;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgba8) uniform restrict writeonly image2D uimg_rgba8;

#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ReservoirSplat.glsl"
#include "/techniques/gi/InitialSample.glsl"
#include "/techniques/gi/ReprojectInfo.glsl"
#include "/util/Rand.glsl"
#include "/util/Sampling.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/ThreadGroupTiling.glsl"
#include "/util/BSDF.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

#ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
const float RESTIR_BACKUP_PRIOR_PROPOSAL_MIX = 0.5;
#endif

float temporalScaledRatio(float numerator, float scale, float denominator) {
    if (!restir_isPositiveFinite(numerator) || !restir_isPositiveFinite(scale) || !restir_isPositiveFinite(denominator)) {
        return 0.0;
    }

    float commonScale = max(numerator, denominator);
    float ratio = (numerator / commonScale)
        * scale
        / (denominator / commonScale);
    return restir_isPositiveFinite(ratio) ? ratio : 0.0;
}

float temporalProposalMulDiv(float factorA, float factorB, float divisor) {
    if (!restir_isPositiveFinite(factorA) || !restir_isPositiveFinite(factorB) || !restir_isPositiveFinite(divisor)) {
        return 0.0;
    }

    float result = (factorA / divisor) * factorB;
    if (restir_isPositiveFinite(result)) {
        return result;
    }
    result = (factorB / divisor) * factorA;
    if (restir_isPositiveFinite(result)) {
        return result;
    }

    float logResult = log2(factorA) + log2(factorB) - log2(divisor);
    result = exp2(logResult);
    return restir_isPositiveFinite(result) ? result : 0.0;
}


shared mat3 shared_prevViewToCurrView;
shared vec3 shared_prevViewToCurrViewTrans;

ReSTIRReservoir readPreviousReservoir(ivec2 texelPos) {
    uvec4 packedReservoir = history_restir_reservoirTemporal_load(texelPos);
    return restir_reservoir_unpack(packedReservoir);
}

uint readPreviousPrimary(ivec2 texelPos, bool oddFrame) {
    return oddFrame
        ? history_restir_primary2_load(texelPos).x
        : history_restir_primary1_load(texelPos).x;
}

uint readSplatNext(ivec2 texelPos) {
    return transient_restir_pairwiseMISMetadata_load(texelPos).x;
}

uint readCurrentPrimary(ivec2 texelPos, bool oddFrame) {
    return oddFrame
        ? history_restir_primary1_load(texelPos).x
        : history_restir_primary2_load(texelPos).x;
}

void writeCurrentPrimary(ivec2 texelPos, uint packedPrimary, bool oddFrame) {
    if (oddFrame) {
        history_restir_primary1_store(texelPos, uvec4(packedPrimary));
    } else {
        history_restir_primary2_store(texelPos, uvec4(packedPrimary));
    }
}

struct TemporalHistorySample {
    ivec2 texelPos;
    uint packedPrimary;
    ReSTIRReservoir reservoir;
    vec4 sampleValue;
    vec3 previousPrimary;
    vec3 currentPrimary;
    vec3 previousGeomNormal;
    vec3 currentGeomNormal;
    vec3 currentHit;
    vec3 currentHitNormal;
    #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
    ResampleMaterial previousMaterial;
    #endif
};

struct TemporalPathShift {
    vec4 Y;
    vec3 hitNormal;
    float jacobian;
};

bool temporalReconnectionJacobian(
    float sourceDistance2,
    float targetDistance2,
    float sourceCosine,
    float targetCosine,
    out float jacobian
) {
    jacobian = 0.0;
    if (
        !restir_isPositiveFinite(sourceDistance2)
        || !restir_isPositiveFinite(targetDistance2)
        || sourceCosine <= RESTIR_RECONNECTION_MIN_COSINE
        || targetCosine <= RESTIR_RECONNECTION_MIN_COSINE
    ) {
        return false;
    }
    float log2Jacobian = log2(sourceDistance2)
        + log2(targetCosine)
        - log2(targetDistance2)
        - log2(sourceCosine);
    if (!restir_isFinite(log2Jacobian) || abs(log2Jacobian) > RESTIR_RECONNECTION_MAX_LOG2_JACOBIAN) {
        return false;
    }
    jacobian = exp2(log2Jacobian);
    return restir_isPositiveFinite(jacobian);
}

TemporalHistorySample temporalHistorySample_init() {
    TemporalHistorySample source;
    source.texelPos = ivec2(0);
    source.packedPrimary = 0u;
    source.reservoir = restir_initReservoir();
    source.sampleValue = vec4(0.0);
    source.previousPrimary = vec3(0.0);
    source.currentPrimary = vec3(0.0);
    source.previousGeomNormal = vec3(0.0);
    source.currentGeomNormal = vec3(0.0);
    source.currentHit = vec3(0.0);
    source.currentHitNormal = vec3(0.0);
    #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
    source.previousMaterial = resampleMaterial_init();
    #endif
    return source;
}

bool loadTemporalHistorySample(ivec2 texelPos, bool oddFrame, inout TemporalHistorySample source) {
    source.texelPos = texelPos;
    source.reservoir = readPreviousReservoir(texelPos);
    source.packedPrimary = readPreviousPrimary(texelPos, oddFrame);
    if (
        !restir_isReservoirValid(source.reservoir)
        || !restir_isFinite(source.reservoir.Y.w)
        || source.packedPrimary == 0u
    ) {
        return false;
    }

    source.sampleValue = history_restir_prevSample_fetch(texelPos);
    if (!restir_isPositiveFinite(source.sampleValue.w) || !restir_isFinite(source.sampleValue.rgb)) {
        return false;
    }
    #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
    source.previousMaterial = resampleMaterial_unpack(history_restir_prevResampleMaterial_fetch(texelPos));
    #endif

    source.previousPrimary = restir_splatUnpackPrimary(
        texelPos,
        source.packedPrimary,
        global_prevCamProjInverse
    );
    source.currentPrimary = shared_prevViewToCurrView * source.previousPrimary + shared_prevViewToCurrViewTrans;
    vec3 previousGeomNormal = history_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0;
    if (
        !restir_isFinite(source.previousPrimary)
        || !restir_isFinite(source.currentPrimary)
        || !restir_isFinite(previousGeomNormal)
        || dot(previousGeomNormal, previousGeomNormal) <= 1e-4
    ) {
        return false;
    }
    source.previousGeomNormal = normalize(previousGeomNormal);
    source.currentGeomNormal = shared_prevViewToCurrView * source.previousGeomNormal;
    if (!restir_isFinite(source.currentGeomNormal)) {
        return false;
    }
    source.currentHit = vec3(0.0);
    source.currentHitNormal = vec3(0.0);
    if (source.reservoir.Y.w > 0.0) {
        vec3 previousHit = source.previousPrimary + source.reservoir.Y.xyz * source.reservoir.Y.w;
        source.currentHit = shared_prevViewToCurrView * previousHit + shared_prevViewToCurrViewTrans;
        vec3 previousHitNormal = nzpacking_unpackNormalOct32(
            history_restir_prevHitNormal_fetch(texelPos).x
        );
        if (!restir_isFinite(source.currentHit) || !restir_isFinite(previousHitNormal) || dot(previousHitNormal, previousHitNormal) <= 1e-4) {
            return false;
        }
        source.currentHitNormal = shared_prevViewToCurrView * previousHitNormal;
        if (!restir_isFinite(source.currentHitNormal)) {
            return false;
        }
    }
    return true;
}

bool shiftTemporalPathPrimary(
    TemporalHistorySample source,
    out vec4 shiftedY
) {
    shiftedY = source.reservoir.Y;
    if (source.reservoir.Y.w <= 0.0) {
        vec3 currentDirection = shared_prevViewToCurrView * source.reservoir.Y.xyz;
        if (!restir_isFinite(currentDirection) || dot(currentDirection, currentDirection) <= 1e-8) {
            return false;
        }
        shiftedY.xyz = normalize(currentDirection);
        return true;
    }

    vec3 sourceOffset = source.currentHit - source.currentPrimary;
    float sourceDistance2 = dot(sourceOffset, sourceOffset);
    if (!restir_isPositiveFinite(sourceDistance2)) {
        return false;
    }

    vec3 sourceDirection = sourceOffset * inversesqrt(sourceDistance2);
    shiftedY = vec4(sourceDirection, sqrt(sourceDistance2));
    return restir_isFinite(shiftedY.xyz);
}

bool shiftTemporalPathReconnect(
    TemporalHistorySample source,
    vec3 targetPrimary,
    out TemporalPathShift shifted
) {
    shifted.Y = source.reservoir.Y;
    shifted.hitNormal = vec3(0.0);
    shifted.jacobian = 1.0;
    if (source.reservoir.Y.w <= 0.0) {
        vec3 currentDirection = shared_prevViewToCurrView * source.reservoir.Y.xyz;
        if (!restir_isFinite(currentDirection) || dot(currentDirection, currentDirection) <= 1e-8) {
            return false;
        }
        shifted.Y.xyz = normalize(currentDirection);
        return true;
    }
    vec3 sourceOffset = source.currentHit - source.currentPrimary;
    vec3 targetOffset = source.currentHit - targetPrimary;
    float sourceDistance2 = dot(sourceOffset, sourceOffset);
    float targetDistance2 = dot(targetOffset, targetOffset);
    if (!restir_isPositiveFinite(sourceDistance2) || !restir_isPositiveFinite(targetDistance2)) {
        return false;
    }

    vec3 targetDirection = targetOffset * inversesqrt(targetDistance2);
    vec3 sourceDirection = sourceOffset * inversesqrt(sourceDistance2);
    float sourceCosine = -dot(sourceDirection, source.currentHitNormal);
    float targetCosine = -dot(targetDirection, source.currentHitNormal);
    if (!temporalReconnectionJacobian(
        sourceDistance2,
        targetDistance2,
        sourceCosine,
        targetCosine,
        shifted.jacobian
    )) {
        return false;
    }
    shifted.Y = vec4(targetDirection, sqrt(targetDistance2));
    shifted.hitNormal = source.currentHitNormal;
    return restir_isPositiveFinite(shifted.jacobian)
        && restir_isFinite(shifted.Y.xyz);
}

#ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
bool temporalRasterSubpixelInBounds(vec2 rasterSubpixel) {
    return all(greaterThanEqual(rasterSubpixel, vec2(0.0)))
        && all(lessThan(rasterSubpixel, vec2(1.0)));
}

float temporalReprojectionSurfaceWeight(ReprojectInfo reprojInfo, ivec2 texelPos) {
    vec2 previousTexelPos = clamp(
        reprojInfo.curr2PrevScreenPos * uval_mainImageSize,
        vec2(1.0),
        uval_mainImageSize - 1.0
    );
    ivec2 gatherTexelPos = ivec2(floor(previousTexelPos - 0.5) + 1.0);
    ivec2 gatherOffset = texelPos - gatherTexelPos;
    if (gatherOffset == ivec2(-1, 0)) {
        return reprojInfo.bilateralWeights.x;
    }
    if (gatherOffset == ivec2(0, 0)) {
        return reprojInfo.bilateralWeights.y;
    }
    if (gatherOffset == ivec2(0, -1)) {
        return reprojInfo.bilateralWeights.z;
    }
    if (gatherOffset == ivec2(-1, -1)) {
        return reprojInfo.bilateralWeights.w;
    }
    return 0.0;
}

bool reconstructPrimaryOnPlane(
    ivec2 texelPos,
    vec2 rasterSubpixel,
    float viewZ,
    vec3 planePrimary,
    vec3 planeGeomNormal,
    vec2 jitterUV,
    mat4 projectionInverse,
    out vec3 reconstructedPrimary
) {
    if (!restir_isFinite(planePrimary) || !restir_isFinite(planeGeomNormal)) {
        return false;
    }
    vec2 screenPos = (vec2(texelPos) + rasterSubpixel) * uval_mainImageSizeRcp - jitterUV;
    vec3 rayPoint = coords_toViewCoord(screenPos, viewZ, projectionInverse);
    float planeDenominator = dot(planeGeomNormal, rayPoint);
    if (!restir_isFinite(rayPoint) || !restir_isFinite(planeDenominator) || abs(planeDenominator) <= 1e-8) {
        return false;
    }

    float planeScale = dot(planeGeomNormal, planePrimary) / planeDenominator;
    if (!restir_isPositiveFinite(planeScale)) {
        return false;
    }

    reconstructedPrimary = rayPoint * planeScale;
    return restir_isFinite(reconstructedPrimary);
}

bool reconstructBackupTargetPrimary(
    ivec2 texelPos,
    uint sourcePackedPrimary,
    vec2 backupSubpixelDelta,
    float centerViewZ,
    vec3 centerPrimary,
    vec3 centerGeomNormal,
    out vec3 targetPrimary
) {
    vec2 sourceRasterSubpixel = restir_splatUnpackPrimaryOffset(sourcePackedPrimary) + uval_prevTaaJitter;
    vec2 targetRasterSubpixel = sourceRasterSubpixel + backupSubpixelDelta;
    if (!temporalRasterSubpixelInBounds(sourceRasterSubpixel)
        || !temporalRasterSubpixelInBounds(targetRasterSubpixel)) {
        return false;
    }
    return reconstructPrimaryOnPlane(
        texelPos,
        targetRasterSubpixel,
        centerViewZ,
        centerPrimary,
        centerGeomNormal,
        uval_taaJitterUV,
        global_camProjInverse,
        targetPrimary
    );
}

bool reconstructBackupProposalSourcePrimary(
    ivec2 outputTexelPos,
    vec3 candidateCurrentPrimary,
    TemporalHistorySample backupSource,
    vec2 backupSubpixelDelta,
    out vec3 candidatePreviousPrimary,
    out vec3 candidateSourceCurrentPrimary
) {
    vec2 candidateScreenPixel = coords_viewToScreen(candidateCurrentPrimary, global_camProj).xy
        * uval_mainImageSize;
    vec2 targetRasterSubpixel = candidateScreenPixel + uval_taaJitter - vec2(outputTexelPos);
    vec2 sourceRasterSubpixel = targetRasterSubpixel - backupSubpixelDelta;
    if (!temporalRasterSubpixelInBounds(targetRasterSubpixel)
        || !temporalRasterSubpixelInBounds(sourceRasterSubpixel)) {
        return false;
    }
    if (!reconstructPrimaryOnPlane(
        backupSource.texelPos,
        sourceRasterSubpixel,
        backupSource.previousPrimary.z,
        backupSource.previousPrimary,
        backupSource.previousGeomNormal,
        uval_prevTaaJitterUV,
        global_prevCamProjInverse,
        candidatePreviousPrimary
    )) {
        return false;
    }

    candidateSourceCurrentPrimary = shared_prevViewToCurrView * candidatePreviousPrimary
        + shared_prevViewToCurrViewTrans;
    return restir_isFinite(candidateSourceCurrentPrimary);
}

float evaluateBackupProposalTerm(
    ivec2 outputTexelPos,
    TemporalHistorySample backupSource,
    vec2 backupSubpixelDelta,
    vec3 currentPrimary,
    vec4 candidateY,
    vec3 currentHitNormal,
    vec3 candidateRadiance,
    float candidateTargetPHat
) {
    vec3 candidatePreviousPrimary;
    vec3 candidateSourceCurrentPrimary;
    if (!reconstructBackupProposalSourcePrimary(
        outputTexelPos,
        currentPrimary,
        backupSource,
        backupSubpixelDelta,
        candidatePreviousPrimary,
        candidateSourceCurrentPrimary
    )) {
        return 0.0;
    }

    vec3 previousDirection;
    float forwardJacobian = 1.0;
    if (candidateY.w > 0.0) {
        vec3 currentHit = currentPrimary + candidateY.xyz * candidateY.w;
        vec3 sourceOffset = currentHit - candidateSourceCurrentPrimary;
        float sourceDistance2 = dot(sourceOffset, sourceOffset);
        float targetDistance2 = candidateY.w * candidateY.w;
        if (!restir_isPositiveFinite(sourceDistance2) || !restir_isPositiveFinite(targetDistance2)) {
            return 0.0;
        }

        vec3 sourceDirection = sourceOffset * inversesqrt(sourceDistance2);
        float sourceCosine = -dot(sourceDirection, currentHitNormal);
        float targetCosine = -dot(candidateY.xyz, currentHitNormal);
        if (!restir_isPositiveFinite(sourceCosine) || !restir_isPositiveFinite(targetCosine)) {
            return 0.0;
        }
        if (!temporalReconnectionJacobian(
            sourceDistance2,
            targetDistance2,
            sourceCosine,
            targetCosine,
            forwardJacobian
        )) {
            return 0.0;
        }

        vec3 previousHit = coord_viewCurrToPrev(vec4(currentHit, 1.0), false).xyz;
        previousDirection = normalize(previousHit - candidatePreviousPrimary);
    } else {
        previousDirection = normalize(coords_dir_worldToViewPrev(coords_dir_viewToWorld(candidateY.xyz)));
    }
    if (!restir_isFinite(previousDirection)) {
        return 0.0;
    }

    vec3 previousNormal = history_viewNormal_fetch(backupSource.texelPos).xyz * 2.0 - 1.0;
    if (!restir_isFinite(previousNormal) || dot(previousNormal, previousNormal) <= 1e-4) {
        return 0.0;
    }
    previousNormal = normalize(previousNormal);
    float previousPHat = restir_stabilizeTemporalTargetPHat(
        evalTargetFunction(
            candidateRadiance,
            backupSource.previousGeomNormal,
            previousNormal,
            previousDirection,
            normalize(-candidatePreviousPrimary),
            backupSource.previousMaterial
        )
    );
    if (!restir_reconnectionDensityRatioValid(previousPHat, candidateTargetPHat * forwardJacobian)) {
        return 0.0;
    }
    float confidence = clamp(backupSource.reservoir.m, 0.0, float(SETTING_GI_TEMPORAL_REUSE_LIMIT));
    return temporalProposalMulDiv(
        RESTIR_BACKUP_PRIOR_PROPOSAL_MIX * confidence,
        previousPHat,
        forwardJacobian
    );
}
#endif

float readPreviousConfidence(ivec2 texelPos, bool oddFrame) {
    if (any(lessThan(texelPos, ivec2(0))) || any(greaterThanEqual(texelPos, uval_mainImageSizeI))) {
        return 0.0;
    }
    if (readPreviousPrimary(texelPos, oddFrame) == 0u) {
        return 0.0;
    }
    ReSTIRReservoir previousReservoir = readPreviousReservoir(texelPos);
    if (!restir_isReservoirValid(previousReservoir)) {
        return 0.0;
    }
    float confidence = previousReservoir.m;
    return restir_isFinite(confidence) ? max(confidence, 0.0) : 0.0;
}

void sampleTemporalSplat(
    ivec2 texelPos,
    ivec2 sourceTexelPos,
    uint sourceNode,
    vec3 centerGeomNormal,
    vec3 centerNormal,
    ResampleMaterial material,
    float canonicalConfidence,
    bool oddFrame,
    #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
    bool backupProposalValid,
    TemporalHistorySample backupSource,
    vec2 backupSubpixelDelta,
    #endif
    inout ReSTIRReservoir reservoir,
    inout float wSum,
    inout vec4 finalSample,
    inout vec3 finalHitNormal,
    inout vec3 finalPrimaryViewPos
) {
    TemporalHistorySample source = temporalHistorySample_init();
    if (!loadTemporalHistorySample(sourceTexelPos, oddFrame, source)) {
        return;
    }

    float primaryJacobian = restir_splatPrimaryJacobian(
        source.previousPrimary,
        source.previousGeomNormal,
        source.currentPrimary,
        source.currentGeomNormal
    );
    if (!restir_isPositiveFinite(primaryJacobian)) {
        return;
    }

    vec4 shiftedY;
    if (!shiftTemporalPathPrimary(source, shiftedY)) {
        return;
    }

    vec3 viewDirection = normalize(-source.currentPrimary);
    float currentPHat = restir_stabilizeTemporalTargetPHat(
        evalTargetFunction(
            source.sampleValue.rgb,
            centerGeomNormal,
            centerNormal,
            shiftedY.xyz,
            viewDirection,
            material
        )
    );
    float previousPHat = max(source.sampleValue.w, 0.0);
    float previousConfidence = clamp(source.reservoir.m, 0.0, float(SETTING_GI_TEMPORAL_REUSE_LIMIT));
    if (!restir_isPositiveFinite(currentPHat) || !restir_isPositiveFinite(previousPHat) || !restir_isPositiveFinite(previousConfidence)) {
        return;
    }
    if (!restir_reconnectionDensityRatioValid(previousPHat, currentPHat * primaryJacobian)) {
        return;
    }
    float splatConfidence = previousConfidence;
    #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
    splatConfidence *= RESTIR_BACKUP_PRIOR_PROPOSAL_MIX;
    #endif
    float proposalDenominator = canonicalConfidence * currentPHat
        + temporalProposalMulDiv(splatConfidence, previousPHat, primaryJacobian);
    #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
    if (backupProposalValid) {
        float backupTerm = evaluateBackupProposalTerm(
            texelPos,
            backupSource,
            backupSubpixelDelta,
            source.currentPrimary,
            shiftedY,
            source.currentHitNormal,
            source.sampleValue.rgb,
            currentPHat
        );
        proposalDenominator += backupTerm;
    }
    #endif
    float sourceMass = max(source.reservoir.avgWY, 0.0) * previousPHat;
    float candidateWeight = sourceMass
        * temporalScaledRatio(currentPHat, splatConfidence, proposalDenominator);
    if (!restir_isPositiveFinite(sourceMass) || !restir_isPositiveFinite(candidateWeight) || !restir_isFinite(wSum)) {
        return;
    }

    float candidateRand = restir_updateRand(texelPos, sourceNode ^ 0x9e3779b9u);
    if (restir_updateReservoir(reservoir, wSum, shiftedY, candidateWeight, 0.0, candidateRand)) {
        finalSample = vec4(source.sampleValue.rgb, currentPHat);
        finalHitNormal = source.currentHitNormal;
        finalPrimaryViewPos = source.currentPrimary;
    }
}

float evaluateSplatProposalTerm(
    vec3 currentPrimary,
    vec3 geomNormal,
    vec3 sampleDirection,
    float hitDistance,
    vec3 hitRadiance,
    float currentPHat,
    bool oddFrame,
    bool historyReusable
) {
    if (!historyReusable) {
        return 0.0;
    }

    vec3 prevPrimary = coord_viewCurrToPrev(vec4(currentPrimary, 1.0), false).xyz;
    vec4 prevClip = global_prevCamProj * vec4(prevPrimary, 1.0);
    if (prevClip.z <= 0.0 || any(greaterThanEqual(abs(prevClip.xy), prevClip.ww))) {
        return 0.0;
    }

    vec2 prevScreen = prevClip.xy / prevClip.w * 0.5 + 0.5 + uval_prevTaaJitterUV;
    ivec2 prevTexelPos = ivec2(floor(prevScreen * uval_mainImageSize));
    if (any(lessThan(prevTexelPos, ivec2(0))) || any(greaterThanEqual(prevTexelPos, uval_mainImageSizeI))) {
        return 0.0;
    }

    float prevViewZ = history_viewZ_fetch(prevTexelPos).x;
    if (prevViewZ <= -65536.0) {
        return 0.0;
    }

    vec2 prevCenterScreen = coords_texelToUV(prevTexelPos, uval_mainImageSizeRcp) - uval_prevTaaJitterUV;
    vec3 prevCenter = coords_toViewCoord(prevCenterScreen, prevViewZ, global_prevCamProjInverse);
    vec3 prevGeomNormal = normalize(coords_dir_worldToViewPrev(coords_dir_viewToWorld(geomNormal)));
    vec3 prevCenterGeomNormal = normalize(history_geomViewNormal_fetch(prevTexelPos).xyz * 2.0 - 1.0);
    if (restir_splatSurfaceConfidence(prevPrimary, prevGeomNormal, prevCenter, prevCenterGeomNormal) <= 0.0) {
        return 0.0;
    }

    ReSTIRReservoir reverseReservoir = readPreviousReservoir(prevTexelPos);
    if (
        !restir_isReservoirValid(reverseReservoir)
        || readPreviousPrimary(prevTexelPos, oddFrame) == 0u
    ) {
        return 0.0;
    }
    float reverseConfidence = clamp(reverseReservoir.m, 0.0, float(SETTING_GI_TEMPORAL_REUSE_LIMIT));
    if (reverseConfidence <= 0.0) {
        return 0.0;
    }

    vec3 prevSampleDirection;
    if (hitDistance > 0.0) {
        vec3 currHit = currentPrimary + sampleDirection * hitDistance;
        vec3 prevHit = coord_viewCurrToPrev(vec4(currHit, 1.0), false).xyz;
        prevSampleDirection = normalize(prevHit - prevPrimary);
    } else {
        prevSampleDirection = normalize(coords_dir_worldToViewPrev(coords_dir_viewToWorld(sampleDirection)));
    }

    vec3 prevNormal = history_viewNormal_fetch(prevTexelPos).xyz * 2.0 - 1.0;
    if (!restir_isFinite(prevNormal) || dot(prevNormal, prevNormal) <= 1e-4) {
        return 0.0;
    }
    prevNormal = normalize(prevNormal);
    ResampleMaterial previousMaterial = resampleMaterial_unpack(history_restir_prevResampleMaterial_fetch(prevTexelPos));
    float previousPHat = restir_stabilizeTemporalTargetPHat(
        evalTargetFunction(
            hitRadiance,
            prevCenterGeomNormal,
            prevNormal,
            prevSampleDirection,
            normalize(-prevPrimary),
            previousMaterial
        )
    );
    float forwardJacobian = restir_splatPrimaryJacobian(prevPrimary, prevGeomNormal, currentPrimary, geomNormal);
    if (!restir_isPositiveFinite(previousPHat) || !restir_isPositiveFinite(forwardJacobian)) {
        return 0.0;
    }
    if (!restir_reconnectionDensityRatioValid(previousPHat, currentPHat * forwardJacobian)) {
        return 0.0;
    }

    #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
    reverseConfidence *= RESTIR_BACKUP_PRIOR_PROPOSAL_MIX;
    #endif
    return temporalProposalMulDiv(reverseConfidence, previousPHat, forwardJacobian);
}

float evaluateTemporalConfidence(vec2 previousScreenPos, float canonicalConfidence, bool oddFrame, bool historyReusable) {
    if (!historyReusable) {
        return canonicalConfidence;
    }

    vec2 curr2PrevTexelPos = clamp(
        previousScreenPos * uval_mainImageSize,
        vec2(1.0),
        uval_mainImageSize - 1.0
    );
    vec2 prevBase = curr2PrevTexelPos - 0.5;
    ivec2 gatherTexelPos = ivec2(floor(prevBase) + 1.0);
    vec2 f = fract(prevBase);
    vec4 bilinearWeights = vec4(
        (1.0 - f.x) * f.y,
        f.x * f.y,
        f.x * (1.0 - f.y),
        (1.0 - f.x) * (1.0 - f.y)
    );

    vec4 previousConfidence = vec4(
        readPreviousConfidence(gatherTexelPos + ivec2(-1, 0), oddFrame),
        readPreviousConfidence(gatherTexelPos, oddFrame),
        readPreviousConfidence(gatherTexelPos + ivec2(0, -1), oddFrame),
        readPreviousConfidence(gatherTexelPos + ivec2(-1, -1), oddFrame)
    );
    float historyConfidence = dot(bilinearWeights, previousConfidence);
    return min(canonicalConfidence + historyConfidence, float(SETTING_GI_TEMPORAL_REUSE_LIMIT));
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
        uint packedReservoirDirection = 0u;
        bool packedReservoirDirectionValid = false;
        bool oddFrame = bool(frameCounter & 1);
        uint splatHead = readCurrentPrimary(texelPos, oddFrame);
        writeCurrentPrimary(texelPos, RESTIR_SPLAT_NULL, oddFrame);
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos.xy, 4, texelPos);
        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

            vec3 V = normalize(-viewPos);

            vec3 targetGeomNormal = normalize(transient_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0);
            vec3 targetNormal = transient_viewNormal_fetch(texelPos).xyz * 2.0 - 1.0;
            restir_InitialCandidate initialCandidate = restir_initialCandidate_load(texelPos);
            float hitDistance = initialCandidate.hitDistance;
            vec3 hitRadiance = initialCandidate.radiance;
            vec3 sampleDirView = initialCandidate.rayDirView;
            float samplePdf = initialCandidate.pdf;
            ResampleMaterial storedMaterial = resampleMaterial_unpack(
                transient_restir_resampleMaterial_fetch(texelPos)
            );

            float denoiserHitDistance = hitDistance;
            if (denoiserHitDistance <= RESTIR_INITIAL_CANDIDATE_NEEDS_VOXEL) {
                denoiserHitDistance = -1.0;
            }
            transient_gi_initialSampleHitDistance_store(texelPos, vec4(denoiserHitDistance));

            vec4 finalSample = vec4(0.0);
            vec3 finalHitNormal = vec3(0.0);
            vec3 finalPrimaryViewPos = viewPos;

            float wSum = 0.0;
            bool initialValid = samplePdf > 0.0;
            float canonicalConfidence = 1.0;
            float canonicalPHat = 0.0;
            float canonicalBaseWeight = 0.0;
            if (samplePdf > 0.0) {
                canonicalPHat = restir_stabilizeTemporalTargetPHat(
                    evalTargetFunction(
                        hitRadiance,
                        targetGeomNormal,
                        targetNormal,
                        sampleDirView,
                        V,
                        storedMaterial
                    )
                );
                canonicalBaseWeight = canonicalPHat / samplePdf;
                if (!restir_isPositiveFinite(canonicalBaseWeight)) {
                    canonicalBaseWeight = 0.0;
                }
                finalSample = vec4(hitRadiance, canonicalPHat);
                finalHitNormal = initialCandidate.hitNormalView;
            }

            uvec4 reprojInfoData = transient_gi_diffuse_reprojInfo_fetch(texelPos);
            ReprojectInfo reprojInfo = reprojectInfo_unpack(reprojInfoData);
            vec2 temporalPreviousScreenPos = reprojInfo.curr2PrevScreenPos;
            float ageResetRand = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 1u), RANDOM_FRAME);
            float pSpec = 1.0;
            uint packedGBufferData2 = texelFetch(usam_gbufferSolidData2, texelPos, 0).r;
            if (storedMaterial.dielectric > 0.0) {
                float NdotV = saturate(dot(targetNormal, V));
                float fresnelV = saturate(resampleMaterial_fresnel(storedMaterial, NdotV));
                vec3 albedo = colors2_material_toWorkSpace(unpackUnorm4x8(packedGBufferData2).rgb);
                float albedoLuma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, albedo);
                pSpec = fresnelV * safeRcp(albedoLuma * (1.0 - fresnelV) + fresnelV);
                // Clamping this to avoid dead locks that causes fireflies
                pSpec = sqrt(clamp(pSpec, 0.01, 0.99));
            }
            pSpec = pow(storedMaterial.roughness, pSpec);
            transient_diffBounceProbability_store(texelPos, vec4(pSpec));

            float historyRetention = global_historyResetFactor * reprojInfo.historyResetFactor;
            bool isHand = bool(bitfieldExtract(packedGBufferData2, 24, 1));
            bool historyReusable = !isHand && historyRetention > ageResetRand;
            #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
            TemporalHistorySample backupSource = temporalHistorySample_init();
            TemporalPathShift backupShift;
            backupShift.Y = vec4(0.0, 0.0, 0.0, -1.0);
            backupShift.hitNormal = vec3(0.0);
            backupShift.jacobian = 0.0;
            vec3 backupTargetPrimary = viewPos;
            bool backupProposalValid = false;
            bool backupCandidateValid = false;
            float backupPHat = 0.0;
            float backupConfidence = 0.0;
            vec2 backupSubpixelDelta = vec2(0.0);
            if (historyReusable) {
                vec2 previousTexelPos = clamp(
                    temporalPreviousScreenPos * uval_mainImageSize,
                    vec2(1.0),
                    uval_mainImageSize - 1.0
                );
                vec2 previousBase = previousTexelPos - 0.5;
                ivec2 gatherTexelPos = ivec2(floor(previousBase) + 1.0);
                vec2 bilinearFraction = fract(previousBase);
                vec4 bilinearWeights = vec4(
                    (1.0 - bilinearFraction.x) * bilinearFraction.y,
                    bilinearFraction.x * bilinearFraction.y,
                    bilinearFraction.x * (1.0 - bilinearFraction.y),
                    (1.0 - bilinearFraction.x) * (1.0 - bilinearFraction.y)
                );

                float backupSelect = rand_r2Seq1(frameCounter);
                ivec2 backupOffset = ivec2(-1, -1);
                if (backupSelect < bilinearWeights.x) {
                    backupOffset = ivec2(-1, 0);
                } else if (backupSelect < bilinearWeights.x + bilinearWeights.y) {
                    backupOffset = ivec2(0, 0);
                } else if (backupSelect < bilinearWeights.x + bilinearWeights.y + bilinearWeights.z) {
                    backupOffset = ivec2(0, -1);
                }

                ivec2 backupTexelPos = gatherTexelPos + backupOffset;
                float backupSurfaceWeight = temporalReprojectionSurfaceWeight(reprojInfo, backupTexelPos);
                ivec2 previousBaseTexel = ivec2(floor(previousBase));
                backupSubpixelDelta = vec2(backupTexelPos - previousBaseTexel) - bilinearFraction;
                bool backupInBounds = all(greaterThanEqual(backupTexelPos, ivec2(0)))
                    && all(lessThan(backupTexelPos, uval_mainImageSizeI));
                if (
                    backupInBounds
                    && backupSurfaceWeight > 0.9
                    && loadTemporalHistorySample(backupTexelPos, oddFrame, backupSource)
                ) {
                    backupConfidence = clamp(
                        backupSource.reservoir.m,
                        0.0,
                        float(SETTING_GI_TEMPORAL_REUSE_LIMIT)
                    );
                    backupProposalValid = backupConfidence > 0.0;
                    if (
                        backupProposalValid
                        && reconstructBackupTargetPrimary(
                            texelPos,
                            backupSource.packedPrimary,
                            backupSubpixelDelta,
                            viewZ,
                            viewPos,
                            targetGeomNormal,
                            backupTargetPrimary
                        )
                        && shiftTemporalPathReconnect(backupSource, backupTargetPrimary, backupShift)
                    ) {
                        backupPHat = restir_stabilizeTemporalTargetPHat(
                            evalTargetFunction(
                                backupSource.sampleValue.rgb,
                                targetGeomNormal,
                                targetNormal,
                                backupShift.Y.xyz,
                                normalize(-backupTargetPrimary),
                                storedMaterial
                            )
                        );
                        backupCandidateValid = restir_isPositiveFinite(backupPHat)
                            && restir_reconnectionDensityRatioValid(
                                backupSource.sampleValue.w,
                                backupPHat * backupShift.jacobian
                            );
                    }
                }
            }
            #endif

            float canonicalTerm = canonicalConfidence * canonicalPHat;
            float canonicalDenominator = canonicalTerm;
            if (canonicalTerm > 0.0 && historyReusable) {
                canonicalDenominator += evaluateSplatProposalTerm(
                    viewPos,
                    targetGeomNormal,
                    sampleDirView,
                    hitDistance,
                    hitRadiance,
                    canonicalPHat,
                    oddFrame,
                    historyReusable
                );
                #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
                if (backupProposalValid) {
                    canonicalDenominator += evaluateBackupProposalTerm(
                        texelPos,
                        backupSource,
                        backupSubpixelDelta,
                        viewPos,
                        vec4(sampleDirView, hitDistance),
                        initialCandidate.hitNormalView,
                        hitRadiance,
                        canonicalPHat
                    );
                }
                #endif
            }
            float canonicalMIS = canonicalTerm > 0.0
                ? temporalScaledRatio(canonicalPHat, canonicalConfidence, canonicalDenominator)
                : 1.0;
            if (canonicalBaseWeight > 0.0) {
                temporalReservoir.Y = vec4(sampleDirView, hitDistance);
                wSum = canonicalBaseWeight * canonicalMIS;
            }

            #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
            if (backupCandidateValid) {
                float backupTerm = temporalProposalMulDiv(
                    RESTIR_BACKUP_PRIOR_PROPOSAL_MIX * backupConfidence,
                    backupSource.sampleValue.w,
                    backupShift.jacobian
                );
                float backupDenominator = canonicalConfidence * backupPHat + backupTerm;
                backupDenominator += evaluateSplatProposalTerm(
                    backupTargetPrimary,
                    targetGeomNormal,
                    backupShift.Y.xyz,
                    backupShift.Y.w,
                    backupSource.sampleValue.rgb,
                    backupPHat,
                    oddFrame,
                    historyReusable
                );
                float backupSourceMass = max(backupSource.reservoir.avgWY, 0.0)
                    * backupSource.sampleValue.w;
                float backupWeight = backupSourceMass * temporalScaledRatio(
                    backupPHat,
                    RESTIR_BACKUP_PRIOR_PROPOSAL_MIX * backupConfidence,
                    backupDenominator
                );
                if (
                    restir_isPositiveFinite(backupSourceMass)
                    && backupTerm >= 0.0
                    && restir_isFinite(backupTerm)
                    && restir_isPositiveFinite(backupDenominator)
                    && restir_isPositiveFinite(backupWeight)
                    && restir_isFinite(wSum)
                ) {
                    float backupRand = restir_updateRand(texelPos, 0x6a09e667u);
                    if (restir_updateReservoir(
                        temporalReservoir,
                        wSum,
                        backupShift.Y,
                        backupWeight,
                        0.0,
                        backupRand
                    )) {
                        finalSample = vec4(backupSource.sampleValue.rgb, backupPHat);
                        finalHitNormal = backupShift.hitNormal;
                        finalPrimaryViewPos = backupTargetPrimary;
                    }
                }
            }
            #endif

            uint splatNode = splatHead;
            uint chainLength = 0u;
            if (historyReusable) {
                while (splatNode != RESTIR_SPLAT_NULL && chainLength < RESTIR_SPLAT_MAX_CHAIN_LENGTH) {
                    ivec2 sourceTexelPos = restir_splatDecodeNode(splatNode);
                    sampleTemporalSplat(
                        texelPos,
                        sourceTexelPos,
                        splatNode,
                        targetGeomNormal,
                        targetNormal,
                        storedMaterial,
                        canonicalConfidence,
                        oddFrame,
                        #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
                        backupProposalValid,
                        backupSource,
                        backupSubpixelDelta,
                        #endif
                        temporalReservoir,
                        wSum,
                        finalSample,
                        finalHitNormal,
                        finalPrimaryViewPos
                    );
                    splatNode = readSplatNext(sourceTexelPos);
                    ++chainLength;
                }
            }

            bool chainOverflow = historyReusable && splatNode != RESTIR_SPLAT_NULL;
            float temporalConfidence = evaluateTemporalConfidence(
                temporalPreviousScreenPos,
                canonicalConfidence,
                oddFrame,
                historyReusable
            );
            #ifdef SETTING_GI_TEMPORAL_BACKUP_SAMPLE
            if (backupProposalValid) {
                temporalConfidence = min(
                    temporalConfidence + backupConfidence,
                    float(SETTING_GI_TEMPORAL_REUSE_LIMIT)
                );
            }
            #endif
            if (chainOverflow) {
                temporalReservoir = restir_initReservoir();
                finalSample = vec4(hitRadiance, canonicalPHat);
                finalHitNormal = initialCandidate.hitNormalView;
                finalPrimaryViewPos = viewPos;
                wSum = canonicalBaseWeight;
                temporalConfidence = canonicalConfidence;
                if (canonicalBaseWeight > 0.0) {
                    temporalReservoir.Y = vec4(sampleDirView, hitDistance);
                }
            }

            uint packedPrimary = 0u;
            bool storageGeometryValid = restir_isFinite(finalPrimaryViewPos)
                && restir_isFinite(temporalReservoir.Y.xyz)
                && restir_isFinite(temporalReservoir.Y.w)
                && dot(temporalReservoir.Y.xyz, temporalReservoir.Y.xyz) > 1e-8;
            bool finiteSecondaryHit = storageGeometryValid && temporalReservoir.Y.w > 0.0;
            vec3 storageHitViewPos = finiteSecondaryHit
                ? finalPrimaryViewPos + temporalReservoir.Y.xyz * temporalReservoir.Y.w
                : vec3(0.0);
            storageGeometryValid = storageGeometryValid
                && (!finiteSecondaryHit || restir_isFinite(storageHitViewPos));
            if (storageGeometryValid) {
                if (!isHand) {
                    packedPrimary = restir_splatPackPrimary(
                        texelPos,
                        finalPrimaryViewPos,
                        global_camProj
                    );
                    if (packedPrimary != 0u) {
                        finalPrimaryViewPos = restir_splatUnpackPrimary(
                            texelPos,
                            packedPrimary,
                            global_camProjInverse
                        );
                    }
                }
                if (finiteSecondaryHit) {
                    vec3 storageHitOffset = storageHitViewPos - finalPrimaryViewPos;
                    float storageHitDistance2 = dot(storageHitOffset, storageHitOffset);
                    storageGeometryValid = restir_isPositiveFinite(storageHitDistance2);
                    if (storageGeometryValid) {
                        temporalReservoir.Y.xyz = storageHitOffset * inversesqrt(storageHitDistance2);
                    }
                }
            }
            if (storageGeometryValid) {
                packedReservoirDirection = nzpacking_packNormalOct32(temporalReservoir.Y.xyz);
                temporalReservoir.Y.xyz = nzpacking_unpackNormalOct32(packedReservoirDirection);
                if (finiteSecondaryHit) {
                    float projectedHitDistance = dot(storageHitViewPos - finalPrimaryViewPos, temporalReservoir.Y.xyz);
                    storageGeometryValid = restir_isPositiveFinite(projectedHitDistance);
                    if (storageGeometryValid) {
                        temporalReservoir.Y.w = projectedHitDistance;
                    }
                }
                storageGeometryValid = restir_isFinite(finalPrimaryViewPos)
                    && restir_isFinite(temporalReservoir.Y);
                packedReservoirDirectionValid = storageGeometryValid;
            }

            float finalTargetBRDF = 0.0;
            if (storageGeometryValid && restir_isFinite(finalSample.rgb)) {
                finalTargetBRDF = evalTargetBRDF(
                    targetGeomNormal,
                    targetNormal,
                    temporalReservoir.Y.xyz,
                    normalize(-finalPrimaryViewPos),
                    storedMaterial
                );
                finalSample.w = restir_stabilizeTemporalTargetPHat(
                    length(finalSample.rgb * finalTargetBRDF)
                );
            } else {
                finalSample = vec4(0.0);
            }
            if (restir_isFinite(finalSample.rgb) && restir_isPositiveFinite(finalSample.w)) {
                float samplePeak = max(max(finalSample.x, finalSample.y), max(finalSample.z, finalSample.w));
                float sampleScale = (FP16_MAX * 0.5) * rcp(samplePeak);
                if (restir_isPositiveFinite(sampleScale)) {
                    vec4 scaledSample = vec4(finalSample.rgb * sampleScale, 0.0);
                    finalSample.rgb = unpackHalf4x16(packHalf4x16(clamp(scaledSample, 0.0, FP16_MAX))).rgb;
                    finalSample.w = restir_quantizeStoredTargetPHat(
                        length(finalSample.rgb * finalTargetBRDF)
                    );
                    if (!restir_isPositiveFinite(finalSample.w)) {
                        finalSample = vec4(0.0);
                    }
                } else {
                    finalSample = vec4(0.0);
                }
            } else {
                finalSample = vec4(0.0);
            }

            float finalAvgWY = wSum * safeRcp(finalSample.w);
            if (
                restir_isPositiveFinite(finalSample.w)
                && restir_isPositiveFinite(wSum)
                && restir_isPositiveFinite(finalAvgWY)
                && restir_isPositiveFinite(temporalConfidence)
            ) {
                temporalReservoir.avgWY = finalAvgWY;
                temporalReservoir.m = temporalConfidence;
            } else {
                temporalReservoir = restir_initReservoir();
                temporalReservoir.Y.w = -1.0;
                finalSample = vec4(0.0);
                finalHitNormal = vec3(0.0);
            }

            if (!restir_isReservoirValid(temporalReservoir) || isHand) {
                packedPrimary = 0u;
            }
            writeCurrentPrimary(texelPos, packedPrimary, oddFrame);

            SpatialSampleData spatialSample = spatialSampleData_init();
            spatialSample.sampleValue = finalSample;
            spatialSample.geomNormal = targetGeomNormal;
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
                vec3 winR = finalSample.rgb;
                #endif
                vec3 winV = V;
                #if !USE_REFERENCE
                winV = normalize(-finalPrimaryViewPos);
                #endif
                vec3 winNormal = resampleMaterial_resolveNormal(
                    targetGeomNormal,
                    targetNormal,
                    winV
                );
                if (
                    dot(targetGeomNormal, winL) > 0.0
                    && dot(targetGeomNormal, winV) > 0.0
                    && dot(winNormal, winL) > 0.0
                ) {
                    ResampleBRDF winBRDF = resampleMaterial_evalBRDF(
                        storedMaterial,
                        winNormal,
                        winL,
                        winV
                    );
                    float diffRatio = winBRDF.diffuse * safeRcp(winBRDF.full);

                    #if USE_REFERENCE
                    vec3 totalOutput = winR * winBRDF.full;
                    #else
                    vec3 totalOutput = (winR * winBRDF.full) * temporalReservoir.avgWY;
                    #endif
                    ssgiDiffOut = vec4(totalOutput * diffRatio, winHitDist);

                    ssgiSpecOut = vec4(totalOutput * (1.0 - diffRatio), winHitDist);
                    float denoiseNDotV = saturate(dot(winNormal, winV));
                    vec3 specDenoiseFactor = resampleMaterial_specularDenoiseFactor(storedMaterial, denoiseNDotV);
                    ssgiSpecOut.rgb *= rcp(specDenoiseFactor);

                    if (restir_isFinite(ssgiDiffOut.rgb) && restir_isFinite(ssgiDiffOut.w)) {
                        ssgiDiffOut = clamp(ssgiDiffOut, 0.0, FP16_MAX);
                    } else {
                        ssgiDiffOut = vec4(0.0);
                    }
                    if (restir_isFinite(ssgiSpecOut.rgb) && restir_isFinite(ssgiSpecOut.w)) {
                        ssgiSpecOut = clamp(ssgiSpecOut, 0.0, FP16_MAX);
                    } else {
                        ssgiSpecOut = vec4(0.0);
                    }
                }
            }

            transient_ssgiDiffOut_store(texelPos, ssgiDiffOut);
            transient_ssgiSpecOut_store(texelPos, ssgiSpecOut);
            #endif
        }
        bool finalReservoirValid = restir_isReservoirValid(temporalReservoir);
        if (!finalReservoirValid) {
            temporalReservoir.Y.w = -1.0;
        }
        uint finalPackedDirection = packedReservoirDirectionValid && finalReservoirValid
            ? packedReservoirDirection
            : restir_reservoir_packDirection(temporalReservoir.Y.xyz);
        uvec4 packedReservoir = uvec4(
            finalPackedDirection,
            floatBitsToUint(temporalReservoir.m),
            floatBitsToUint(temporalReservoir.avgWY),
            floatBitsToUint(temporalReservoir.Y.w)
        );
        transient_restir_reservoirTemporal_store(texelPos, packedReservoir);
    }
}
