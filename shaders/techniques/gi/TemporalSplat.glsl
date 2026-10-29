#ifndef INCLUDE_techniques_gi_TemporalSplat_glsl
#define INCLUDE_techniques_gi_TemporalSplat_glsl a

#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/gi/ReservoirSplat.glsl"

shared mat3 restir_prevViewToCurrView;
shared vec3 restir_prevViewToCurrViewTrans;

void restir_temporalSplatInitShared() {
    if (gl_LocalInvocationIndex == 0u) {
        restir_prevViewToCurrView = mat3(gbufferModelView) * mat3(gbufferPrevModelViewInverse);
        restir_prevViewToCurrViewTrans = mat3(gbufferModelView)
            * (gbufferPrevModelViewInverse[3].xyz - uval_cameraDelta)
            + gbufferModelView[3].xyz;
    }
}

ReSTIRReservoir restir_temporalSplatReadPreviousReservoir(ivec2 texelPos) {
    return restir_reservoir_unpack(history_restir_reservoirTemporal_fetch(texelPos));
}

uint restir_temporalSplatReadPreviousPrimary(ivec2 texelPos) {
    return history_restir_primary_fetch(texelPos).x;
}

void restir_temporalSplatWriteMetadata(
    ivec2 texelPos,
    uint nextNode,
    uint packedPrimary,
    float primaryJacobian
) {
    // Spatial pass 0 replaces this splat payload with pairwise state after chain resolution.
    transient_restir_pairwiseMISMetadata_store(texelPos, uvec4(
        nextNode,
        packedPrimary,
        floatBitsToUint(primaryJacobian),
        0u
    ));
}

void restir_temporalSplat(ivec2 texelPos) {
    ReSTIRReservoir previousReservoir = restir_temporalSplatReadPreviousReservoir(texelPos);
    uint packedPrimary = restir_temporalSplatReadPreviousPrimary(texelPos);
    if (!restir_isReservoirValid(previousReservoir) || packedPrimary == 0u) {
        return;
    }

    vec3 prevPrimary = restir_splatUnpackPrimary(texelPos, packedPrimary, global_prevCamProjInverse);
    vec3 currPrimary = restir_prevViewToCurrView * prevPrimary + restir_prevViewToCurrViewTrans;
    vec4 currClip = global_camProj * vec4(currPrimary, 1.0);
    if (currClip.z <= 0.0 || any(greaterThanEqual(abs(currClip.xy), currClip.ww))) {
        return;
    }

    vec2 currScreen = currClip.xy / currClip.w * 0.5 + 0.5 + uval_taaJitterUV;
    ivec2 outputTexelPos = ivec2(floor(currScreen * uval_mainImageSize));
    if (any(lessThan(outputTexelPos, ivec2(0))) || any(greaterThanEqual(outputTexelPos, uval_mainImageSizeI))) {
        return;
    }

    float currViewZ = texelFetch(usam_gbufferSolidViewZ, outputTexelPos, 0).x;
    if (currViewZ <= -65536.0) {
        return;
    }

    vec2 centerScreen = coords_texelToUV(outputTexelPos, uval_mainImageSizeRcp) - uval_taaJitterUV;
    vec3 centerViewPos = coords_toViewCoord(centerScreen, currViewZ, global_camProjInverse);
    vec3 previousGeomNormal = history_geomViewNormal_fetch(texelPos).xyz * 2.0 - 1.0;
    vec3 prevGeomNormal = normalize(previousGeomNormal);
    vec3 transformedPrevGeomNormal = restir_prevViewToCurrView * prevGeomNormal;
    vec3 currGeomNormal = normalize(transformedPrevGeomNormal);
    vec3 centerGeomNormal = normalize(transient_geomViewNormal_fetch(outputTexelPos).xyz * 2.0 - 1.0);
    if (restir_splatSurfaceConfidence(currPrimary, currGeomNormal, centerViewPos, centerGeomNormal) <= 0.0) {
        return;
    }

    float primaryJacobian = 0.0;
    if (
        restir_isFinite(prevPrimary)
        && restir_isFinite(currPrimary)
        && restir_isFinite(previousGeomNormal)
        && dot(previousGeomNormal, previousGeomNormal) > 1e-4
    ) {
        primaryJacobian = restir_splatPrimaryJacobian(
            prevPrimary,
            prevGeomNormal,
            currPrimary,
            currGeomNormal
        );
    }

    uint node = restir_splatEncodeNode(texelPos);
    uint previousHead = transient_restir_primary_atomicExchange(outputTexelPos, node);
    restir_temporalSplatWriteMetadata(texelPos, previousHead, packedPrimary, primaryJacobian);
}

#endif
