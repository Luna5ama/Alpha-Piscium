#ifndef INCLUDE_techniques_gi_ReservoirSplat_glsl
#define INCLUDE_techniques_gi_ReservoirSplat_glsl a

#include "/techniques/gi/Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Math.glsl"

#define RESTIR_SPLAT_NULL 0u
#define RESTIR_SPLAT_MAX_CHAIN_LENGTH 64u

#define RESTIR_SPLAT_PRIMARY_SUBPIXEL_BITS 5u
#define RESTIR_SPLAT_PRIMARY_SUBPIXEL_MASK 31u
#define RESTIR_SPLAT_PRIMARY_DEPTH_MASK 0xfffffc00u
#define RESTIR_SPLAT_PRIMARY_SUBPIXEL_SCALE 16.0

uint restir_splatEncodeNode(ivec2 texelPos) {
    return (uint(texelPos.y) << 16u) | (uint(texelPos.x) + 1u);
}

ivec2 restir_splatDecodeNode(uint node) {
    return ivec2(int((node & 0xffffu) - 1u), int(node >> 16u));
}

uint restir_splatFetchCurrentPrimary(ivec2 texelPos) {
    return transient_restir_primary_fetch(texelPos).x;
}

uint restir_splatPackPrimary(ivec2 texelPos, vec3 viewPos, mat4 projection) {
    vec2 screenPixel = coords_viewToScreen(viewPos, projection).xy * uval_mainImageSize;
    vec2 relativePixel = clamp(screenPixel - vec2(texelPos) + 0.5, vec2(0.0), vec2(1.999999));
    uvec2 packedSubpixel = uvec2(relativePixel * RESTIR_SPLAT_PRIMARY_SUBPIXEL_SCALE);
    uint depthBits = floatBitsToUint(viewPos.z);
    uint depthRoundBias = 0x1ffu + ((depthBits >> 10u) & 1u);
    uint packedDepth = (depthBits + depthRoundBias) & RESTIR_SPLAT_PRIMARY_DEPTH_MASK;
    return packedDepth | packedSubpixel.x | (packedSubpixel.y << RESTIR_SPLAT_PRIMARY_SUBPIXEL_BITS);
}

vec2 restir_splatUnpackPrimaryOffset(uint packedPrimary) {
    uvec2 packedSubpixel = uvec2(
        packedPrimary & RESTIR_SPLAT_PRIMARY_SUBPIXEL_MASK,
        (packedPrimary >> RESTIR_SPLAT_PRIMARY_SUBPIXEL_BITS) & RESTIR_SPLAT_PRIMARY_SUBPIXEL_MASK
    );
    return (vec2(packedSubpixel) + 0.5) * (1.0 / RESTIR_SPLAT_PRIMARY_SUBPIXEL_SCALE) - 0.5;
}

vec3 restir_splatUnpackPrimary(ivec2 texelPos, uint packedPrimary, mat4 projectionInverse) {
    vec2 relativePixel = restir_splatUnpackPrimaryOffset(packedPrimary);
    vec2 screenPos = (vec2(texelPos) + relativePixel) * uval_mainImageSizeRcp;
    float viewZ = uintBitsToFloat(packedPrimary & RESTIR_SPLAT_PRIMARY_DEPTH_MASK);
    return coords_toViewCoord(screenPos, viewZ, projectionInverse);
}

float restir_splatSurfaceConfidence(
    vec3 exactViewPos,
    vec3 exactGeomNormal,
    vec3 centerViewPos,
    vec3 centerGeomNormal
) {
    float normalDot = saturate(dot(exactGeomNormal, centerGeomNormal));
    float planeDistance = gi_planeDistance(exactViewPos, exactGeomNormal, centerViewPos, centerGeomNormal);
    float planeDistanceThreshold = max(0.01, abs(centerViewPos.z) * 0.001);
    return float(normalDot >= 0.99 && planeDistance <= planeDistanceThreshold);
}

float restir_splatPrimaryJacobian(
    vec3 prevViewPos,
    vec3 prevGeomNormal,
    vec3 currViewPos,
    vec3 currGeomNormal
) {
    float prevDist2 = dot(prevViewPos, prevViewPos);
    float currDist2 = dot(currViewPos, currViewPos);
    if (
        prevDist2 <= 1e-8
        || currDist2 <= 1e-8
        || !restir_isFinite(prevDist2)
        || !restir_isFinite(currDist2)
    ) {
        return 0.0;
    }

    float prevRcpDist = inversesqrt(prevDist2);
    float currRcpDist = inversesqrt(currDist2);
    float prevCosN = dot(prevGeomNormal, -prevViewPos * prevRcpDist);
    float currCosN = dot(currGeomNormal, -currViewPos * currRcpDist);
    if (
        prevCosN <= RESTIR_RECONNECTION_MIN_COSINE
        || currCosN <= RESTIR_RECONNECTION_MIN_COSINE
    ) {
        return 0.0;
    }

    float prevCosV = abs(prevViewPos.z) * prevRcpDist;
    float currCosV = abs(currViewPos.z) * currRcpDist;
    float prevProjectionScale = abs(global_prevCamProj[0][0] * global_prevCamProj[1][1]);
    float currProjectionScale = abs(global_camProj[0][0] * global_camProj[1][1]);
    float log2Jacobian = log2(currCosN) - log2(prevCosN)
        + 3.0 * (log2(prevCosV) - log2(currCosV))
        + log2(prevDist2) - log2(currDist2)
        + log2(currProjectionScale) - log2(prevProjectionScale);
    if (
        !restir_isFinite(log2Jacobian)
        || abs(log2Jacobian) > RESTIR_RECONNECTION_MAX_LOG2_JACOBIAN
    ) {
        return 0.0;
    }
    return exp2(log2Jacobian);
}

#endif
