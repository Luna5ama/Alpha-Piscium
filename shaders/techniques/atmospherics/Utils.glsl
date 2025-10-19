#ifndef INCLUDE_techniques_atmospherics_Utils_glsl
#define INCLUDE_techniques_atmospherics_Utils_glsl a

vec3 volumetrics_intergrateScatteringLerpLightOpticalDepth(vec3 sctCoeff, vec3 extCoeff, float segmentLen, float lightRayLen1, float lightRayLen2) {
    vec3 v = exp(-extCoeff * lightRayLen1) - exp(-extCoeff * (lightRayLen2 + segmentLen));
    vec3 numer = sctCoeff * segmentLen * v;
    vec3 denom = extCoeff * (lightRayLen2 - lightRayLen1 + segmentLen);
    if (any(equal(denom, vec3(0.0)))) return vec3(0.0);
    return numer / denom;
}

#endif