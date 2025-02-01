#ifndef INCLUDE_util_Coords.glsl
#define INCLUDE_util_Coords.glsl
#include "../_Base.glsl"
#include "Math.glsl"

float coords_linearizeDepth(float depth, float near, float far) {
    return (near * far) / (depth * (near - far) + far);
}

vec3 coords_toViewCoord(vec2 texCoord, float viewZ, mat4 projInv) {
    vec2 ndcXY = texCoord * 2.0 - 1.0;
    vec2 clipXY = ndcXY * -viewZ;
    vec2 viewXY = clipXY * vec2(projInv[0][0], projInv[1][1]);
    return vec3(viewXY, viewZ);
}

vec2 OctWrap(vec2 v) {
    return (1.0 - abs(v.yx)) * mix(vec2(-1.0), vec2(1.0), vec2(greaterThanEqual(v.xy, vec2(0.0))));
}

vec3 coords_polarAzimuthEqualAreaInverse(vec2 f) {
    f = f * 2.0 - 1.0;

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += mix(vec2(t), vec2(-t), vec2(greaterThanEqual(n.xy, vec2(0.0))));
    return normalize(n);
}

vec2 coords_polarAzimuthEqualArea(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = mix(OctWrap(n.xy), n.xy, float(n.z >= 0.0));
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}

vec4 coords_projDiv(mat4 m, vec4 c) {
    vec4 r = m * c;
    return r / r.w;
}

vec4 coord_sceneCurrToPrev(vec4 sceneCurr) {
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec4 scenePrev = sceneCurr;
    scenePrev.xyz += cameraDelta;
    return scenePrev;
}

vec4 coord_scenePrevToCurr(vec4 scenePrev) {
    vec3 cameraDelta = cameraPosition - previousCameraPosition;
    vec4 sceneCurr = scenePrev;
    sceneCurr.xyz -= cameraDelta;
    return sceneCurr;
}

#endif