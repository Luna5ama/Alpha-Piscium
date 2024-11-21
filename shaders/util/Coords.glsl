/*
    References:
        [WEI] Weisstein, Eric W. "Lambert Azimuthal Equal-Area Projection." Wolfram MathWorld.
        https://mathworld.wolfram.com/LambertAzimuthalEqual-AreaProjection.html
*/
#ifndef INCLUDE_util_Coords.glsl
#define INCLUDE_util_Coords.glsl
#include "../_Base.glsl"
#include "Math.glsl"
#include "R2Seqs.glsl"

float coords_linearizeDepth(float depth, float near, float far) {
    return (near * far) / (depth * (near - far) + far);
}

vec3 coords_toViewCoord(vec2 texCoord, float viewZ, mat4 projInv) {
    vec2 ndcXY = texCoord * 2.0 - 1.0;
    vec2 clipXY = ndcXY * -viewZ;
    vec2 viewXY = clipXY * vec2(projInv[0][0], projInv[1][1]);
    return vec3(viewXY, viewZ);
}

// See [WEI]
// phi1: Standard parallel
// lambda0: Central longitude
vec3 coords_azimuthEqualAreaInverse(vec2 texCoord, float phi1, float lambda0) {
    float sinPhi1 = sin(phi1);
    float cosPhi1 = cos(phi1);

    vec2 xy0 = texCoord * 2.0 - 1.0;
    xy0 *= 2.0;

    float p = length(xy0);
    if (p > 2.0) return vec3(0.0);

    float c = 2.0 * asin(0.5 * p);

    float lat = asin((cos(c) * sinPhi1) + ((xy0.y * sin(c) * cosPhi1) / p));
    float lon = lambda0 + atan(p * cosPhi1 * cos(c) - xy0.y * sinPhi1 * sin(c), xy0.x * sin(c));

    vec3 normal;
    normal.x = cos(lat) * cos(lon);
    normal.y = sin(lat);
    normal.z = cos(lat) * sin(lon);
    return normal;
}

vec2 coords_azimuthEqualArea(vec3 unitVector, float phi1, float lambda0) {
    float sinPhi1 = sin(phi1);
    float cosPhi1 = cos(phi1);

    float lat = asin(unitVector.y);
    float lon = atan(unitVector.x, unitVector.z);

    float kPrime = sqrt(2.0 / (1.0 + sinPhi1 * sin(lat) + cosPhi1 * cos(lat) * cos(lon - lambda0)));
    float x = kPrime * cos(lat) * sin(lon - lambda0);
    float y = kPrime * (cosPhi1 * sin(lat) - sinPhi1 * cos(lat) * cos(lon - lambda0));

    vec2 xy0 = vec2(x, y);
    xy0 /= 2.0;
    xy0 = xy0 * 0.5 + 0.5;

    return xy0;
}

vec3 coords_polarAzimuthEqualAreaInverse(vec2 texCoord) {
    //    const float phi1 = PI_HALF_CONST; // Standard parallel
    //    const float lambda0 = 0.0; // Central longitude
    const float sinPhi1 = 1.0;
    const float cosPhi1 = 0.0;

    vec2 xy0 = texCoord * 2.0 - 1.0;
    xy0 *= 2.0;

    float p = length(xy0);
    if (p > 2.0) return vec3(0.0);

    float c = 2.0 * asin(0.5 * p);

    float lat = asin((cos(c) * sinPhi1) + ((xy0.y * sin(c) * cosPhi1) / p));
    float lon =  atan(p * cosPhi1 * cos(c) - xy0.y * sinPhi1 * sin(c), xy0.x * sin(c));

    vec3 normal;
    normal.x = cos(lat) * cos(lon);
    normal.y = sin(lat);
    normal.z = cos(lat) * sin(lon);
    return normal;
}

vec2 coords_polarAzimuthEqualArea(vec3 unitVector) {
    //    const float phi1 = PI_HALF_CONST; // Standard parallel
    //    const float lambda0 = 0.0; // Central longitude
    const float sinPhi1 = 1.0;
    const float cosPhi1 = 0.0;

    float lat = asin(unitVector.y);
    float lon = atan(unitVector.x, unitVector.z);

    float kPrime = sqrt(2.0 / (1.0 + sinPhi1 * sin(lat) + cosPhi1 * cos(lat) * cos(lon)));
    float x = kPrime * cos(lat) * sin(lon);
    float y = kPrime * (cosPhi1 * sin(lat) - sinPhi1 * cos(lat) * cos(lon));

    vec2 xy0 = vec2(x, y);
    xy0 /= 2.0;
    xy0 = xy0 * 0.5 + 0.5;

    return xy0;
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