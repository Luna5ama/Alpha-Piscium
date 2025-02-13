#ifndef INCLUDE_util_Coords_glsl
#define INCLUDE_util_Coords_glsl
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

vec3 coords_octDecode11(vec2 f) {
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += mix(vec2(t), vec2(-t), vec2(greaterThanEqual(n.xy, vec2(0.0))));
    return normalize(n);
}

vec2 coords_octEncode11(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = mix(OctWrap(n.xy), n.xy, float(n.z >= 0.0));
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

vec3 coords_octDecode01(vec2 f) {
    return coords_octDecode11(f * 2.0 - 1.0);
}

vec2 coords_octEncode01(vec3 n) {
    return coords_octEncode11(n) * 0.5 + 0.5;
}

vec2 coords_equirectanglarForward(vec3 direction) {
    float phi = atan(direction.z, direction.x); // Horizontal angle (longitude)
    float theta = asin(direction.y); // Vertical angle (latitude)

    // Map angles to the [0, 1] range for UV coordinates
    vec2 uv = vec2((phi + PI) / (2.0 * PI), (theta + PI / 2.0) / PI);
    return uv;
}

vec3 coords_equirectanglarBackward(vec2 uv) {
    // Map UV back to angles
    float phi = uv.x * 2.0 * PI - PI; // Longitude
    float theta = uv.y * PI - PI / 2.0; // Latitude

    // Convert spherical coordinates back to a 3D direction vector
    vec3 direction = vec3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi));
    return normalize(direction);
}

vec2 coords_mercatorForward(vec3 direction) {
    float lon = atan(direction.z, direction.x);   // Longitude
    float lat = asin(direction.y);                // Latitude

    // Convert longitude and latitude to UV coordinates with Y flipped
    vec2 uv = vec2((lon + PI) / (2.0 * PI), 0.5 + log(tan(PI / 4.0 + lat / 2.0)) / (2.0 * PI));
    return uv;
}

vec3 coords_mercatorBackward(vec2 uv) {
    // Convert UV coordinates back to longitude and latitude with Y flipped
    float lon = uv.x * 2.0 * PI - PI;               // Longitude
    float lat = 2.0 * atan(exp((uv.y - 0.5) * 2.0 * PI)) - PI / 2.0;  // Latitude

    // Convert longitude and latitude back to a 3D direction vector
    vec3 direction = vec3(cos(lat) * cos(lon), sin(lat), cos(lat) * sin(lon));
    return normalize(direction);
}

#endif