/*
    References:
        [WRE18] Wrensch, Benjamin. "Reverse Z Cheatsheet". IOLITE Development Blog. 2023.
*/
#ifndef INCLUDE_util_Coords_glsl
#define INCLUDE_util_Coords_glsl a
#include "Math.glsl"
#include "Mat3.glsl"

float coords_linearizeDepth(float depth, float near, float far) {
    return (near * far) / (depth * (near - far) + far);
}

float coords_viewZToReversedZ(float viewZ, float near) {
    return near / -viewZ;
}

float coords_reversedZToViewZ(float revZ, float near) {
    return near / -revZ;
}

vec3 coords_toViewCoord(vec2 texCoord, float viewZ, mat4 projInv) {
    vec2 ndcXY = texCoord * 2.0 - 1.0;
    vec2 clipXY = ndcXY * -viewZ;
    vec2 viewXY = clipXY * vec2(projInv[0][0], projInv[1][1]);
    return vec3(viewXY, viewZ);
}

vec2 OctWrap(vec2 v) {
    return (1.0 - abs(v.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(v.xy, vec2(0.0)));
}

vec3 coords_octDecode11(vec2 f) {
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += mix(vec2(t), vec2(-t), greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}

vec2 coords_octEncode11(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = n.z >= 0.0 ? n.xy : OctWrap(n.xy);
    return n.xy;
}

vec4 coords_projDiv(mat4 m, vec4 c) {
    vec4 r = m * c;
    return r / r.w;
}

vec4 coord_sceneCurrToPrev(vec4 sceneCurr, bool isHand) {
    vec3 cameraDelta = isHand ? vec3(0.0) : uval_cameraDelta;
    vec4 scenePrev = sceneCurr;
    scenePrev.xyz += cameraDelta;
    return scenePrev;
}

vec4 coord_sceneCurrToPrev(vec4 sceneCurr) {
    vec3 cameraDelta = uval_cameraDelta;
    vec4 scenePrev = sceneCurr;
    scenePrev.xyz += cameraDelta;
    return scenePrev;
}

vec4 coord_scenePrevToCurr(vec4 scenePrev) {
    vec3 cameraDelta = uval_cameraDelta;
    vec4 sceneCurr = scenePrev;
    sceneCurr.xyz -= cameraDelta;
    return sceneCurr;
}

vec4 coord_viewCurrToPrev(vec4 currViewPos, bool isHand) {
    vec4 currScenePos = gbufferModelViewInverse * currViewPos;
    vec4 prevViewCoord;
    if (isHand) {
        currScenePos.xyz = currScenePos.xyz + gbufferModelViewInverse[3].xyz;
        vec4 prevScenePos = currScenePos;
        prevScenePos.xyz = prevScenePos.xyz - gbufferPrevModelViewInverse[3].xyz;
        prevViewCoord = gbufferModelView * prevScenePos;
    } else {
        vec4 prevScenePos = coord_sceneCurrToPrev(currScenePos);
        prevViewCoord = gbufferPrevModelView * prevScenePos;
    }
    return prevViewCoord;
}

vec3 coords_octDecode01(vec2 f) {
    return coords_octDecode11(f * 2.0 - 1.0);
}

vec2 coords_octEncode01(vec3 n) {
    return coords_octEncode11(n) * 0.5 + 0.5;
}

vec2 coords_equirectanglarForward(vec3 direction) {
    float phi = atan(direction.z, direction.x);// Horizontal angle (longitude)
    float theta = asin(direction.y);// Vertical angle (latitude)

    // Map angles to the [0, 1] range for UV coordinates
    vec2 uv = vec2((phi + PI) / (2.0 * PI), (theta + PI / 2.0) / PI);
    return uv;
}

vec3 coords_equirectanglarBackward(vec2 uv) {
    // Map UV back to angles
    float phi = uv.x * 2.0 * PI - PI;// Longitude
    float theta = uv.y * PI - PI / 2.0;// Latitude

    // Convert spherical coordinates back to a 3D direction vector
    vec3 direction = vec3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi));
    return normalize(direction);
}

vec2 coords_equirectanglarForwardHorizonBoost(vec3 direction) {
    float phi = atan(direction.z, direction.x);// Horizontal angle (longitude)
    float theta = asin(direction.y);// Vertical angle (latitude)

    // Map angles to the [0, 1] range for UV coordinates
    vec2 uv = vec2((phi + PI) / (2.0 * PI), 0.5 + 0.5 * sign(theta) * sqrt(abs(theta) * RCP_PI_HALF));
    return uv;
}

vec3 coords_equirectanglarBackwardHorizonBoost(vec2 uv) {
    // Map UV back to angles
    float phi = uv.x * 2.0 * PI - PI; // Longitude
    float theta = uv.y * 2.0 - 1.0;
    theta = sign(theta) * pow2(theta);
    theta *= PI_HALF;

    // Convert spherical coordinates back to a 3D direction vector
    vec3 direction = vec3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi));
    return normalize(direction);
}

vec2 coords_mercatorForward(vec3 direction) {
    float lon = atan(direction.z, direction.x);// Longitude
    float lat = asin(direction.y);// Latitude

    // Convert longitude and latitude to UV coordinates with Y flipped
    vec2 uv = vec2((lon + PI) / (2.0 * PI), 0.5 + log(tan(PI / 4.0 + lat / 2.0)) / (2.0 * PI));
    return uv;
}

vec3 coords_mercatorBackward(vec2 uv) {
    // Convert UV coordinates back to longitude and latitude with Y flipped
    float lon = uv.x * 2.0 * PI - PI;// Longitude
    float lat = 2.0 * atan(exp((uv.y - 0.5) * 2.0 * PI)) - PI / 2.0;// Latitude

    // Convert longitude and latitude back to a 3D direction vector
    vec3 direction = vec3(cos(lat) * cos(lon), sin(lat), cos(lat) * sin(lon));
    return normalize(direction);
}

// xy: uv
// z: sign of the major axis (0=-, 1=+)
// w: index of the major axis (0=X, 1=Y, 2=Z)
void coords_cubeMapForward(vec3 direction, out vec2 sliceUV, out vec2 sliceID) {
    vec4 axisDataX = vec4(direction.yz, direction.x, 0.0);
    vec4 axisDataY = vec4(direction.xz, direction.y, 1.0);
    vec4 axisDataZ = vec4(direction.xy, direction.z, 2.0);
    vec4 maxAxisData = axisDataX;
    maxAxisData = abs(direction.y) > abs(maxAxisData.z) ? axisDataY : maxAxisData;
    maxAxisData = abs(direction.z) > abs(maxAxisData.z) ? axisDataZ : maxAxisData;
    sliceUV = (maxAxisData.xy / abs(maxAxisData.z)) * 0.5 + 0.5;
    sliceID = vec2(float(maxAxisData.z > 0.0), float(maxAxisData.w));
}

void coords_cubeMapBackward(out vec3 direction, vec2 sliceUV, vec2 sliceID) {
    vec3 directionTemp = vec3(sliceID.x, sliceUV) * 2.0 - 1.0;
    directionTemp = sliceID.y == 1.0 ? directionTemp.yxz : directionTemp;
    directionTemp = sliceID.y == 2.0 ? directionTemp.yzx : directionTemp;
    direction = normalize(directionTemp);
}

vec2 coords_texelToUV(ivec2 texelPos, vec2 rcpTextureSize) {
    return saturate((vec2(texelPos) + 0.5) * rcpTextureSize);
}

ivec2 coords_clampTexelPos(ivec2 texelPos, ivec2 imageSizeV) {
    return clamp(texelPos, ivec2(0), imageSizeV);
}

const mat3 _COORDS_EQUATORIAL_TO_GALACTIC = mat3(
    -0.0548755604, 0.4941094279, -0.8676661490,
    -0.8734370902, -0.4448296300,  -0.1980763734,
    -0.4838350155, 0.7469822445,  0.4559837762
);

vec3 coords_equatorialToGalactic(vec3 equatorial) {
    return _COORDS_EQUATORIAL_TO_GALACTIC * equatorial;
}

const mat3 _COORDS_WORLD_TO_EQUATORIAL = mat3(
    0.0,  1.0,  0.0,   // Y+ (Up) -> X+ (RA 0h)
    1.0,  0.0,  0.0,   // X+ (East) -> Y+ (RA 6h)
    0.0,  0.0,  -1.0    // Z- (North) -> Z+ (Dec +90°)
);

vec3 coords_worldToEquatorial(vec3 world) {
    return _COORDS_WORLD_TO_EQUATORIAL * world;
}

// Converts equatorial coordinates to ecliptic coordinates
// equatorial: input vector in equatorial coordinates
// solarLon: longitude of the Sun in radians, 0.0 PI = Spring Equinox, 0.5 PI = Summer Solstice, 1.0 PI = Autumn Equinox, 1.5 PI = Winter Solstice
// hourAngle: hour angle of the observer in radians, 0.0 = 0h, 0.5 PI = 6h, 1.0 PI = 12h, 1.5 PI = 18h
// observerLat: latitude of the observer in radians
vec3 coords_equatorial_observerRotation(vec3 equatorial, float solarLon, float hourAngle, float observerLat) {
    mat3 latRotation = mat3_rotateY(observerLat);
    mat3 solarRotation = mat3_rotateZ(PI - solarLon - hourAngle);
    return solarRotation * latRotation * equatorial;
}

vec2 coords_equatorial_rectangularToSpherical(vec3 equatorial) {
    float dec = asin(equatorial.z); // Declination
    float ra = atan(equatorial.y, equatorial.x); // Right Ascension
    return vec2(ra, dec);
}

vec3 coords_dir_viewToWorld(vec3 dirView) {
    return normalize((mat3(gbufferModelViewInverse) * dirView));
}

vec3 coords_dir_worldToView(vec3 dirWorld) {
    return normalize((mat3(gbufferModelView) * dirWorld));
}

vec3 coords_pos_viewToWorld(vec3 posView, mat4 viewMatInverse) {
    return (gbufferModelViewInverse * vec4(posView, 1.0)).xyz;
}

vec3 coords_pos_worldToView(vec3 posWorld, mat4 viewMat) {
    return (gbufferModelView * vec4(posWorld, 1.0)).xyz;
}

vec3 coords_viewToScreen(vec3 viewPos, mat4 proj) {
    vec4 clipPos = proj * vec4(viewPos, 1.0);
    vec3 ndcPos = clipPos.xyz / clipPos.w;
    return ndcPos * 0.5 + 0.5;
}

#endif