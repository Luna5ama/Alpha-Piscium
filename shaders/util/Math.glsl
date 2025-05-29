#ifndef INCLUDE_util_Math_glsl
#define INCLUDE_util_Math_glsl a
#include "/_Base.glsl"

#define PI 3.14159265358979323846
#define PI_2 2.0 * PI
#define PI_HALF 0.5 * PI
#define PI_QUARTER 0.25 * PI
#define RCP_PI 1.0 / PI
#define RCP_PI_2 1.0 / PI_2
#define RCP_PI_HALF 1.0 / PI_HALF

#define FLT_MAX uintBitsToFloat(0x7F7FFFFF)
#define FLT_MIN uintBitsToFloat(0x00800000)
#define FLT_TRUE_MIN uintBitsToFloat(0x00000001)
#define FLT_POS_INF uintBitsToFloat(0x7F800000)
#define FLT_NEG_INF uintBitsToFloat(0xFF800000)

float rcp(float x) { return 1.0 / x; }
vec2 rcp(vec2 x) { return 1.0 / x; }
vec3 rcp(vec3 x) { return 1.0 / x; }
vec4 rcp(vec4 x) { return 1.0 / x; }

float saturate(float x) { return clamp(x, 0.0, 1.0); }

vec2 saturate(vec2 x) { return clamp(x, 0.0, 1.0); }

vec3 saturate(vec3 x) { return clamp(x, 0.0, 1.0); }

vec4 saturate(vec4 x) { return clamp(x, 0.0, 1.0); }

float max2(vec2 v) { return max(v.x, v.y); }

float linearStep(float edge0, float edge1, float x) {
    return saturate((x - edge0) / (edge1 - edge0));
}

vec2 linearStep(float edge0, float edge1, vec2 x) {
    return saturate((x - edge0) / (edge1 - edge0));
}

vec3 linearStep(float edge0, float edge1, vec3 x) {
    return saturate((x - edge0) / (edge1 - edge0));
}

vec4 linearStep(float edge0, float edge1, vec4 x) {
    return saturate((x - edge0) / (edge1 - edge0));
}

float toRadians(float degrees) {
    return degrees * PI / 180.0;
}

float pow2(float x) { return x * x; }
vec2 pow2(vec2 x) { return x * x; }
vec3 pow2(vec3 x) { return x * x; }
vec4 pow2(vec4 x) { return x * x; }

float pow3(float x) { return x * x * x; }
vec2 pow3(vec2 x) { return x * x * x; }
vec3 pow3(vec3 x) { return x * x * x; }
vec4 pow3(vec4 x) { return x * x * x; }

float pow4(float x) {
    float x2 = x * x;
    return x2 * x2;
}
vec2 pow4(vec2 x) {
    vec2 x2 = x * x;
    return x2 * x2;
}
vec3 pow4(vec3 x) {
    vec3 x2 = x * x;
    return x2 * x2;
}
vec4 pow4(vec4 x) {
    vec4 x2 = x * x;
    return x2 * x2;
}

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}
vec2 pow5(vec2 x) {
    vec2 x2 = x * x;
    return x2 * x2 * x;
}
vec3 pow5(vec3 x) {
    vec3 x2 = x * x;
    return x2 * x2 * x;
}
vec4 pow5(vec4 x) {
    vec4 x2 = x * x;
    return x2 * x2 * x;
}

float pow6(float x) {
    float x2 = x * x;
    return x2 * x2 * x2;
}
vec2 pow6(vec2 x) {
    vec2 x2 = x * x;
    return x2 * x2 * x2;
}
vec3 pow6(vec3 x) {
    vec3 x2 = x * x;
    return x2 * x2 * x2;
}
vec4 pow6(vec4 x) {
    vec4 x2 = x * x;
    return x2 * x2 * x2;
}

float lengthSq(float x) { return x * x; }
float lengthSq(vec2 x) { return dot(x, x); }
float lengthSq(vec3 x) { return dot(x, x); }
float lengthSq(vec4 x) { return dot(x, x); }

float rayleighPhase(float cosTheta) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks phase function for Mie scattering
float miePhase(float cosTheta, float g) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

#endif