/*
    References:
        [WHY23] Whyte, Killian. "Soft Clamp Function". 2023.
            https://math.stackexchange.com/questions/4726955/trying-to-find-a-soft-clamp-function-a-bit-differenct-characterstics-from-smo

        You can find full license texts in /licenses
*/
#ifndef INCLUDE_util_Math_glsl
#define INCLUDE_util_Math_glsl a
#include "/Base.glsl"

#define PI 3.14159265358979323846
#define PI_2 (2.0 * PI)
#define PI_HALF (0.5 * PI)
#define PI_QUARTER (0.25 * PI)
#define RCP_PI (1.0 / PI)
#define RCP_PI_2 (1.0 / PI_2)
#define RCP_PI_HALF (1.0 / PI_HALF)

#define GOLDEN_RATIO 1.618033988749
#define GOLDEN_ANGLE 2.39996322972865332
#define SPHERE_SOLID_ANGLE (4.0 * PI)

#define FLT_MAX uintBitsToFloat(0x7F7FFFFF)
#define FLT_MIN uintBitsToFloat(0x00800000)
#define FLT_TRUE_MIN uintBitsToFloat(0x00000001)
#define FLT_POS_INF uintBitsToFloat(0x7F800000)
#define FLT_NEG_INF uintBitsToFloat(0xFF800000)
#define FP16_MAX 65504.0

#define INT32_MAX 2147483647
#define INT32_MIN -2147483648

#define rcp(x) (1.0 / (x))
#define saturate(x) clamp(x, 0.0, 1.0)

float mmin3(float x, float y, float z) { return min(min(x, y), z); }
float mmax3(float x, float y, float z) { return max(max(x, y), z); }

float max2(float x, float y) { return max(x, y); }
float max2(vec2 v) { return max(v.x, v.y); }
float mmax3(vec3 v) { return max(max(v.x, v.y), v.z); }
float max4(float x, float y, float z, float w) { return max(max(x, y), max(z, w)); }
float max4(vec4 v) { return max(max(v.x, v.y), max(v.z, v.w)); }

float min2(float x, float y) { return min(x, y); }
float min2(vec2 v) { return min(v.x, v.y); }
float mmin3(vec3 v) { return min(min(v.x, v.y), v.z); }
float min4(float x, float y, float z, float w) { return min(min(x, y), min(z, w)); }
float min4(vec4 v) { return min(min(v.x, v.y), min(v.z, v.w)); }

float sum2(float x, float y) { return x + y; }
float sum2(vec2 v) { return v.x + v.y; }
float sum3(float x, float y, float z) { return x + y + z; }
float sum3(vec3 v) { return v.x + v.y + v.z; }
float sum4(float x, float y, float z, float w) { return x + y + z + w; }
float sum4(vec4 v) { return v.x + v.y + v.z + v.w; }

float linearStep(float edge0, float edge1, float x) { return saturate((x - edge0) / (edge1 - edge0)); }
vec2 linearStep(float edge0, float edge1, vec2 x) { return saturate((x - edge0) / (edge1 - edge0)); }
vec3 linearStep(float edge0, float edge1, vec3 x) { return saturate((x - edge0) / (edge1 - edge0)); }
vec4 linearStep(float edge0, float edge1, vec4 x) { return saturate((x - edge0) / (edge1 - edge0)); }

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

float pow7(float x) {
    float x2 = x * x;
    return x2 * x2 * x2 * x;
}
vec2 pow7(vec2 x) {
    vec2 x2 = x * x;
    return x2 * x2 * x2 * x;
}
vec3 pow7(vec3 x) {
    vec3 x2 = x * x;
    return x2 * x2 * x2 * x;
}
vec4 pow7(vec4 x) {
    vec4 x2 = x * x;
    return x2 * x2 * x2 * x;
}

float pow8(float x) {
    float x2 = x * x;
    float x4 = x2 * x2;
    return x4 * x4;
}
vec2 pow8(vec2 x) {
    vec2 x2 = x * x;
    vec2 x4 = x2 * x2;
    return x4 * x4;
}
vec3 pow8(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return x4 * x4;
}
vec4 pow8(vec4 x) {
    vec4 x2 = x * x;
    vec4 x4 = x2 * x2;
    return x4 * x4;
}

float lengthSq(float x) { return x * x; }
float lengthSq(vec2 x) { return dot(x, x); }
float lengthSq(vec3 x) { return dot(x, x); }
float lengthSq(vec4 x) { return dot(x, x); }

// - r0: ray origin
// - rd: normalized ray direction
// - s0: sphere center
// - sR: sphere radius
// - Returns distance from r0 to first intersecion with sphere,
//   or -1.0 if no intersection.
float raySphereIntersectNearest(vec3 r0, vec3 rd, vec3 s0, float sR) {
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sR * sR);
    float delta = b * b - 4.0*a*c;
    if (delta < 0.0 || a == 0.0) {
        return -1.0;
    }
    float sol0 = (-b - sqrt(delta)) / (2.0*a);
    float sol1 = (-b + sqrt(delta)) / (2.0*a);
    if (sol0 < 0.0 && sol1 < 0.0) {
        return -1.0;
    }
    if (sol0 < 0.0) {
        return max(0.0, sol1);
    }
    else if (sol1 < 0.0) {
        return max(0.0, sol0);
    }
    return max(0.0, min(sol0, sol1));
}

// [WHY23]
float softMin(float x, float maxV) {
    float phiX = x - maxV / 2.0;
    float phi = maxV / (1.0 + exp((-4.0 * phiX) / maxV));
    return min(x, phi);
}

#endif