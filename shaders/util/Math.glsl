#ifndef INCLUDE_Math.glsl
#define INCLUDE_Math.glsl
const float PI_CONST = 3.14159265358979323846;
const float PI_2_CONST = 2.0 * PI_CONST;
const float PI_HALF_CONST = 0.5 * PI_CONST;
const float PI_QUARTER_CONST = 0.25 * PI_CONST;
const float RCP_PI_CONST = 1.0 / PI_CONST;
const float RCP_PI_2_CONST = 1.0 / PI_2_CONST;
const float RCP_PI_HALF_CONST = 1.0 / PI_HALF_CONST;

float rcp(float x) { return 1.0 / x; }
vec2 rcp(vec2 x) { return 1.0 / x; }
vec3 rcp(vec3 x) { return 1.0 / x; }
vec4 rcp(vec4 x) { return 1.0 / x; }

float saturate(float x) { return clamp(x, 0.0, 1.0); }

vec2 saturate(vec2 x) { return clamp(x, 0.0, 1.0); }

vec3 saturate(vec3 x) { return clamp(x, 0.0, 1.0); }

vec4 saturate(vec4 x) { return clamp(x, 0.0, 1.0); }

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
    return degrees * PI_CONST / 180.0;
}
#endif