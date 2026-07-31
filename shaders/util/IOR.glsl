#ifndef INCLUDE_util_IOR_glsl
#define INCLUDE_util_IOR_glsl a

const float AIR_IOR = 1.00029;
const float WATER_IOR = 1.333;

vec3 ior_f0ToIor(vec3 f0) {
    vec3 f0Sqrt = sqrt(f0) * 0.99999;
    return AIR_IOR * ((1.0 + f0Sqrt) / (1.0 - f0Sqrt));
}

float ior_f0ToIor(float f0) {
    float f0Sqrt = sqrt(f0) * 0.99999;
    return AIR_IOR * ((1.0 + f0Sqrt) / (1.0 - f0Sqrt));
}

float ior_iorToF0(float ior) {
    return pow2((ior - AIR_IOR) / (ior + AIR_IOR));
}

#endif