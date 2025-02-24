#ifndef INCLUDE_util_Dither_glsl
#define INCLUDE_util_Dither_glsl a

vec4 dither_u8(vec4 x, float noiseV) {
    vec4 result = x;
    result *= 255.0;
    result = round(result + (noiseV - 0.5));
    result /= 255.0;
    return result;
}

vec3 dither_u8(vec3 x, float noiseV) {
    vec3 result = x;
    result *= 255.0;
    result = round(result + (noiseV - 0.5));
    result /= 255.0;
    return result;
}

vec2 dither_u8(vec2 x, float noiseV) {
    vec2 result = x;
    result *= 255.0;
    result = round(result + (noiseV - 0.5));
    result /= 255.0;
    return result;
}

float dither_u8(float x, float noiseV) {
    float result = x;
    result *= 255.0;
    result = round(result + (noiseV - 0.5));
    result /= 255.0;
    return result;
}

vec4 dither_fp16(vec4 x, float noiseV) {
    return uintBitsToFloat(floatBitsToUint(x) + uint(float(0x7FFFu) * (noiseV - 0.5)));
}

vec3 dither_fp16(vec3 x, float noiseV) {
    return uintBitsToFloat(floatBitsToUint(x) + uint(float(0x7FFFu) * (noiseV - 0.5)));
}

vec2 dither_fp16(vec2 x, float noiseV) {
    return uintBitsToFloat(floatBitsToUint(x) + uint(float(0x7FFFu) * (noiseV - 0.5)));
}

float dither_fp16(float x, float noiseV) {
    return uintBitsToFloat(floatBitsToUint(x) + uint(float(0x7FFFu) * (noiseV - 0.5)));
}

#endif