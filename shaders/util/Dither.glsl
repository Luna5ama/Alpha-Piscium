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

const int DITHER_FP16_MAX_BITS = 0x477FE000;

float dither_fp16(float x, float noiseV) {
    if (isnan(x) || x == 0.0) {
        return 0.0;
    }
    uint bits = floatBitsToUint(x);
    uint signBit = bits & 0x80000000u;
    int magnitudeBits = min(int(bits & 0x7fffffffu), DITHER_FP16_MAX_BITS);
    int delta = int(float(0x7fffu) * (noiseV - 0.5));
    magnitudeBits = clamp(magnitudeBits + delta, 0, DITHER_FP16_MAX_BITS);
    return uintBitsToFloat(signBit | uint(magnitudeBits));
}

vec2 dither_fp16(vec2 x, float noiseV) {
    return vec2(dither_fp16(x.x, noiseV), dither_fp16(x.y, noiseV));
}

vec3 dither_fp16(vec3 x, float noiseV) {
    return vec3(
        dither_fp16(x.x, noiseV),
        dither_fp16(x.y, noiseV),
        dither_fp16(x.z, noiseV)
    );
}

vec4 dither_fp16(vec4 x, float noiseV) {
    return vec4(
        dither_fp16(x.x, noiseV),
        dither_fp16(x.y, noiseV),
        dither_fp16(x.z, noiseV),
        dither_fp16(x.w, noiseV)
    );
}

#endif
