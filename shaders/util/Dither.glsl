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

const uint DITHER_FP16_MAX_BITS = 0x477FE000u;

float dither_fp16(float x, float noiseV) {
    if (isnan(x) || x == 0.0) {
        return 0.0;
    }
    uint bits = floatBitsToUint(x);
    uint signBit = bits & 0x80000000u;
    uint magnitudeBits = min(bits & 0x7FFFFFFFu, DITHER_FP16_MAX_BITS);
    if (magnitudeBits < 0x38800000u) {
        // FP16 subnormals have a fixed 2^-24 spacing.
        float scaled = abs(x) * 16777216.0;
        float rounded = floor(scaled) + float(fract(scaled) > noiseV);
        return uintBitsToFloat(signBit | floatBitsToUint(rounded * exp2(-24.0)));
    }
    uint roundingOffset = 8191u - min(uint(noiseV * 8192.0), 8191u);
    uint roundedBits = (magnitudeBits + roundingOffset) & 0xFFFFE000u;
    return uintBitsToFloat(signBit | roundedBits);
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
