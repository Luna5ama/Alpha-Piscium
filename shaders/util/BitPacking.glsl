#ifndef INCLUDE_util_BitPacking_glsl
#define INCLUDE_util_BitPacking_glsl a
#include "/Base.glsl"

uint packU10(float v) { return uint(clamp(v, 0.0, 1.0) * 1023.0); }
float unpackU10(uint v) { return clamp(float(v) / 1023.0, 0.0, 1.0); }
uint packS10(float v) { return packU10(v * 0.5 + 0.5); }
float unpackS10(uint v) { return unpackU10(v) * 2.0 - 1.0; }

uint packU11(float v) { return uint(clamp(v, 0.0, 1.0) * 2047.0); }
float unpackU11(uint v) { return clamp(float(v) / 2047.0, 0.0, 1.0); }
uint packS11(float v) { return packU11(v * 0.5 + 0.5); }
float unpackS11(uint v) { return unpackU11(v) * 2.0 - 1.0; }

uint packU15(float v) { return uint(clamp(v, 0.0, 1.0) * 32767.0); }
float unpackU15(uint v) { return clamp(float(v) / 32767.0, 0.0, 1.0); }
uint packS15(float v) { return packU15(v * 0.5 + 0.5); }
float unpackS15(uint v) { return unpackU15(v) * 2.0 - 1.0; }

uint packU16(float v) { return uint(clamp(v, 0.0, 1.0) * 65535.0); }
float unpackU16(uint v) { return clamp(float(v) / 65535.0, 0.0, 1.0); }
uint packS16(float v) { return packU16(v * 0.5 + 0.5); }
float unpackS16(uint v) { return unpackU16(v) * 2.0 - 1.0; }

uint packUInt2x16(uvec2 v) {
    uint result = bitfieldInsert(0u, v.x, 0, 16);
    result = bitfieldInsert(result, v.y, 16, 16);
    return result;
}

uvec2 unpackUInt2x16(uint v) {
    uvec2 result = uvec2(bitfieldExtract(v, 0, 16), bitfieldExtract(v, 16, 16));
    return result;
}

uint packSnorm3x10(vec3 v) {
    uint result = packS10(v.x);
    result = bitfieldInsert(result, packS10(v.y), 10, 10);
    result = bitfieldInsert(result, packS10(v.z), 20, 10);
    return result;
}

vec3 unpackSnorm3x10(uint v) {
    vec3 result;
    result.x = unpackS10(bitfieldExtract(v, 0, 10));
    result.y = unpackS10(bitfieldExtract(v, 10, 10));
    result.z = unpackS10(bitfieldExtract(v, 20, 10));
    return result;
}

uvec2 packHalf4x16(vec4 v) {
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}

vec4 unpackHalf4x16(uvec2 v) {
    return vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y));
}


#endif