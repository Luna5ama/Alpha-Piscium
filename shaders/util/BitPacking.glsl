#ifndef INCLUDE_util_BitPacking_glsl
#define INCLUDE_util_BitPacking_glsl a
#include "/_Base.glsl"

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

#endif