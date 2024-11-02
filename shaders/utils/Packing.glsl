uint packing_packU8(float v) {
    return clamp(uint(v * 255.0), 0u, 255u);
}

float packing_unpackU8(uint v) {
    return float(v) / 255.0;
}

uint packing_packS10(float v) {
    return clamp(uint((v * 0.5 + 0.5) * 1022.0), 0u, 1023u);
}

float packing_unpackS10(uint v) {
    return max((float(v) - 511) / 511.0, -1.0);
}

uint packing_packU11(float v) {
    return clamp(uint(v * 2047.0), 0u, 2047u);
}

float packing_unpackU11(uint v) {
    return float(v) / 2047.0;
}

uint packing_packU12(float v) {
    return clamp(uint(v * 4095.0), 0u, 4095u);
}

float packing_unpackU12(uint v) {
    return float(v) / 4095.0;
}

uint packing_packS15(float v) {
    return clamp(uint((v * 0.5 + 0.5) * 32767.0), 0u, 32767u);
}

float packing_unpackS15(uint v) {
    return max((float(v) - 16383) / 16383.0, -1.0);
}

uint packing_packS16(float v) {
    return clamp(uint((v * 0.5 + 0.5) * 65535.0), 0u, 65535u);
}

float packing_unpackS16(uint v) {
    return max((float(v) - 32767) / 32767.0, -1.0);
}