#ifndef INCLUDE_techniques_gi_ReprojectInfo_glsl
#define INCLUDE_techniques_gi_ReprojectInfo_glsl a

struct ReprojectInfo {
    vec4 bilateralWeights;
    float historyResetFactor;
    vec2 curr2PrevScreenPos;
};

ReprojectInfo reprojectInfo_init() {
    ReprojectInfo info;
    info.bilateralWeights = vec4(0.0);
    info.historyResetFactor = 0.0;
    info.curr2PrevScreenPos = vec2(-1.0);
    return info;
}

ReprojectInfo reprojectInfo_unpack(uvec4 packedData) {
    ReprojectInfo info;
    info.curr2PrevScreenPos = uintBitsToFloat(packedData.xy);
    info.bilateralWeights = unpackUnorm4x8(packedData.z);
    info.historyResetFactor = uintBitsToFloat(packedData.w);
    return info;
}

uvec4 reprojectInfo_pack(ReprojectInfo info) {
    uvec4 packedData;
    packedData.xy = floatBitsToUint(info.curr2PrevScreenPos);
    packedData.z = packUnorm4x8(info.bilateralWeights);
    packedData.w = floatBitsToUint(info.historyResetFactor);
    return packedData;
}

#endif
