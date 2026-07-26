#include "/Base.glsl"

vec4 rcas_loadInput(ivec2 texelPos, bool center);

vec4 fsr1_rcasOutput;

vec4 LoadRCas_Input(ivec2 p, bool center) {
    return rcas_loadInput(p, center);
}

void StoreRCasOutput(ivec2 p, vec4 color) {
    fsr1_rcasOutput = color;
}

uvec4 RCasSample() {
    return uvec4(0);
}

uvec4 RCasConfig() {
    // https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/sdk/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp#L1107
    #ifdef SETTING_FSR3
    float sharpness = SETTING_FSR3_SHARPNESS;
    #else
    float sharpness = mix(1.0, SETTING_TAA_CAS_SHARPNESS, global_motionFactor.w);
    #endif
    float sharpnessRemapped = sharpness * -2.0 + 2.0;
    // https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/sdk/include/FidelityFX/gpu/fsr1/ffx_fsr1.h#L661-L672
    float sharpnessConfig = exp2(-sharpnessRemapped);

    uvec4 config = uvec4(0);
    config.x = floatBitsToUint(sharpnessConfig);
    config.y = packHalf2x16(vec2(sharpnessConfig));
    return config;
}

#include "ffx_fsr1_rcas.glsl"

vec4 fsr1_rcas(ivec2 outputTexelPos) {
    #ifdef SETTING_FSR3
    if (SETTING_FSR3_SHARPNESS == 0.0) return rcas_loadInput(outputTexelPos, true);
    #endif
    CurrFilter(FFX_MIN16_U2(outputTexelPos));
    return fsr1_rcasOutput;
}
