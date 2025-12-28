#include "/Base.glsl"


vec4 rcas_loadInput(ivec2 texelPos);
void rcas_storeOutput(ivec2 texelPos, vec4 color);
// Returns sharpness value in [0, 1]
float rcas_sharpness();

vec4 LoadRCas_Input(ivec2 p) {
    return rcas_loadInput(p);
}

void StoreRCasOutput(ivec2 p, vec4 color) {
    rcas_storeOutput(p, color);
}

uvec4 RCasSample() {
    return uvec4(0);
}


uvec4 RCasConfig() {
    // https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/sdk/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp#L1107
    float sharpnessRemapped = rcas_sharpness() * -2.0 + 2.0;
    // https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/sdk/include/FidelityFX/gpu/fsr1/ffx_fsr1.h#L661-L672
    float sharpness = exp2(-sharpnessRemapped);

    uvec4 config = uvec4(0);
    config.x = floatBitsToUint(sharpness);
    config.y = packHalf2x16(vec2(sharpness));
    return config;
}

#include "ffx_fsr1_rcas.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    RCAS(uvec3(gl_LocalInvocationIndex, 0u, 0u), gl_WorkGroupID, gl_GlobalInvocationID);
}