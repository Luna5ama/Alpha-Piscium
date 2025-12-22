#version 460 compatibility

#include "/techniques/SSGI.glsl"
#include "/techniques/gi/Common.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_temp1;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 result = vec4(0.0);
        float hitDistSum = 0.0;

        const uint SPP = MC_SPP;
        uint baseRand = RANDOM_FRAME * SPP;
        for (uint i = 0u; i < SPP; ++i) {
            vec4 tempResult = ssgiRef(texelPos, baseRand + i);
            result.rgb += tempResult.rgb;
            if (tempResult.a >= 0.0) {
                result.a += min(tempResult.a, 16.0);
                hitDistSum += 1.0;
            }
        }
        result.rgb /= float(SPP);
        result.a = hitDistSum <= 0.0 ? -1.0 : clamp(result.a / hitDistSum, 0.0, 16.0);

//        imageStore(uimg_temp1, texelPos, vec4(result, 1.0));
        transient_ssgiOut_store(texelPos, result);
    }
}