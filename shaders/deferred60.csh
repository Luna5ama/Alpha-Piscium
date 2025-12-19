#version 460 compatibility

#include "/techniques/SSGI.glsl"
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
        vec3 result = vec3(0.0);

        const uint SPP = MC_SPP;
        uint baseRand = RANDOM_FRAME * SPP;
        for (uint i = 0u; i < SPP; ++i) {
            result += ssgiRef(texelPos, baseRand + i);
        }
        result /= float(SPP);

//        imageStore(uimg_temp1, texelPos, vec4(result, 1.0));
        transient_ssgiOut_store(texelPos, vec4(result, 1.0));
    }
}