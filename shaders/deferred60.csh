#version 460 compatibility

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32f) uniform restrict image2D uimg_temp1;
layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;
layout(rgba16f) uniform restrict image2D uimg_rgba16f;
#include "/techniques/SSGI.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        if (RANDOM_FRAME < MAX_FRAMES){
            if (RANDOM_FRAME >= 0) {
                vec3 result = vec3(0.0);

                #if USE_REFERENCE == 2
//                result = gtvbgi(texelPos);
                #elif USE_REFERENCE == 1
                const uint SPP = MC_SPP;
                uint baseRand = RANDOM_FRAME * SPP;
                for (uint i = 0u; i < SPP; ++i) {
                    result += ssgiRef(texelPos, baseRand + i);
                }
                result /= float(SPP);
                #else
//                result = uintBitsToFloat(imageLoad(uimg_csrgba32ui, csrgba32ui_temp4_texelToTexel(texelPos)).rgb);
                #endif

                transient_ssgiOut_store(texelPos, vec4(result, 16.0));
            }
//            imageStore(uimg_temp1, texelPos, ssgiOut);
        } else {
//            if (all(greaterThan(texelPos, uval_mainImageSizeI - 8))) {
//                imageStore(uimg_temp1, texelPos, vec4(111.0));
//            }
        }
    }
}