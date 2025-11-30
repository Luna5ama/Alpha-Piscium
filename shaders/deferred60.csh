#version 460 compatibility

#include "/techniques/SSGI.glsl"
#include "/techniques/gtvbgi/GTVBGI2.glsl"
#include "/util/GBufferData.glsl"
#include "/techniques/textile/CSRGBA32UI.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba32f) uniform restrict image2D uimg_temp1;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        if (RANDOM_FRAME < MAX_FRAMES){
            vec4 ssgiOut = vec4(0.0);
            if (RANDOM_FRAME >= 0) {

                vec4 result = vec4(0.0);

                #if USE_REFERENCE == 2
                result.xyz = gtvbgi(texelPos);
                result.w = 1.0;
                #elif USE_REFERENCE == 1
                const uint SPP = MC_SPP;
                uint baseRand = RANDOM_FRAME * SPP;
                for (uint i = 0u; i < SPP; ++i) {
                    result.xyz += ssgiRef(texelPos, baseRand + i);
                    result.w++;
                }
                #else
                result.xyz = uintBitsToFloat(imageLoad(uimg_csrgba32ui, csrgba32ui_temp4_texelToTexel(texelPos)).rgb);
                result.w = 1.0;
                #endif

                ssgiOut = imageLoad(uimg_temp1, texelPos);
                ssgiOut.a += result.w;
                ssgiOut.rgb = mix(ssgiOut.rgb, result.xyz, 1.0 / ssgiOut.a);
            }
            imageStore(uimg_temp1, texelPos, ssgiOut);
        } else {
//            if (all(greaterThan(texelPos, uval_mainImageSizeI - 8))) {
//                imageStore(uimg_temp1, texelPos, vec4(111.0));
//            }
        }
    }
}