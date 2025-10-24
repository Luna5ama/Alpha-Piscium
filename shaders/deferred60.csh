#version 460 compatibility

#include "/techniques/SSGI.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Hash.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_temp1;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    sst_init();

    if (RANDOM_FRAME < MAX_FRAMES){
        vec4 ssgiOut = vec4(0.0);
        if (RANDOM_FRAME >= 0) {
            ssgiOut = imageLoad(uimg_temp1, texelPos);

            vec3 result;
            #if USE_REFERENCE
            result = ssgiRef(texelPos);
            #else
            result = texelFetch(usam_temp3, texelPos, 0).rgb;
            #endif
            ssgiOut.a += 1.0;
            ssgiOut.rgb = mix(ssgiOut.rgb, result, 1.0 / ssgiOut.a);
        }
        imageStore(uimg_temp1, texelPos, ssgiOut);
    } else {
        if (all(greaterThan(texelPos, uval_mainImageSizeI - 8))) {
            imageStore(uimg_temp1, texelPos, vec4(111.0));
        }
    }
}