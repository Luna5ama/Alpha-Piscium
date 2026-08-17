#include "/base/Configs.glsl"
#include "/Base.glsl"
#include "/base/TextOptions.glsl"
#include "/util/Dither.glsl"
#include "/util/Rand.glsl"
layout(location = 0) out vec4 rt_out;

void main() {
    ivec2 texelPos = ivec2(gl_FragCoord.xy);
    float ditherNoise = rand_IGN(texelPos, frameCounter);
    #if !SUPER_RESOLUTION_ACTIVE && !INTERNAL_FSR3_ACTIVE && RENDER_SCALE_ACTIVE
    vec2 outputUV = (vec2(texelPos) + 0.5) * uval_viewImageSizeRcp;
    rt_out = texture(usam_main, outputUV);
    #else
    rt_out = texelFetch(usam_main, texelPos, 0);
    #endif
    rt_out = dither_u8(rt_out, ditherNoise);
}
