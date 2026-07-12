#include "/base/Configs.glsl"
#include "/Base.glsl"
#include "/base/TextOptions.glsl"
#include "/util/Dither.glsl"
#include "/util/Rand.glsl"
#if SETTING_RENDER_SCALE < 10
#ifdef SETTING_FSR1
#include "/techniques/ffx/fsr1/ffx_fsr1_easu.glsl"
#endif
#endif

layout(location = 0) out vec4 rt_out;

void main() {
    ivec2 texelPos = ivec2(gl_FragCoord.xy);
    float ditherNoise = rand_IGN(texelPos, frameCounter);
    #if SETTING_RENDER_SCALE < 10
    #ifdef SETTING_FSR1
    rt_out = vec4(fsr1_easu(texelPos), 1.0);
    #else
    vec2 outputUV = (vec2(texelPos) + 0.5) * uval_viewImageSizeRcp;
    rt_out = texture(usam_main, outputUV);
    #endif
    #else
    rt_out = texelFetch(usam_main, texelPos, 0);
    #endif
    rt_out = dither_u8(rt_out, ditherNoise);
}
