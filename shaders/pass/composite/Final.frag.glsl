#include "/base/Configs.glsl"
#include "/base/TextOptions.glsl"
#include "/techniques/textile/CSRGBA16F.glsl"
#include "/util/Dither.glsl"
#include "/util/Rand.glsl"

#if SETTING_DEBUG_OUTPUT == 3
#include "/techniques/DebugOutput.glsl"
#endif

layout(location = 0) out vec4 rt_out;

void main() {
    ivec2 texelPos = ivec2(gl_FragCoord.xy);
    float ditherNoise = rand_IGN(texelPos, frameCounter);
    rt_out = texelFetch(usam_csrgba16f, csrgba16f_temp1_texelToTexel(texelPos), 0);
    rt_out = dither_u8(rt_out, ditherNoise);
}