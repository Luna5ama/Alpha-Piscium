#include "/base/Configs.glsl"
#include "/Base.glsl"
#include "/base/TextOptions.glsl"
#include "/util/Dither.glsl"
#include "/util/Rand.glsl"
layout(location = 0) out vec4 rt_out;

void main() {
    ivec2 texelPos = ivec2(gl_FragCoord.xy);
    float ditherNoise = rand_IGN(texelPos, frameCounter);
    #ifdef SETTING_FSR3
    ivec2 fsr3OutputTexel = ivec2(texelPos.x + int(uval_viewImageSize.x) * 2, texelPos.y);
    rt_out = texelFetch(usam_fsr3UpscaleAtlas, fsr3OutputTexel, 0);
    #elif SETTING_RENDER_SCALE < 10
    vec2 outputUV = (vec2(texelPos) + 0.5) * uval_viewImageSizeRcp;
    rt_out = texture(usam_main, outputUV);
    #else
    rt_out = texelFetch(usam_main, texelPos, 0);
    #endif
    rt_out = dither_u8(rt_out, ditherNoise);
    #ifdef SETTING_FSR3
    rt_out.a = 1.0;
    #endif
}