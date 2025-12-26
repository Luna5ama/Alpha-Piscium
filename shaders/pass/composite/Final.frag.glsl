#include "/base/Configs.glsl"
#include "/base/TextOptions.glsl"
#include "/util/Dither.glsl"
#include "/util/Rand.glsl"

layout(location = 0) out vec4 rt_out;

void main() {
    ivec2 texelPos = ivec2(gl_FragCoord.xy);
    float ditherNoise = rand_IGN(texelPos, frameCounter);
    rt_out = texelFetch(usam_temp1, texelPos, 0);
    rt_out = dither_u8(rt_out, ditherNoise);
}