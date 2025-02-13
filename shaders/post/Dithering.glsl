#include "/util/Rand.glsl"
#include "/util/Math.glsl"

void dithering(inout vec4 outputColor) {
    float noiseIGN = rand_IGN(gl_GlobalInvocationID.xy, frameCounter);
    outputColor = dither(outputColor, noiseIGN, 255.0);
}