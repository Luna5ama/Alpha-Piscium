#include "../_Util.glsl"

void dithering(inout vec4 outputColor) {
    float noiseIGN = rand_IGN(gl_GlobalInvocationID.xy, frameCounter);
    outputColor = dither(outputColor, noiseIGN, 255.0);
}