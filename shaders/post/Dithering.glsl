#include "/util/Rand.glsl"
#include "/util/Math.glsl"

void dithering(vec2 texelPos, inout vec3 outputColor) {
    float noiseIGN = rand_IGN(texelPos, frameCounter);
    outputColor = dither(outputColor, noiseIGN, 255.0);
}