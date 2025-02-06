#version 460 compatibility

#include "util/FullScreenComp.glsl"
#include "post/Dithering.glsl"
#include "general/DebugOutput.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);
        dithering(outputColor);
        #if SETTING_DEBUG_OUTPUT == 3
        debugOutput(outputColor);
        #endif
        imageStore(uimg_main, texelPos, outputColor);
    }
}