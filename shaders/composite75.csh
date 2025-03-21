#version 460 compatibility

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#include "/util/FullScreenComp.glsl"
#include "/general/DebugOutput.glsl"

layout(rgba16f) restrict uniform image2D uimg_main;

void main() {
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);
        debugOutput(outputColor);
        imageStore(uimg_main, texelPos, outputColor);
    }
}