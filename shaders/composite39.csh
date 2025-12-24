#version 460 compatibility

#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_main;
layout(rgba16f) uniform writeonly image2D uimg_rgba16f;
#include "/techniques/DOF.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);
        outputColor.rgb = dof_sample(texelPos);
        imageStore(uimg_main, texelPos, outputColor);
    }
}