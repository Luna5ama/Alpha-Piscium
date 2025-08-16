#version 460 compatibility

#include "/techniques/DOF.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);


layout(rgba16f) uniform restrict image2D uimg_main;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, global_mainImageSizeI))) {
        vec4 outputColor = imageLoad(uimg_main, texelPos);
        outputColor.rgb = dof_sample(usam_temp1, texelPos);
        imageStore(uimg_main, texelPos, outputColor);
    }
}