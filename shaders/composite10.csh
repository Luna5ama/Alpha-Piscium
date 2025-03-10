#version 460 compatibility

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

#define SSVBIL_SAMPLE_STEPS SETTING_VBGI_STEPS
#define SSVBIL_SAMPLE_SLICES SETTING_VBGI_SLICES
#include "/post/gtvbgi/GTVBGI.glsl"
#include "/util/Morton.glsl"

layout(rgba16f) uniform writeonly image2D uimg_ssvbil;

void main() {
    ivec2 texelPos2x2 = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos2x2, global_mainImageSizeI))) {
        vec4 giOut = gtvbgi(texelPos2x2);
        imageStore(uimg_ssvbil, texelPos2x2, giOut);
    }
}