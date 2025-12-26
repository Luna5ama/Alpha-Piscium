#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/HiZCheck.glsl"
#include "/util/GBufferData.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(rgba8) uniform restrict image2D uimg_rgba8;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
#include "/techniques/gi/AntiFireFlyRCRS.glsl"

void main() {
    if (hiz_groupGroundCheckSubgroup(gl_WorkGroupID.xy, 4)) {
        ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
        antiFireFlyRCRS(texelPos);
    }
}
