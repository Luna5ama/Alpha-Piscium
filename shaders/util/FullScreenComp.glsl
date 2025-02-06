#ifndef INCLUDE_util_FullScreenComp.glsl
#define INCLUDE_util_FullScreenComp.glsl

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);
ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);

#endif