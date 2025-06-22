#version 460 compatibility

#include "/Base.glsl"

layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

void main() {
    global_shadowAABBMinPrev = global_shadowAABBMin;
    global_shadowAABBMaxPrev = global_shadowAABBMax;
    vec4 shadowAABBMin = shadowModelView * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 shadowAABBMax = shadowModelView * vec4(0.0, 0.0, 0.0, 1.0);
    global_shadowAABBMin = ivec3(floor(shadowAABBMin.xyz));
    global_shadowAABBMax = ivec3(ceil(shadowAABBMax.xyz));
}