#version 460 compatibility
#define COMP 1

#include "/pass/composite/VoxyMerge.glsl"

layout (local_size_x = 8, local_size_y = 8) in;

void main() {
    voxy_merge();
}
