#version 460 compatibility

#define GBUFFER_PASS_ALPHA_TEST a
#define GBUFFER_PASS_NO_LIGHTING a
#include "/pass/geometry/GBufferSolid.frag.glsl"