#version 460 compatibility

#define GBUFFER_PASS_ALPHA_TEST a
#define GBUFFER_PASS_TEXTURED a
#define GBUFFER_PASS_ENTITY_COLOR a
#include "/pass/geometry/GBufferSolid.frag.glsl"