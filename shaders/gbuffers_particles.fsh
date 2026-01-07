#version 460 compatibility
#define FRAG 1

#define GBUFFER_PASS_PARTICLE a
#define GBUFFER_PASS_ALPHA_TEST a
#define GBUFFER_PASS_TEXTURED a
#include "/pass/geometry/GBufferSolid.frag.glsl"