#version 460 compatibility

#define GBUFFER_PASS_PARTICLE a
#define GBUFFER_PASS_ALPHA_TEST a
#define GBUFFER_PASS_MATERIAL_ID a
#define GBUFFER_PASS_TEXTURED a
#include "/gbuffer/GBufferSolid.frag.glsl"