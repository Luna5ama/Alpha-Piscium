#version 460 compatibility

#define GBUFFER_PASS_ALPHA_TEST a
#define GBUFFER_PASS_MATERIAL_ID_OVERRIDE 65534
#define GBUFFER_PASS_TEXTURED a
#include "gbuffer/GBufferPass.frag"