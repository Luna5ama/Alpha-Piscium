#version 460 compatibility

#define GBUFFER_PASS_ALPHA_TEST a
#define GBUFFER_PASS_MATERIAL_ID_OVERRIDE MATERIAL_ID_HAND
#define GBUFFER_PASS_TEXTURED a
#define GBUFFER_PASS_TRANLUCENT a
#include "gbuffer/GBufferPass.frag"