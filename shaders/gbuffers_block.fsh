#version 460 compatibility

#define GBUFFER_PASS_ALPHA_TEST a
#define GBUFFER_PASS_MATERIAL_ID a
#define GBUFFER_PASS_TEXTURED a
#ifndef IRIS_FEATURE_ENTITY_TRANSLUCENT
#define GBUFFER_PASS_TRANLUCENT a
#endif
#include "gbuffer/GBufferPass.frag"