#version 460 compatibility
#define VERT 1

#define GBUFFER_PASS_MATERIAL_ID a
#define GBUFFER_PASS_TEXTURED a
#define GBUFFER_PASS_STEEP_PARALLAX a
#include "/pass/geometry/GBufferSolid.vert.glsl"