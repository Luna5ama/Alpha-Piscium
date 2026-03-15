#version 460 compatibility
#define VERT 1

#define SHADOW_PASS_ALPHA_TEST a
#define SHADOW_PASS_VOXELIZE a
#include "/pass/geometry/ShadowPass.vert.glsl"