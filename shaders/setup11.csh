#version 460 compatibility
#define COMP 1
/*const*/
#define MATERIAL_DEPTH_MIP_LEVEL 5
#define MATERIAL_DEPTH_MIP_WORK_GROUPS 64
/*const*/

#include "/pass/setup/MaterialDepthMip.comp.glsl"
