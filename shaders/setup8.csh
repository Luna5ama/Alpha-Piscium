#version 460 compatibility
#define COMP 1
/*const*/
#define MATERIAL_DEPTH_MIP_LEVEL 2
#define MATERIAL_DEPTH_MIP_WORK_GROUPS 512
/*const*/

#include "/pass/setup/MaterialDepthMip.comp.glsl"
