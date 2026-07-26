#version 460 compatibility
#define COMP 1
/*const*/
#define MATERIAL_DEPTH_MIP_LEVEL 3
#define MATERIAL_DEPTH_MIP_WORK_GROUPS 256
/*const*/

#include "/pass/setup/MaterialDepthMip.comp.glsl"
