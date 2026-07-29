#version 460 compatibility
#define COMP 1
/*const*/
#define MATERIAL_DEPTH_MIP_LEVEL 8
#define MATERIAL_DEPTH_MIP_WORK_GROUPS 4
/*const*/

#include "/techniques/parallax/MaterialDepthMip.comp.glsl"
