#version 460 compatibility
#define COMP 1
/*const*/
#define MATERIAL_DEPTH_MIP_LEVEL 5
#define MATERIAL_DEPTH_MIP_WORK_GROUPS 16
/*const*/

#include "/techniques/parallax/MaterialDepthMip.comp.glsl"
