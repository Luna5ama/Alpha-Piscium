#version 460 compatibility
#define COMP 1
/*const*/
#define BLOOM_DOWNSAMPLE
#define BLOOM_PASS 3
#define BLOOM_RENDER_SCALE 0.125
/*const*/

#include "/techniques/Bloom.comp.glsl"
