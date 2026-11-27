#version 460 compatibility
#define COMP 1
/*const*/
#define BLOOM_DOWNSAMPLE
#define BLOOM_PASS 4
#define BLOOM_RENDER_SCALE 0.0625
/*const*/

#include "/techniques/Bloom.comp.glsl"
