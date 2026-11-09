#version 460 compatibility
#define COMP 1
/*const*/
#define BLOOM_DOWNSAMPLE
#define BLOOM_PASS 7
#define BLOOM_RENDER_SCALE 0.0078125
/*const*/

#include "/techniques/Bloom.comp.glsl"
