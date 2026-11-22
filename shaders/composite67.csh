#version 460 compatibility
#define COMP 1
/*const*/
#define BLOOM_PASS 1
#define BLOOM_FINAL_RECONSTRUCTION
#define BLOOM_RENDER_SCALE 1.0
/*const*/

#include "/techniques/Bloom.comp.glsl"
