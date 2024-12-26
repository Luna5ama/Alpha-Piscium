#include "../../_Util.glsl"

// xy: start, zw: end
ivec2 bloom_outputSize();
void bloom_output(ivec2 coord, vec4 data);

ivec2 groupBasePixel = ivec2(gl_WorkGroupID.xy) << 4;
