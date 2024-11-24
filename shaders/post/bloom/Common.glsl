#include "../../_Util.glsl"

// xy: start, zw: end
ivec2 bloom_outputSize();
vec4 bloom_input(ivec2 coord);
void bloom_output(ivec2 coord, vec4 data);

ivec2 groupBasePixel = ivec2(gl_WorkGroupID.xy) << 4;

vec4 readCache(ivec2 pos) {
    uvec2 packedData = shared_dataCache[pos.y][pos.x];
    vec4 data;
    data.xy = unpackHalf2x16(packedData.x);
    data.zw = unpackHalf2x16(packedData.y);
    return data;
}

void writeCache(ivec2 pos, vec4 data) {
    uvec2 packedData;
    packedData.x = packHalf2x16(data.xy);
    packedData.y = packHalf2x16(data.zw);
    shared_dataCache[pos.y][pos.x] = packedData;
}
