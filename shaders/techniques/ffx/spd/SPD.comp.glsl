#include "/Base.glsl"
#include "../ffx_core.glsl"

// -------------------------------------------------------- API --------------------------------------------------------
#if SPD_CHANNELS == 1
#define SPD_DATA_TYPE float
#define SPD_DATA_CAST_TO_4(x) vec4(x, 0.0, 0.0, 0.0)
#define SPD_DATA_CAST_FROM_4(v) v.x
#elif SPD_CHANNELS == 2
#define SPD_DATA_TYPE vec2
#define SPD_DATA_CAST_TO_4(x) vec4(x, 0.0, 0.0)
#define SPD_DATA_CAST_FROM_4(v) v.xy
#elif SPD_CHANNELS == 3
#define SPD_DATA_TYPE vec3
#define SPD_DATA_CAST_TO_4(x) vec4(x, 0.0)
#define SPD_DATA_CAST_FROM_4(v) v.xyz
#elif SPD_CHANNELS == 4
#define SPD_DATA_TYPE vec4
#define SPD_DATA_CAST_TO_4(x) x
#define SPD_DATA_CAST_FROM_4(v) v
#endif

// Requirements:
// #define SPD_CHANNELS <1, 2, 3, 4> - Number of channels to process
// #define SPD_HALF <0, 1> - Use half precision (1) or full precision
// #define SPD_OP <0, 1, 2> - Downsample operation: 0 for min, 1 for max, 2 for average, 3 for sum, or leave undefined for custom
//
// and following functions:
SPD_DATA_TYPE spd_loadInput(ivec2 texelPos, uint slice);
SPD_DATA_TYPE spd_loadOutput(ivec2 texelPos, uint level, uint slice);
void spd_storeOutput(ivec2 texelPos, uint level, uint slice, SPD_DATA_TYPE value);
uint spd_mipCount();
void spd_init();
vec4 spd_reduce4(vec4 v0, vec4 v1, vec4 v2, vec4 v3);

#ifdef SPD_OP
vec4 spd_reduce4(vec4 v0, vec4 v1, vec4 v2, vec4 v3) {
    #if SPD_OP == 0
    return ffxMin(ffxMin(v0, v1), ffxMin(v2, v3));
    #elif SPD_OP == 1
    return ffxMax(ffxMax(v0, v1), ffxMax(v2, v3));
    #elif SPD_OP == 2
    return (v0 + v1 + v2 + v3) * 0.25;
    #else
    return v0 + v1 + v2 + v3;
    #endif
}
#endif


// --------------------------------------------------- Adaptor stuff ---------------------------------------------------
shared uint shared_counter;

void SpdIncreaseAtomicCounter(uint slice) {
    shared_counter = atomicAdd(global_atomicCounters[slice], 1u);
}

uint SpdGetAtomicCounter() {
    return shared_counter;
}

void SpdResetAtomicCounter(uint slice) {
    global_atomicCounters[slice] = 0u;
}

#if SPD_HALF

#else
shared SPD_DATA_TYPE shared_intermediate[16][16];

vec4 SpdLoadSourceImage(ivec2 tex, uint slice) {
    return SPD_DATA_CAST_TO_4(spd_loadInput(tex, slice));
}

vec4 SpdLoad(ivec2 tex, uint slice) {
    return SPD_DATA_CAST_TO_4(spd_loadOutput(tex, 6, slice));
}

void SpdStore(FfxInt32x2 pix, vec4 outValue, uint mip, uint slice) {
    spd_storeOutput(pix, mip + 1, slice, SPD_DATA_CAST_FROM_4(outValue));
}

vec4 SpdLoadIntermediate(uint x, uint y) {
    return SPD_DATA_CAST_TO_4(shared_intermediate[x][y]);
}

void SpdStoreIntermediate(uint x, uint y, vec4 value) {
    shared_intermediate[x][y] = SPD_DATA_CAST_FROM_4(value);
}

vec4 SpdReduce4(vec4 v0, vec4 v1, vec4 v2, vec4 v3) {
    return spd_reduce4(v0, v1, v2, v3);
}
#endif

#include "ffx_spd.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    spd_init();
    uint totalNumWorkGroups = gl_NumWorkGroups.x * gl_NumWorkGroups.y;
    #if SPD_HALF
    SpdDownsampleH(gl_WorkGroupID.xy, gl_LocalInvocationIndex, spd_mipCount(), totalNumWorkGroups, gl_WorkGroupID.z);
    #else
    SpdDownsample(gl_WorkGroupID.xy, gl_LocalInvocationIndex, spd_mipCount(), totalNumWorkGroups, gl_WorkGroupID.z);
    #endif
}