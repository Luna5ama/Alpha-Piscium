/*
    References:
        [JIM14] Jimenez, Jorge. "Next Generation Post Processing in Call of Duty: Advanced Warfare" SIGGRAPH 2014.
            https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
*/
#include "/Base.glsl"
#include "/util/Colors.glsl"

const float BASE_BLOOM_INTENSITY = 0.01;

/*const*/
#define BLOOM_USE_KARIS_AVERAGE 1

#if BLOOM_DOWN_SAMPLE
#define BLOOM_SCALE_DIV BLOOM_PASS
#if BLOOM_USE_KARIS_AVERAGE
#if BLOOM_PASS <= 2
#define BLOOM_KARIS_AVERAGE 1
#endif
#endif

#elif BLOOM_UP_SAMPLE
#define BLOOM_SCALE_DIV (BLOOM_PASS - 1)
#endif

#ifndef BLOOM_NON_STANDALONE
#if BLOOM_SCALE_DIV == 0
const vec2 workGroupsRender = vec2(1.0, 1.0);
#elif BLOOM_SCALE_DIV == 1
const vec2 workGroupsRender = vec2(0.5, 0.5);
#elif BLOOM_SCALE_DIV == 2
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif BLOOM_SCALE_DIV == 3
const vec2 workGroupsRender = vec2(0.125, 0.125);
#elif BLOOM_SCALE_DIV == 4
const vec2 workGroupsRender = vec2(0.0625, 0.0625);
#elif BLOOM_SCALE_DIV == 5
const vec2 workGroupsRender = vec2(0.03125, 0.03125);
#elif BLOOM_SCALE_DIV == 6
const vec2 workGroupsRender = vec2(0.015625, 0.015625);
#elif BLOOM_SCALE_DIV == 7
const vec2 workGroupsRender = vec2(0.0078125, 0.0078125);
#elif BLOOM_SCALE_DIV == 8
const vec2 workGroupsRender = vec2(0.00390625, 0.00390625);
#elif BLOOM_SCALE_DIV == 9
const vec2 workGroupsRender = vec2(0.001953125, 0.001953125);
#elif BLOOM_SCALE_DIV == 10
const vec2 workGroupsRender = vec2(0.0009765625, 0.0009765625);
#endif
layout(local_size_x = 16, local_size_y = 16) in;
#endif

void bloom_init();
vec4 bloom_main(ivec2 texelPos);

vec4 _bloom_imageLoad(ivec2 coord);
void _bloom_imageStore(ivec2 coord, vec4 data);
vec4 _bloom_imageSample(vec2 uv);

#if BLOOM_DOWN_SAMPLE

#if BLOOM_PASS == 1
vec4 _bloom_imageSample(vec2 uv) {
    return texture(usam_main, uv);
}
#else
vec4 _bloom_imageSample(vec2 uv) {
    return texture(usam_temp4, uv);
}
#endif

layout(rgba16f) uniform writeonly image2D uimg_temp4;
vec4 _bloom_imageLoad(ivec2 coord) {
    return vec4(0.0);
}
void _bloom_imageStore(ivec2 coord, vec4 data) {
    imageStore(uimg_temp4, coord, data);
}


#elif BLOOM_UP_SAMPLE

vec4 _bloom_imageSample(vec2 uv) {
    return texture(usam_temp4, uv);
}
#if BLOOM_PASS == 1
vec4 _bloom_imageLoad(ivec2 coord) {
    return imageLoad(uimg_main, coord);
}
void _bloom_imageStore(ivec2 coord, vec4 data) {
    imageStore(uimg_main, coord, data);
}
#else
layout(rgba16f) uniform restrict image2D uimg_temp4;
vec4 _bloom_imageLoad(ivec2 coord) {
    return imageLoad(uimg_temp4, coord);
}
void _bloom_imageStore(ivec2 coord, vec4 data) {
    imageStore(uimg_temp4, coord, data);
}
#endif

#endif

#define BIT_MASK(x) ((1 << (x)) - 1)

#if BLOOM_DOWN_SAMPLE
ivec2 bloom_inputSize = global_mipmapSizesI[BLOOM_PASS - 1];
ivec2 bloom_outputSize = global_mipmapSizesI[BLOOM_PASS];

int inputOffset = global_mipmapSizePrefixes[max(BLOOM_PASS - 2, 0)].x - uval_mainImageSizeI.x + BLOOM_PASS - 1;
int outputOffset = global_mipmapSizePrefixes[max(BLOOM_PASS - 1, 0)].x - uval_mainImageSizeI.x + BLOOM_PASS;

#elif BLOOM_UP_SAMPLE
ivec2 bloom_inputSize = global_mipmapSizesI[BLOOM_PASS];
ivec2 bloom_outputSize = global_mipmapSizesI[BLOOM_PASS - 1];

int inputOffset = global_mipmapSizePrefixes[max(BLOOM_PASS - 1, 0)].x - uval_mainImageSizeI.x + BLOOM_PASS;
int outputOffset = global_mipmapSizePrefixes[max(BLOOM_PASS - 2, 0)].x - uval_mainImageSizeI.x + BLOOM_PASS - 1;

#endif

ivec2 inputStartPixel = ivec2(inputOffset, 1);
ivec2 inputEndPixel = inputStartPixel + bloom_inputSize;
vec2 inputStartTexel = (vec2(inputStartPixel) + 0.0) * uval_mainImageSizeRcp;
vec2 inputEndTexel = (vec2(inputEndPixel) - 0.0) * uval_mainImageSizeRcp;

#if BLOOM_DOWN_SAMPLE
vec4 bloom_readInputDown(ivec2 coord) {
    vec2 readPosUV = vec2(coord + inputStartPixel) * uval_mainImageSizeRcp;
    readPosUV = clamp(readPosUV, inputStartTexel, inputEndTexel);
    vec4 inputValue = _bloom_imageSample(readPosUV);
    #if BLOOM_PASS == 1
    float emissiveFlag = float(inputValue.a < 0.0);
    inputValue.a = abs(inputValue.a);
    inputValue.rgb *= mix(inputValue.a, saturate(inputValue.a * 0.5), emissiveFlag);
    inputValue *= BASE_BLOOM_INTENSITY;
    #endif
    return inputValue;
}

void bloom_writeOutput(ivec2 coord, vec4 data) {
    coord.y += 1;
    coord.x += outputOffset;
    _bloom_imageStore(coord, data);
}
// ------ Down Sample Pass ------
shared uvec2 shared_dataCache[35][18];

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

ivec2 groupBasePixel = ivec2(gl_WorkGroupID.xy) << 4;

#if BLOOM_KARIS_AVERAGE
void weightedSum(vec4 color, float baseWeight, inout vec4 colorSum, inout float weightSum) {
    float weight = baseWeight;
    weight *= colors_karisWeight(color.rgb / BASE_BLOOM_INTENSITY);
    colorSum += color * weight;
    weightSum += weight;
}
#else
void weightedSum(vec4 color, float baseWeight, inout vec4 colorSum, inout float weightSum) {
    float weight = baseWeight;
    colorSum += color * weight;
    weightSum += weight;
}
#endif

void computeReadPos(uint index, out ivec2 writePos, out ivec2 readPos) {
    writePos = ivec2(index % 18, index / 18);
    readPos = writePos;
    readPos.x = readPos.x << 1;
    readPos += groupBasePixel << 1;
    readPos -= 1;
    readPos.x += (writePos.y & 1);
}

void bloom_init() {
    ivec2 writePos;
    ivec2 readPos;

    computeReadPos(gl_LocalInvocationIndex, writePos, readPos);
    writeCache(writePos, max(bloom_readInputDown(readPos), 0.0));

    computeReadPos(gl_LocalInvocationIndex + 256, writePos, readPos);
    writeCache(writePos, max(bloom_readInputDown(readPos), 0.0));

    computeReadPos(gl_LocalInvocationIndex + 512, writePos, readPos);
    if (writePos.y < 35) {
        writeCache(writePos, max(bloom_readInputDown(readPos), 0.0));
    }

    barrier();
}
vec4 bloom_main(ivec2 texelPos) {
    ivec2 centerPos = ivec2(gl_LocalInvocationID.xy);
    centerPos.y = centerPos.y << 1;

    // e _ f _ g
    // _ a _ b _
    // h _ i _ j
    // _ c _ d _
    // k _ l _ m
    // a,b,c,d: 0.5
    // e,f,h,i: 0.125
    // f,g,i,j: 0.125
    // h,i,k,l: 0.125
    // i,j,l,m: 0.125
    vec4 e = readCache(centerPos);
    vec4 f = readCache(centerPos + ivec2(1, 0));
    vec4 g = readCache(centerPos + ivec2(2, 0));

    vec4 a = readCache(centerPos + ivec2(0, 1));
    vec4 b = readCache(centerPos + ivec2(1, 1));

    vec4 h = readCache(centerPos + ivec2(0, 2));
    vec4 i = readCache(centerPos + ivec2(1, 2));
    vec4 j = readCache(centerPos + ivec2(2, 2));

    vec4 c = readCache(centerPos + ivec2(0, 3));
    vec4 d = readCache(centerPos + ivec2(1, 3));

    vec4 k = readCache(centerPos + ivec2(0, 4));
    vec4 l = readCache(centerPos + ivec2(1, 4));
    vec4 m = readCache(centerPos + ivec2(2, 4));

    vec4 colorSum = vec4(0.0);
    float weightSum = 0.0;
    weightedSum((a + b + c + d) * 0.25, 0.5, colorSum, weightSum);
    weightedSum((e + f + h + i) * 0.25, 0.125, colorSum, weightSum);
    weightedSum((f + g + i + j) * 0.25, 0.125, colorSum, weightSum);
    weightedSum((h + i + k + l) * 0.25, 0.125, colorSum, weightSum);
    weightedSum((i + j + l + m) * 0.25, 0.125, colorSum, weightSum);

    return colorSum / weightSum;
}
#elif BLOOM_UP_SAMPLE
vec4 bloom_readInputUp(ivec2 coord, ivec2 offset) {
    vec2 readPosUV = vec2((vec2(coord) + offset * SETTING_BLOOM_RADIUS + 0.5) * 0.5 + inputStartPixel) * uval_mainImageSizeRcp;
    readPosUV = clamp(readPosUV, inputStartTexel, inputEndTexel);
    return _bloom_imageSample(readPosUV);
}

void bloom_writeOutput(ivec2 coord, vec4 data) {
    coord.y += 1;
    coord.x += outputOffset;
    vec4 writeData = _bloom_imageLoad(coord);
    writeData += data;
    _bloom_imageStore(coord, writeData);
}
// ------ Up Sample Pass ------
void bloom_init() { }
vec4 bloom_main(ivec2 texelPos) {
    // a b c
    // d e f
    // g h i
    // a,c,g,i: 1/16 (0.0625)
    // b,d,f,h: 2/16 (0.125)
    // e: 4/16 (0.25)
    vec4 result = vec4(0.0);

    vec4 a = bloom_readInputUp(texelPos, ivec2(-1, -1));
    vec4 c = bloom_readInputUp(texelPos, ivec2(1, -1));
    vec4 g = bloom_readInputUp(texelPos, ivec2(-1, 1));
    vec4 i = bloom_readInputUp(texelPos, ivec2(1, 1));
    result += (a + c + g + i) * 0.0625;

    vec4 b = bloom_readInputUp(texelPos, ivec2(0, -1));
    vec4 d = bloom_readInputUp(texelPos, ivec2(-1, 0));
    vec4 f = bloom_readInputUp(texelPos, ivec2(1, 0));
    vec4 h = bloom_readInputUp(texelPos, ivec2(0, 1));
    result += (b + d + f + h) * 0.125;

    vec4 e = bloom_readInputUp(texelPos, ivec2(0, 0));
    result += e * 0.25;

    return result;
}
vec4 bloom_mainOutput(ivec2 texelPos) {
    vec4 result = bloom_main(texelPos);
    #if !BLOOM_USE_KARIS_AVERAGE
    result *= 0.6;
    #endif
    result *= SETTING_BLOOM_INTENSITY;
    if (isEyeInWater == 1) {
        result *= SETTING_BLOOM_UNDERWATER_BOOST;
    }
    return result;
}
#endif

#ifndef BLOOM_NON_STANDALONE
void main() {
    bloom_init();
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(texelPos, bloom_outputSize))) {
        vec4 result = bloom_main(texelPos);
        bloom_writeOutput(texelPos, result);
    }
}
#endif
/*const*/