#ifndef INCLUDE_techniques_SST2_glsl
#define INCLUDE_techniques_SST2_glsl a
/*
    References:
        [Ulu14] Uludag, Yasin. "Hi-Z Screen-Space Cone-Traced Reflectionss".
            GPU Pro 5. 2014.
        [BEU20] Beug, Anthony Paul. "Screen Space Reflection Techniques"
            University of Regina Master Thesis. 2020.
            https://ourspace.uregina.ca/bitstream/handle/10294/9245/Beug_Anthony_MSC_CS_Spring2020.pdf
        [Lee21] Lee, Sugu. "Screen Space Reflections : Implementation and optimization – Part 2 : HI-Z Tracing Method"
            Computer Graphics for Games. 2021.
            https://sugulee.wordpress.com/2021/01/19/screen-space-reflections-implementation-and-optimization-part-2-hi-z-tracing-method/
        [SUP20] Supnik, Benjamin. "A Tip for HiZ SSR - Parametric 't' Tracing".
            The Hacks of Life. 2020.
            https://hacksoflife.blogspot.com/2020/10/a-tip-for-hiz-ssr-parametric-t-tracing.html#:~:text=HiZ%20tracing%20for%20screen%20space,not%20possibly%20be%20an%20occluder.
        [GRE17] Grenier, Jean-Philippe. "Notes On Screen Space HIZ Tracing".
            jpg's blog.
            https://www.jpgrenier.org/ssr.html

    Other Credits:
        - GeforceLegend (https://github.com/GeForceLegend) - Helping with optimizations.
*/
#ifdef SST_DEBUG_PASS
#define GLOBAL_DATA_MODIFIER buffer
layout(std430, binding = 3) buffer TestBuffer {
    vec4 ssbo_testBuffer[];
};
#endif

#include "/util/Coords.glsl"
#include "/util/Morton.glsl"
#include "/util/BitPacking.glsl"
#include "/util/NZPacking.glsl"

const int START_LEVEL = 1;
const int STOP_LEVEL = 0;

shared ivec2 shared_mipmapTilesOffsets[16];
shared uint shared_maxMipLevel;
shared vec4 shared_cellCounts[16];
shared float shared_thicknessFactor;

struct SSTRay {
    ivec2 pRayOriginTexelPos;
    vec3 pRayStart;
    vec3 pRayDir;
    float pRayVecLen;
    float currT;
    int currLevel;
};

SSTRay sstray_init() {
    SSTRay ray;
    ray.pRayOriginTexelPos = ivec2(0x3FFF);
    ray.pRayStart = vec3(0.0);
    ray.pRayDir = vec3(0.0);
    ray.pRayVecLen = 0.0;
    ray.currT = 0.0;
    ray.currLevel = START_LEVEL;
    return ray;
}

uvec4 sstray_pack(SSTRay ray) {
    uvec4 packedData;
    packedData.x = nzpacking_packNormalOct32(ray.pRayDir);

    uvec2 texelPosAnd = uvec2(ray.pRayOriginTexelPos) & uvec2(0x3FFF); // 14 bits
    packedData.y = texelPosAnd.x;
    packedData.y = bitfieldInsert(packedData.y, texelPosAnd.y, 14, 14);
    packedData.y = bitfieldInsert(packedData.y, uint(ray.currLevel & 0xF), 28, 4); // 4 bits

    packedData.z = floatBitsToUint(ray.pRayVecLen);
    packedData.w = floatBitsToUint(ray.currT);
    return packedData;
}

uvec4 sstray_packedUpdate(uvec4 packedData, float newT, int newLevel) {
    packedData.y = bitfieldInsert(packedData.y, uint(newLevel & 0xF), 28, 4); // 4 bits
    packedData.w = floatBitsToUint(newT);
    return packedData;
}

SSTRay sstray_unpack(uvec4 packedData) {
    SSTRay ray = sstray_init();
    ray.pRayDir = nzpacking_unpackNormalOct32(packedData.x);

    uvec2 texelPosAnd;
    texelPosAnd.x = bitfieldExtract(packedData.y, 0, 14);
    texelPosAnd.y = bitfieldExtract(packedData.y, 14, 14);
    ray.pRayOriginTexelPos = ivec2(texelPosAnd);
    ray.currLevel = int(bitfieldExtract(packedData.y, 28, 4));

    ray.pRayVecLen = uintBitsToFloat(packedData.z);
    ray.currT = uintBitsToFloat(packedData.w);
    return ray;
}

void sstray_recoverOrigin(inout SSTRay ray, float viewZ) {
    vec2 screenPosXY = coords_texelToUV(ray.pRayOriginTexelPos, uval_mainImageSizeRcp);
    ray.pRayStart = vec3(screenPosXY, coords_viewZToReversedZ(viewZ, nearPlane));
}

SSTRay sstray_setup(ivec2 texelPos, vec3 originView, vec3 rayDirView){
    vec4 originClip = global_camProj * vec4(originView, 1.0);
    vec3 originNDC = originClip.xyz / originClip.w;
    vec3 originScreen = originNDC;
    originScreen.xy = originScreen.xy * 0.5 + 0.5;// Not applying this to Z because we are using Reverse Z

    float maxViewT = 1.0;
    maxViewT = rayDirView.z > 0.0 ? min((-nearPlane - originView.z) / rayDirView.z, maxViewT) : maxViewT;

    vec4 rayDirTempClip = global_camProj * vec4(originView + rayDirView * maxViewT, 1.0);
    vec3 rayDirTempNDC = rayDirTempClip.xyz / rayDirTempClip.w;
    vec3 rayDirTempScreen = rayDirTempNDC;
    rayDirTempScreen.xy = rayDirTempScreen.xy * 0.5 + 0.5;

    vec3 rayDirScreen = normalize(rayDirTempScreen - originScreen);
    vec3 rcpRayDirScreen = rcp(rayDirScreen);

    float maxT = 10000.0;
    maxT = rayDirScreen.z != 0.0f ? min((float(rayDirScreen.z > 0.0f) - originScreen.z) * rcpRayDirScreen.z, maxT) : maxT;
    maxT = rayDirScreen.x != 0.0f ? min((float(rayDirScreen.x > 0.0f) - originScreen.x) * rcpRayDirScreen.x, maxT) : maxT;
    maxT = rayDirScreen.y != 0.0f ? min((float(rayDirScreen.y > 0.0f) - originScreen.y) * rcpRayDirScreen.y, maxT) : maxT;

    SSTRay ray = sstray_init();
    ray.pRayOriginTexelPos = texelPos;
//    originScreen.xy = coords_texelToUV(ray.pRayOriginTexelPos, uval_mainImageSizeRcp);
    ray.pRayStart = originScreen;
    ray.pRayDir = rayDirScreen;
    ray.pRayVecLen = maxT;
    // Initial cell boundary crossing
    {
        vec3 pRayStart = ray.pRayStart;
        vec3 pRayVector = ray.pRayDir * ray.pRayVecLen;
        vec2 crossStepScaled = sign(pRayVector.xy) * uval_mainImageSizeRcp;
        vec3 invD = rcp(pRayVector);

        // Precompute cell boundary intersection constants
        bvec2 rayDirPositive = greaterThan(pRayVector.xy, vec2(0.0));
        bvec2 rayDirZero = equal(pRayVector.xy, vec2(0.0));
        vec2 invDxy = mix(invD.xy, vec2(0.0), rayDirZero);
        vec2 vec0Fix = fma(-pRayStart.xy, invDxy, vec2(rayDirZero) * 114514.0);
        vec2 cellIdOffsetBase = mix(vec2(-0.01), vec2(1.01), rayDirPositive) * invDxy;

        vec3 currScreenPos = pRayStart;
        vec4 cellCountData = shared_cellCounts[0];
        vec2 cellCount = cellCountData.xy;
        vec2 invCellCount = cellCountData.zw;
        vec2 cellIdx = floor(fma(currScreenPos.xy, cellCount, crossStepScaled * cellCount));

        vec2 cellIdInvD = invDxy * invCellCount;
        vec2 cellIdOffsetPlusFix = fma(cellIdOffsetBase, invCellCount, vec0Fix);
        vec2 tVals = fma(cellIdx, cellIdInvD, cellIdOffsetPlusFix);
        ray.currT = max(min(tVals.x, tVals.y), 0.0);
    }
    return ray;
}

struct SSTResult {
    bool hit;
    vec3 hitScreenPos;
    vec3 lastMissScreenPos;
    vec4 test;
};

SSTResult sst_initResult() {
    SSTResult result;
    result.hit = false;
    result.hitScreenPos = vec3(-1.0);
    result.lastMissScreenPos = vec3(-1.0);
    result.test = vec4(-111.0);
    return result;
}

void sst_init(float thicknessScale) {
    if (gl_LocalInvocationIndex < 16u) {
        uint maxMip = min(findMSB(min(uval_mainImageSizeI.x, uval_mainImageSizeI.y)), 16u);
        if (gl_LocalInvocationIndex == 0u) {
            shared_maxMipLevel = maxMip;
            shared_thicknessFactor = rcp(1.0 - thicknessScale);
        }
        uint mipLevel = min(gl_LocalInvocationIndex, maxMip);
        shared_mipmapTilesOffsets[mipLevel] = global_hizTiles[mipLevel].xy;

        int mipLevelI = int(mipLevel);
        vec4 mainImageSizeParams = vec4(uval_mainImageSize, uval_mainImageSizeRcp);
        vec2 cellCount = ldexp(mainImageSizeParams.xy, ivec2(-mipLevelI));
        vec2 invCellCount = ldexp(mainImageSizeParams.zw, ivec2(mipLevelI));
        shared_cellCounts[mipLevel] = vec4(cellCount, invCellCount);
    }
    barrier();
}

void sst_trace(inout SSTRay ray, uint hiZSteps) {
    vec3 pRayStart = ray.pRayStart;
    vec3 pRayVector = ray.pRayDir * ray.pRayVecLen;

    float maxThicknessFactor = shared_thicknessFactor;
    const float NEAR_Z_THICKNESS_CLAMP = 0.05;

    #ifdef SST_DEBUG_PASS
    const uvec2 DEBUG_COORD = uvec2(970, 760);
    #endif

    vec3 invD = rcp(pRayVector);

    // Precompute cell boundary intersection constants
    bvec2 rayDirPositive = greaterThan(pRayVector.xy, vec2(0.0));
    bvec2 rayDirZero = equal(pRayVector.xy, vec2(0.0));
    vec2 invDxy = mix(invD.xy, vec2(0.0), rayDirZero);
    vec2 vec0Fix = fma(-pRayStart.xy, invDxy, vec2(rayDirZero) * 114514.0);
    vec2 cellIdOffsetBase = mix(vec2(-0.01), vec2(1.01), rayDirPositive) * invDxy;

    float invDz = invD.z;
    float negRayStartZxInvDz = -pRayStart.z * invDz;

    bool isBackwardRay = pRayVector.z > 0.0;

    float currT = ray.currT;

    int maxMip = int(shared_maxMipLevel);
    int currLevel = ray.currLevel;

    for (uint i = 0u; i < hiZSteps; i++) {
        vec4 cellCountData = shared_cellCounts[currLevel];
        vec2 cellCount = cellCountData.xy;
        vec2 invCellCount = cellCountData.zw;

        // Precompute intersection constants for this mip level
        vec2 cellIdInvD = invDxy * invCellCount;
        vec2 cellIdOffsetPlusFix = fma(cellIdOffsetBase, invCellCount, vec0Fix);

        // Position calculation using FMA
        vec3 currScreenPos = fma(pRayVector, vec3(currT), pRayStart);

        // Compute cell index
        vec2 currPosScaled = currScreenPos.xy * cellCount;
        vec2 oldCellIdx = floor(currPosScaled);

        vec2 cellZ = texelFetch(usam_hiz, shared_mipmapTilesOffsets[currLevel] + ivec2(oldCellIdx), 0).rg;

        float maxZThicknessFactor = min(NEAR_Z_THICKNESS_CLAMP, currScreenPos.z) * maxThicknessFactor;

        // Select depth based on ray direction
        float cellDepth = isBackwardRay ? cellZ.y : cellZ.x;
        float depthT = fma(cellDepth, invDz, negRayStartZxInvDz);

        // Precompute cell boundary T
        vec2 tVals = fma(oldCellIdx, cellIdInvD, cellIdOffsetPlusFix);
        float cellBoundaryT = min(tVals.x, tVals.y);

        // Check if depthT crosses cell boundary
        vec2 depthCellIdx = floor(fma(pRayVector.xy, vec2(depthT), pRayStart.xy) * cellCount);
        bool crossesCell = depthT > currT && any(notEqual(oldCellIdx, depthCellIdx));

        #ifdef SST_DEBUG_PASS
        #ifdef SETTING_DEBUG_SST_STEPS
        if (gl_GlobalInvocationID.xy == DEBUG_COORD) {
            ssbo_testBuffer[i + 2048] = vec4(currT, currScreenPos.z, cellZ.x, cellZ.y);
            ssbo_testBuffer[i] = vec4(currScreenPos.xy, 0.0, float(currLevel));
            atomicMax(global_atomicCounters[15], i);
        }
        #endif
        #endif

        currLevel--;

        if (isBackwardRay) {
            bool missedCell = cellZ.y < currScreenPos.z || cellZ.y > maxZThicknessFactor;
            if (missedCell) {
                if (crossesCell) {
                    currT = min(cellBoundaryT, depthT);
                    currLevel += 2;
                } else if (currScreenPos.z <= cellZ.x) {
                    currT = max(currT, depthT);
                } else {
                    currT = cellBoundaryT;
                    currLevel += 2;
                }
            }
        } else {
            if (crossesCell) {
                currT = min(cellBoundaryT, depthT);
                currLevel += 2;
            } else if (maxZThicknessFactor >= cellZ.y) {
                currT = max(depthT, currT);
            } else {
                currT = cellBoundaryT;
                currLevel += 2;
            }
        }

        currLevel = min(currLevel, maxMip);

        if (currT >= 1.0 || currLevel < STOP_LEVEL) {
            currT = -currT;
            break;
        }
    }

    ray.currT = currT;
    ray.currLevel = currLevel;
}

uvec2 _sst2_convertAngleIndex(SSTRay ray) {
    float rayAngle1 = atan(ray.pRayDir.y, ray.pRayDir.x); // -PI to PI
//    float rayAngle2 = asin(ray.pRayDir.z) * PI; // -PI/2 to PI/2, scaled to -PI to PI
    float rayAngle2 = ray.pRayDir.z * PI; // -PI to PI, doesn't seem to lose much accuracy and saves XU

    vec2 angleNormF = linearStep(-PI, PI, vec2(rayAngle1, rayAngle2)); // 0.0 to 1.0
    uvec2 angleNormU = uvec2(angleNormF * 255.0 + 0.5); // 0 to 255
    return angleNormU;
}

uint sst2_encodeBinLocalIndex(ivec2 binLocalPos) {
    return morton_16bEncode(uvec2(binLocalPos));
}

uint _sst2_encodeRayIndexBits(uint binLocalPos, uvec2 angleIndex) {
    uint finalIndex = 0u;
    finalIndex = bitfieldInsert(finalIndex, binLocalPos, 0, 10); // 10 bits (0-1023)
    finalIndex = bitfieldInsert(finalIndex, angleIndex.x, 16, 8); // 8 bits (0-255)
    finalIndex = bitfieldInsert(finalIndex, angleIndex.y, 24, 8); // 8 bits (0-255)
    return finalIndex;
}

uint sst2_encodeRayIndexBits(uint clusterLocalIndex, SSTRay ray) {
    uvec2 angleIndex = _sst2_convertAngleIndex(ray);
    return _sst2_encodeRayIndexBits(clusterLocalIndex, angleIndex);
}

uint sst2_decodeBinLocalIndex(uint rayIndexBits) {
    return bitfieldExtract(rayIndexBits, 0, 10);
}
#endif