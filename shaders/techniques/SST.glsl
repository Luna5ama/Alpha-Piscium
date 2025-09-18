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
*/

#include "HiZ.glsl"
#include "/util/Coords.glsl"

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

float intersectCellBoundary(vec3 o, vec3 d, vec3 invD, vec2 cellIdOffset, vec2 cond2, vec2 cellId, vec2 invCellCount) {
    vec2 cellMin = floor(cellId);
//    vec2 cellMax = cellMin + 1.02;
//    cellMin *= invCellCount;
//    cellMax *= invCellCount;

    vec2 t;
    //    t.x = d.x > 0.0 ? (cellMax.x - o.x) / d.x : (cellMin.x - o.x) / d.x;
    //    t.y = d.y > 0.0 ? (cellMax.y - o.y) / d.y : (cellMin.y - o.y) / d.y;
    vec2 a = cellMin + cellIdOffset;
    a *= invCellCount;
    t = (a - o.xy) * invD.xy + cond2;

    float tEdge = min(t.x, t.y);
    return tEdge;
}

float _sst_reverseZLinearDistance(float a, float b) {
    const float c = near;
    // (-c / a) - (-c / b)
    return ((a - b) * c) / (a * b);
}

shared ivec4 shared_mipmapTiles[16];
shared vec4 shared_cellCounts[16];

void sst_init() {
    if (gl_LocalInvocationIndex < 16u) {
        ivec4 temp = global_mipmapTiles[0][gl_LocalInvocationIndex];
        temp.zw -= 1;
        shared_mipmapTiles[gl_LocalInvocationIndex] = temp;

        int level = int(gl_LocalInvocationIndex);
        vec4 mainImageSizeParams = vec4(global_mainImageSize, global_mainImageSizeRcp);
        vec2 cellCount = ldexp(mainImageSizeParams.xy, ivec2(-level));
        vec2 invCellCount = ldexp(mainImageSizeParams.zw, ivec2(level));
        shared_cellCounts[gl_LocalInvocationIndex] = vec4(cellCount, invCellCount);
    }
    barrier();
}

SSTResult sst_trace(vec3 originView, vec3 rayDirView) {
    vec4 originClip = global_camProj * vec4(originView, 1.0);
    vec3 originNDC = originClip.xyz / originClip.w;
    vec3 originScreen = originNDC;
    originScreen.xy = originScreen.xy * 0.5 + 0.5;// Not applying this to Z because we are using Reverse Z

    vec4 rayDirTempClip = global_camProj * vec4(originView + rayDirView, 1.0);
    vec3 rayDirTempNDC = rayDirTempClip.xyz / rayDirTempClip.w;
    vec3 rayDirTempScreen = rayDirTempNDC;
    rayDirTempScreen.xy = rayDirTempScreen.xy * 0.5 + 0.5;

    vec3 rayDirScreen = normalize(rayDirTempScreen - originScreen);
    vec3 rcpRayDirScreen = rcp(rayDirScreen);

    SSTResult result = sst_initResult();

    float maxT = 10000.0;
    maxT = rayDirScreen.z != 0.0f ? min((float(rayDirScreen.z > 0.0f) - originScreen.z) * rcpRayDirScreen.z, maxT) : maxT;
    maxT = rayDirScreen.x != 0.0f ? min((float(rayDirScreen.x > 0.0f) - originScreen.x) * rcpRayDirScreen.x, maxT) : maxT;
    maxT = rayDirScreen.y != 0.0f ? min((float(rayDirScreen.y > 0.0f) - originScreen.y) * rcpRayDirScreen.y, maxT) : maxT;

    // Original paper uses depth as t, so t = 0.0 where ray point is at z = 0.0 and wise versa.
    // This causes precision issues with Reverse Z
    // So we instead use t = 0.0 for ray start point and t = 1.0 for the ray end point as pointed out in [SUP20]
    vec3 pRayStart = originScreen;
    vec3 pRayVector = rayDirScreen * maxT;

    bool isBackwardRay = pRayVector.z > 0.0;
    float minZ = pRayStart.z;
    float maxZ = pRayStart.z + pRayVector.z;
    float deltaZ = maxZ - minZ;
    const float MAX_THICKNESS = 0.01;
    const float MAX_THICKNESS_FACTOR = 1.0101010101; // 1.0 / (1.0 - MAX_THICKNESS)
    int maxLevels = findMSB(min(global_mainImageSizeI.x, global_mainImageSizeI.y));

    #define START_LEVEL 1
    #define STOP_LEVEL 0

    int level = START_LEVEL;
    float currT = 0.0;
//    const uvec2 DEBUG_COORD = uvec2(1350, 510);
//    const uvec2 DEBUG_COORD = uvec2(487, 250);
    const uint HI_Z_STEPS = 128;

    vec4 mainImageSizeParams = vec4(global_mainImageSize, global_mainImageSizeRcp);
    vec3 invD = rcp(pRayVector);
    bvec2 intersectCond1 = greaterThan(pRayVector.xy, vec2(0.0));
    vec2 cellIdOffset = mix(vec2(-0.01), vec2(1.01), intersectCond1);
    bvec2 vec0Cond = equal(pRayVector.xy, vec2(0.0));
    invD.xy = mix(invD.xy, vec2(0.0), vec0Cond);
    vec2 vec0Fix = vec2(vec0Cond) * 114514.0;

    vec2 crossStep = sign(pRayVector.xy) * mainImageSizeParams.zw;

    {
        vec3 currScreenPos = pRayStart + pRayVector * currT;
        vec2 currTexelPos = currScreenPos.xy * mainImageSizeParams.xy;
        ivec4 newMipTile = shared_mipmapTiles[level];
        vec4 cellCountData = shared_cellCounts[level];
        vec2 cellCount = cellCountData.xy;
        vec2 invCellCount = cellCountData.zw;
        vec2 cellIdx = (currScreenPos.xy + crossStep) * cellCount;
        currT = max(intersectCellBoundary(pRayStart, pRayVector, invD, cellIdOffset, vec0Fix, cellIdx, invCellCount), currT);
    }

    result.hit = false;

    for (uint i = 0; i < HI_Z_STEPS; i++) {
        ivec4 mipTile = shared_mipmapTiles[level];
        vec3 currScreenPos = pRayStart + pRayVector * currT;
        result.lastMissScreenPos = currScreenPos;

        vec4 cellCountData = shared_cellCounts[level];
        vec2 cellCount = cellCountData.xy;
        vec2 invCellCount = cellCountData.zw;
        vec2 oldCellIdx = saturate(currScreenPos.xy) * cellCount;

        ivec2 oldCellIdxI = ivec2(oldCellIdx);
        ivec2 readPos = mipTile.xy + min(oldCellIdxI, mipTile.zw);
        float cellMinZ = texelFetch(usam_hiz, readPos, 0).r;

        float newT = currT;
        if (isBackwardRay) {
            // float linearCurr = coords_reversedZToViewZ(currScreenPos.z, near);
            // float linearDepth = coords_reversedZToViewZ(cellMinZ, near);
            // float diff = (linearDepth - linearCurr);
            // float thickness = level > STOP_LEVEL ? 1145141919810.0 : MAX_THICKNESS * abs(linearCurr);
            // diff >= thickness ...
            // simplyfied to:
            float thicknessFactor = level > STOP_LEVEL ? 1145141919810.0 : currScreenPos.z * MAX_THICKNESS_FACTOR;
            if (any(lessThanEqual(vec2(cellMinZ, thicknessFactor), vec2(currScreenPos.z, cellMinZ)))) {
                newT = intersectCellBoundary(pRayStart, pRayVector, invD, cellIdOffset, vec0Fix, oldCellIdx, invCellCount);
                newT = max(newT, currT);
                level = min(maxLevels, level + 2);
//                v = 1.0;
            }
        } else {
            float depthT = (cellMinZ - pRayStart.z) * invD.z;
            vec3 depthRayPos = pRayStart + pRayVector * depthT;
            vec2 depthRayCellIndex = depthRayPos.xy * cellCount;
            uint cond = uint(depthT > currT) & uint(any(notEqual(oldCellIdxI, ivec2(depthRayCellIndex))));
            if (bool(cond)) {
                newT = min(intersectCellBoundary(pRayStart, pRayVector, invD, cellIdOffset, vec0Fix, oldCellIdx, invCellCount), depthT);
                level = min(maxLevels, level + 2);
//                v = 1.0;
            } else {
                float thicknessFactor = level > STOP_LEVEL ? 1145141919810.0 : currScreenPos.z * MAX_THICKNESS_FACTOR;
                if (thicknessFactor >= cellMinZ) {
                    newT = depthT;
                    newT = max(newT, currT);
                } else {
                    level++;
                }
            }
        }
        currT = newT;
        level--;
        if (level < STOP_LEVEL) {
            result.hitScreenPos = pRayStart + pRayVector * currT;
            result.hit = true;
            break;
        }

        if (currT > 1.0) {
            break;
        }

//        if (gl_GlobalInvocationID.xy == DEBUG_COORD) {
//            testBuffer[i] = vec4(currScreenPos.xy, floor(oldCellIdx));
//            testBuffer[i + 2048] = vec4(float(l), 0.0, 0.0, 0.0);
//        }
    }

    if (!result.hit) {
        level = 0;
        ivec2 mipTile = shared_mipmapTiles[level].zw - 1;

        #define FIX_STEPS 4

        float stepRcp = rcp(float(FIX_STEPS));

        float minT = currT - 0.01;
        float maxT = 1.0;
        float deltaT = maxT - minT;

        for (uint i = 0; i < FIX_STEPS; i++) {
            float t = minT + ((float(i) + 1.0) * stepRcp * deltaT);
            vec3 screenPos = pRayStart + pRayVector * t;
            vec2 texelPos = saturate(screenPos.xy) * mainImageSizeParams.xy;
            vec2 cellIdx = texelPos;
            ivec2 readPos = ivec2(cellIdx);
            readPos = min(readPos, mipTile);
            float cellMinZ = texelFetch(usam_hiz, readPos, 0).r;

            if (cellMinZ > screenPos.z) {
                float linearCurr = coords_reversedZToViewZ(screenPos.z, near);
                float linearDepth = coords_reversedZToViewZ(cellMinZ, near);
                float diff = linearDepth - linearCurr;
                float thickness = MAX_THICKNESS * abs(linearCurr) * 0.01;
                uint cond = uint(diff < thickness) | uint(linearDepth < -65000.0);
                if (bool(cond)) {
                    result.hitScreenPos = screenPos;
                    result.hit = true;
                    return result;
                } else {
                    return result;
                }
            }

//            if (gl_GlobalInvocationID.xy == DEBUG_COORD) {
//                testBuffer[i + HI_Z_STEPS] = vec4(screenPos.xy, floor(cellIdx));
//                testBuffer[i + 2048 + HI_Z_STEPS] = vec4(float(level), 0.0, 0.0, 0.0);
//            }
        }
    }

    return result;
}