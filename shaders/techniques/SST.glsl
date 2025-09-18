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

#define GLOBAL_DATA_MODIFIER readonly
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

float intersectCellBoundary(vec3 o, vec3 d, vec3 invD, vec2 cellId, vec2 invCellCount) {
    vec2 cellMin = floor(cellId) - 0.01;
    vec2 cellMax = cellMin + 1.02;
//    cellMin *= invCellCount;
//    cellMax *= invCellCount;

    vec2 t;
    //    t.x = d.x > 0.0 ? (cellMax.x - o.x) / d.x : (cellMin.x - o.x) / d.x;
    //    t.y = d.y > 0.0 ? (cellMax.y - o.y) / d.y : (cellMin.y - o.y) / d.y;
    vec2 a = mix(cellMin, cellMax, greaterThan(d.xy, vec2(0.0)));
    a *= invCellCount;
    t = mix((a - o.xy) * invD.xy, vec2(114514.0), equal(d.xy, vec2(0.0)));

    float tEdge = min(t.x, t.y);
    return tEdge;
}

bool crossedCellBoundary(vec2 cell_id_one, vec2 cell_id_two) {
    return any(notEqual(ivec2(cell_id_one), ivec2(cell_id_two)));
}

float _sst_reverseZLinearDistance(float a, float b) {
    const float c = near;
    // (-c / a) - (-c / b)
    return ((a - b) * c) / (a * b);
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

    SSTResult result = sst_initResult();

    float maxT = 10000.0;
    maxT = rayDirScreen.z != 0.0f ? min((float(rayDirScreen.z > 0.0f) - originScreen.z) / rayDirScreen.z, maxT) : maxT;
    maxT = rayDirScreen.x != 0.0f ? min((float(rayDirScreen.x > 0.0f) - originScreen.x) / rayDirScreen.x, maxT) : maxT;
    maxT = rayDirScreen.y != 0.0f ? min((float(rayDirScreen.y > 0.0f) - originScreen.y) / rayDirScreen.y, maxT) : maxT;

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
    int maxLevels = findMSB(min(global_mainImageSizeI.x, global_mainImageSizeI.y));

    #define START_LEVEL 1
    #define STOP_LEVEL 0

    int level = START_LEVEL;
    float currT = 0.0;
//    const uvec2 DEBUG_COORD = uvec2(1350, 510);
    const uvec2 DEBUG_COORD = uvec2(487, 250);
    const uint HI_Z_STEPS = 128;

    vec2 mainImageSize = global_mainImageSize;
    vec2 mainImageSizeRcp = global_mainImageSizeRcp;
    vec3 invD = rcp(pRayVector);

    vec2 crossStep = vec2(pRayVector.x >= 0.0 ? 1.0 : -1.0, pRayVector.y >= 0.0 ? 1.0 : -1.0);
    vec2 crossOffset = crossStep / mainImageSize / 1.0;

    {
        vec3 currScreenPos = pRayStart + pRayVector * currT;
        vec2 currTexelPos = currScreenPos.xy * mainImageSize;
        ivec4 newMipTile = global_mipmapTiles[0][level];
        float levelDiv = ldexp(1.0, -level);
        float levelMul = ldexp(1.0, level);
        vec2 newCellCount = mainImageSize * levelDiv;
        vec2 invCellCount = mainImageSizeRcp * levelMul;
        vec2 cellIdx = (currTexelPos + crossStep * 1.0) * levelDiv;
        currT = max(intersectCellBoundary(pRayStart, pRayVector, invD, cellIdx, invCellCount), currT);
    }

    result.hit = false;

    for (uint i = 0; i < HI_Z_STEPS; i++) {
        vec3 currScreenPos = pRayStart + pRayVector * currT;
        result.lastMissScreenPos = currScreenPos;

        ivec4 mipTile = global_mipmapTiles[0][level];

        vec2 cellCount = ldexp(mainImageSize, ivec2(-level));
        vec2 invCellCount = ldexp(mainImageSizeRcp, ivec2(level));
        vec2 oldCellIdx = currScreenPos.xy * cellCount;

        ivec2 oldCellIdxI = ivec2(oldCellIdx);
        ivec2 readPos = mipTile.xy + ivec2(clamp(oldCellIdxI, ivec2(0), mipTile.zw - 1));
        float cellMinZ = texelFetch(usam_hiz, readPos, 0).r;

        float newT = currT;
        if (isBackwardRay) {
            float linearCurr = coords_reversedZToViewZ(currScreenPos.z, near);
            float linearDepth = coords_reversedZToViewZ(cellMinZ, near);
            float diff = (linearDepth - linearCurr);
            uint cond = uint(level > STOP_LEVEL) | uint(linearDepth < -65000.0);
            float thickness = bool(cond) ? 1145141919810.0 : MAX_THICKNESS * abs(linearCurr);
            uint cond2 = uint(cellMinZ <= currScreenPos.z) | uint(diff >= thickness);
            if (bool(cond2)) {
                newT = intersectCellBoundary(pRayStart, pRayVector, invD, oldCellIdx, invCellCount);
                level = min(maxLevels, level + 2);
//                v = 1.0;
            }
        } else {
            float depthT = (cellMinZ - pRayStart.z) * invD.z;
            vec3 depthRayPos = pRayStart + pRayVector * depthT;
            vec2 depthRayCellIndex = depthRayPos.xy * cellCount;
            uint cond = uint(depthT > currT) & uint(any(notEqual(oldCellIdxI, ivec2(depthRayCellIndex))));
            if (bool(cond)) {
                newT = min(intersectCellBoundary(pRayStart, pRayVector, invD, oldCellIdx, invCellCount), depthT);
                level = min(maxLevels, level + 2);
//                v = 1.0;
            } else {
                float linearCurr = coords_reversedZToViewZ(currScreenPos.z, near);
                float linearDepth = coords_reversedZToViewZ(cellMinZ, near);
                float diff = linearDepth - linearCurr;
                uint cond = uint(level > STOP_LEVEL) | uint(linearDepth < -65000.0);
                float thickness = bool(cond) ? 1145141919810.0 : MAX_THICKNESS * abs(linearCurr);
                if (diff < thickness) {
                    newT = depthT;
                } else {
                    level++;
                }
            }
        }
        currT = max(newT, currT);
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
        float levelDiv = ldexp(1.0, -level);
        ivec4 mipTile = global_mipmapTiles[0][level];

        #define FIX_STEPS 4

        float stepRcp = rcp(float(FIX_STEPS));

        float minT = currT - 0.01;
        float maxT = 1.0;
        float deltaT = maxT - minT;

        for (uint i = 0; i < FIX_STEPS; i++) {
            float t = minT + ((float(i) + 1.0) * stepRcp * deltaT);
            vec3 screenPos = pRayStart + pRayVector * t;
            vec2 texelPos = screenPos.xy * global_mainImageSize;
            vec2 cellIdx = texelPos * levelDiv;
            ivec2 readPos = ivec2(cellIdx);
            readPos = mipTile.xy + ivec2(clamp(readPos, ivec2(0), mipTile.zw - 1));
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