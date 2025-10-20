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

    Credit:
        - GeforceLegend (https://github.com/GeForceLegend) - Helping with optimizations.
*/

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

float intersectCellBoundary(vec3 invD, vec2 cellIdOffset, vec2 cond2, vec2 cellId, vec2 invCellCount) {
    vec2 cellMin = trunc(cellId);
    vec2 a = cellMin * invD.xy + cellIdOffset;
    vec2 t = a * invCellCount + cond2;

    float tEdge = min(t.x, t.y);
    return tEdge;
}

shared ivec4 shared_mipmapTiles[16];
shared vec4 shared_cellCounts[16];

void sst_init() {
    if (gl_LocalInvocationIndex < 16u) {
        uint maxMip = min(findMSB(min(uval_mainImageSizeI.x, uval_mainImageSizeI.y)), 12u);
        uint mipLevel = min(gl_LocalInvocationIndex, maxMip);
        ivec4 temp = global_mipmapTiles[0][mipLevel];
        temp.zw -= 1;
        shared_mipmapTiles[mipLevel] = temp;

        int mipLevelI = int(mipLevel);
        vec4 mainImageSizeParams = vec4(uval_mainImageSize, uval_mainImageSizeRcp);
        vec2 cellCount = ldexp(mainImageSizeParams.xy, ivec2(-mipLevelI));
        vec2 invCellCount = ldexp(mainImageSizeParams.zw, ivec2(mipLevelI));
        shared_cellCounts[mipLevel] = vec4(cellCount, invCellCount);
    }
    barrier();
}

SSTResult sst_trace(vec3 originView, vec3 rayDirView, float maxThickness) {
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
    float maxThicknessFactor = rcp(1.0 - maxThickness); // 1.0 / (1.0 - maxThickness)
    const float NEAR_Z_THICKNESS_CLAMP = 0.05;

    #define START_LEVEL 1
    #define STOP_LEVEL 0

    int level = START_LEVEL;
    float currT = 0.0;
//    const uvec2 DEBUG_COORD = uvec2(960, 300);
//    const uvec2 DEBUG_COORD = uvec2(487, 250);
    const uint HI_Z_STEPS = 256;

    vec4 mainImageSizeParams = vec4(uval_mainImageSize, uval_mainImageSizeRcp);
    vec3 invD = rcp(pRayVector);
    bvec2 intersectCond1 = greaterThan(pRayVector.xy, vec2(0.0));
    bvec2 vec0Cond = equal(pRayVector.xy, vec2(0.0));
    invD.xy = mix(invD.xy, vec2(0.0), vec0Cond);
    vec2 vec0Fix = vec2(vec0Cond) * 114514.0 - pRayStart.xy * invD.xy;
    vec2 cellIdOffset = mix(vec2(-0.01), vec2(1.01), intersectCond1) * invD.xy;

    vec2 crossStep = sign(pRayVector.xy) * mainImageSizeParams.zw;

    {
        vec3 currScreenPos = pRayStart + pRayVector * currT;
        vec2 currTexelPos = currScreenPos.xy * mainImageSizeParams.xy;
        ivec4 newMipTile = shared_mipmapTiles[0];
        vec4 cellCountData = shared_cellCounts[0];
        vec2 cellCount = cellCountData.xy;
        vec2 invCellCount = cellCountData.zw;
        vec2 cellIdx = (currScreenPos.xy + crossStep) * cellCount;
        currT = max(intersectCellBoundary(invD, cellIdOffset, vec0Fix, cellIdx, invCellCount), currT);
    }

    result.hit = false;
    float negRayEndZ = -pRayStart.z * invD.z;

    for (uint i = 0; i < HI_Z_STEPS; i++) {
        vec4 cellCountData = shared_cellCounts[level];
        ivec4 mipTile = shared_mipmapTiles[level];
        vec3 currScreenPos = pRayStart + pRayVector * currT;
        result.lastMissScreenPos = currScreenPos;

        vec2 cellCount = cellCountData.xy;
        vec2 invCellCount = cellCountData.zw;
        vec2 oldCellIdx = saturate(currScreenPos.xy) * cellCount;

        ivec2 oldCellIdxI = ivec2(oldCellIdx);
        ivec2 readPos = mipTile.xy + oldCellIdxI;
        float cellMinZ = texelFetch(usam_hiz, readPos, 0).r;

        float thicknessFactor = level > STOP_LEVEL ? 1145141919810.0 : min(NEAR_Z_THICKNESS_CLAMP, currScreenPos.z) * maxThicknessFactor;
        if (isBackwardRay) {
            // float linearCurr = coords_reversedZToViewZ(currScreenPos.z, near);
            // float linearDepth = coords_reversedZToViewZ(cellMinZ, near);
            // float diff = (linearDepth - linearCurr);
            // float thickness = level > STOP_LEVEL ? 1145141919810.0 : MAX_THICKNESS * abs(linearCurr);
            // diff >= thickness ...
            // simplyfied to:
            if (any(lessThanEqual(vec2(cellMinZ, thicknessFactor), vec2(currScreenPos.z, cellMinZ)))) {
                float newT = intersectCellBoundary(invD, cellIdOffset, vec0Fix, oldCellIdx, invCellCount);
                currT = newT;
                level++;
            } else  {
                level--;
            }
        } else {
            float depthT = cellMinZ * invD.z + negRayEndZ;
            if (depthT > currT && any(notEqual(oldCellIdxI, ivec2((pRayStart.xy + pRayVector.xy * depthT) * cellCount)))) {
                float newT = intersectCellBoundary(invD, cellIdOffset, vec0Fix, oldCellIdx, invCellCount);
                currT = min(newT, depthT);
                level++;
            } else {
                if (thicknessFactor >= cellMinZ) {
                    currT = max(depthT, currT);
                    level--;
                }
            }
        }

        //        if (gl_GlobalInvocationID.xy == DEBUG_COORD) {
        //            testBuffer[i] = vec4(currScreenPos.xy, floor(oldCellIdx));
        //            testBuffer[i + 2048] = vec4(float(l), 0.0, 0.0, 0.0);
        //            atomicMax(global_atomicCounters[15], i);
        //        }
        if (currT >= 1.0) {
            break;
        }
        if (level < STOP_LEVEL) {
            result.hitScreenPos = pRayStart + pRayVector * currT;
            result.hit = true;
            break;
        }
    }

//    if (!result.hit && currT > 0.01) {
//        level = 0;
//        ivec2 mipTile = shared_mipmapTiles[level].zw - 1;
//
//        #define FIX_STEPS 4
//
//        float stepRcp = rcp(float(FIX_STEPS));
//
//        float minT = currT - 0.01;
//        float maxT = 1.0;
//        float deltaT = maxT - minT;
//
//        for (uint i = 0; i < FIX_STEPS; i++) {
//            float t = minT + ((float(i) + 1.0) * stepRcp * deltaT);
//            vec3 screenPos = pRayStart + pRayVector * t;
//            vec2 texelPos = saturate(screenPos.xy) * mainImageSizeParams.xy;
//            vec2 cellIdx = texelPos;
//            ivec2 readPos = ivec2(cellIdx);
//            readPos = min(readPos, mipTile);
//            float cellMinZ = texelFetch(usam_hiz, readPos, 0).r;
//
//            if (cellMinZ > screenPos.z) {
//                float linearCurr = coords_reversedZToViewZ(screenPos.z, near);
//                float linearDepth = coords_reversedZToViewZ(cellMinZ, near);
//                float diff = linearDepth - linearCurr;
//                float thickness = maxThickness * abs(linearCurr) * 0.01;
//                uint cond = uint(diff < thickness) | uint(linearDepth < -65000.0);
//                if (bool(cond)) {
//                    result.hitScreenPos = screenPos;
//                    result.hit = true;
//                    return result;
//                } else {
//                    return result;
//                }
//            }
//
////            if (gl_GlobalInvocationID.xy == DEBUG_COORD) {
////                testBuffer[i + HI_Z_STEPS] = vec4(screenPos.xy, floor(cellIdx));
////                testBuffer[i + 2048 + HI_Z_STEPS] = vec4(float(level), 0.0, 0.0, 0.0);
////            }
//        }
//    }

    return result;
}