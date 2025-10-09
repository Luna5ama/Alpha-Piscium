#include "Common.glsl"
#include "clouds/Common.glsl"
#include "clouds/amblut/API.glsl"
#include "clouds/Cirrus.glsl"
#include "clouds/Cumulus.glsl"
#include "clouds/ss/Common.glsl"
#include "air/lut/API.glsl"
#include "/util/BitPacking.glsl"
#include "/util/Celestial.glsl"
#include "/util/Math.glsl"

#define LAYER_INDEX 0
#include "air/UnwarpEpipolar.glsl"

ScatteringResult atmospherics_localComposite(int layerIndex, ivec2 texelPos) {
    ivec2 viewZTexelPos = texelPos;
    viewZTexelPos.y += layerIndex * uval_mainImageSizeIY;
    vec2 layerViewZ = texelFetch(usam_csrg32f, viewZTexelPos, 0).xy;
    vec2 screenPos = (vec2(texelPos)) * uval_mainImageSizeRcp;

    ScatteringResult compositeResult = scatteringResult_init();

    if (any(lessThan(layerViewZ, vec2(0.0)))){
        layerViewZ = -abs(layerViewZ);
        ScatteringResult layerResult;
        bool isDepthBreak = !unwarpEpipolarInsctrImage(layerIndex, screenPos * 2.0 - 1.0, layerViewZ, layerResult);
        #ifdef SETTING_DEPTH_BREAK_CORRECTION
        if (layerIndex == 2) {
            uvec4 balllot = subgroupBallot(isDepthBreak);
            uint correctionCount = subgroupBallotBitCount(balllot);
            if (correctionCount > 0) {
                uint writeIndexBase = 0u;
                if (subgroupElect()) {
                    writeIndexBase = atomicAdd(global_dispatchSize1.w, correctionCount);
                    uint totalCount = writeIndexBase + correctionCount;
                    atomicMax(global_dispatchSize1.x, (totalCount | 0x3Fu) >> 6u);
                }
                writeIndexBase = subgroupBroadcastFirst(writeIndexBase);
                if (isDepthBreak) {
                    uint writeIndex = writeIndexBase + subgroupBallotExclusiveBitCount(balllot);
                    uint texelPosEncoded = packUInt2x16(uvec2(texelPos));
                    indirectComputeData[writeIndex] = texelPosEncoded;
                    layerResult = scatteringResult_init();
                }
            }
        }
        #endif
        compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, true);
    }

    return compositeResult;
}