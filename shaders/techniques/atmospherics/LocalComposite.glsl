#include "Common.glsl"
#include "clouds/Common.glsl"
#include "clouds/amblut/API.glsl"
#include "clouds/Cirrus.glsl"
#include "clouds/Cumulus.glsl"
#include "clouds/ss/Common.glsl"
#include "air/lut/API.glsl"
#include "air/UnwarpEpipolar.glsl"
#include "/util/BitPacking.glsl"
#include "/util/Celestial.glsl"
#include "/util/Math.glsl"

ScatteringResult atmospherics_localComposite(ivec2 texelPos) {
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * uval_mainImageSizeRcp;

    ScatteringResult compositeResult = scatteringResult_init();

    {
        ScatteringResult layerResult;
        #ifndef SETTING_DEPTH_BREAK_CORRECTION
        unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, layerResult);
        #else
        bool isDepthBreak = !unwarpEpipolarInsctrImage(screenPos * 2.0 - 1.0, viewZ, layerResult);
        uvec4 balllot = subgroupBallot(isDepthBreak);
        uint correctionCount = subgroupBallotBitCount(balllot);
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
        #endif
        compositeResult = scatteringResult_blendLayer(compositeResult, layerResult, true);
    }

    return compositeResult;
}