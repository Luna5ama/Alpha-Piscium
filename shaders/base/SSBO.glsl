#ifndef GLOBAL_DATA_MODIFIER
#define GLOBAL_DATA_MODIFIER restrict readonly

#endif

struct AEData {
    vec3 expValues;
    uint shadowCount;
    uint highlightCount;
    uint weightSum;
    uint avgLumHistogram[256];

    #ifdef SETTING_DEBUG_AE
    uint lumHistogram[256];
    uint lumHistogramMaxBinCount;
    float finalAvgLum;
    #endif
};

mat4 gbufferPrevModelView = gbufferPreviousModelView;

layout(std430, binding = 0) GLOBAL_DATA_MODIFIER buffer GlobalData {
    uvec4 global_dispatchSize1;
    uvec4 global_dispatchSize2;
    uvec4 global_dispatchSize3;
    uvec4 global_dispatchSize4;
    mat4 gbufferPrevModelViewInverse;
    mat4 global_shadowRotationMatrix;
    mat4 global_taaJitterMat;
    mat4 global_camProj;
    mat4 global_camProjInverse;
    mat4 global_prevCamProj;
    mat4 global_prevCamProjInverse;
    mat4 global_shadowProjPrev;
    mat4 global_shadowProjInversePrev;
    mat4 global_shadowProj;
    mat4 global_shadowProjInverse;
    mat4 global_shadowView;
    mat4 global_shadowViewInverse;
    vec3 global_prevCameraDelta;
    vec2 global_taaJitter;
    ivec2 global_mainImageSizeI;
    vec2 global_mainImageSize;
    vec2 global_mainImageSizeRcp;
    vec2 global_mipmapSizes[16];
    vec2 global_mipmapSizesRcp[16];
    ivec2 global_mipmapSizesI[16];
    ivec2 global_mipmapSizePrefixes[16];
    ivec4 global_mipmapTiles[2][16];
    uvec2 global_frameMortonJitter;
    ivec3 global_shadowAABBMinPrev;
    ivec3 global_shadowAABBMaxPrev;
    ivec3 global_shadowAABBMin;
    ivec3 global_shadowAABBMax;
    vec3 global_shadowAABBMinHistory;
    vec3 global_shadowAABBMaxHistory;
    ivec3 global_shadowAABBMinNew;
    ivec3 global_shadowAABBMaxNew;
    float global_focusDistance;
    int global_lastWorldTime;
    float global_historyResetFactor;
    float global_turbidity;
    AEData global_aeData;
    uint global_atomicCounters[16];
};

layout(std430, binding = 1) GLOBAL_DATA_MODIFIER buffer IndirectComputeData {
    uint indirectComputeData[];
};

const vec2 SHADOW_MAP_SIZE = vec2(float(shadowMapResolution), 1.0 / float(shadowMapResolution));