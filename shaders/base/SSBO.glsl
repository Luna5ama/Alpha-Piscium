#ifndef GLOBAL_DATA_MODIFIER
#define GLOBAL_DATA_MODIFIER restrict readonly buffer

#endif

struct AEData {
    vec4 lastScreenAvgLum;
    vec4 screenAvgLum;

    vec4 expValues;
    vec2 hsPercents;
    uint shadowCount;
    uint highlightCount;
    uint weightSum;

    #ifdef SETTING_DEBUG_AE
    uint lumHistogram[256];
    uint lumHistogramMaxBinCount;
    float finalAvgLum;
    #endif
};

struct CameraData {
    uint frustumPlaneCount;
    vec4 frustumPlanes[6];
};

layout(std430, binding = 0) GLOBAL_DATA_MODIFIER GlobalData {
    uvec4 global_dispatchSize1;
    uvec4 global_dispatchSize2;
    uvec4 global_dispatchSize3;
    uvec4 global_dispatchSize4;
    mat4 gbufferPrevModelViewInverse;
    mat4 global_shadowRotationMatrix;
    mat4 global_shadowRotationMatrixInverse;
    mat4 global_taaJitterMat;
    mat4 global_camProj;
    mat4 global_camProjInverse;
    mat4 global_prevCamProj;
    mat4 global_prevCamProjInverse;
    mat4 global_shadowProj;
    mat4 global_shadowProjInverse;
    mat4 global_shadowProjNext;
    mat4 global_shadowProjNextInverse;
    mat4 global_shadowView;
    mat4 global_shadowViewInverse;
    mat4 global_sceneToShadowNDC;
    mat4 global_shadowViewToShadowNDC;
    mat4 global_shadowNDCToScene;
    vec3 global_prevCameraDelta;
    CameraData global_cameraData;
    vec2 global_mipmapSizes[16];
    vec2 global_mipmapSizesRcp[16];
    ivec4 global_mipmapTiles3[16];
    ivec2 global_mipmapSizesI[16];
    ivec2 global_mipmapSizePrefixesPadded[16];
    ivec2 global_mipmapSizePrefixes[16];
    ivec4 global_parallaxMipPackedData[15];
    ivec4 global_hizTiles[16];
    ivec4 global_mipmapTileCeil[16];
    ivec4 global_mipmapTileCeilPadded[16];
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
    vec4 global_motionFactor; // x: speed, y: speed diff, z: angle dot, w: speed factor
    vec4 global_taaResetFactor; // x: clamp strictness, y: clamp method, z: history reset, w: clamp mix
    float global_turbidity;
    AEData global_aeData;
    uint global_atomicCounters[16];
    uint global_restirVisibilityRayCount;

    uint rc_entryCounter;
    uint rc_allocationCounter;
    uint rc_keyMismatchCounter;
    uint rc_poolOverflowCounter;
    uint rc_cacheHitCounter;
    uint rc_cacheMissCounter;
    uint pg_selectedCounter;
    uint pg_splatCounter;
    uint pg_noSlotCounter;
    uint pg_neighborCounter;
    uint pg_skyCounter;
    uint rc_paddingCounters[6];

    // Last wider projection with observed opaque geometry. Appended for ABI stability.
    int global_giWideState;
    ivec2 global_giWideSize;
    ivec3 global_giWideEyeInt;
    vec3 global_giWideEyeFract;
    vec2 global_giWideJitter;
    float global_giWideNear;
    float global_giWideRain;
    float global_giWideDiffuseValidity;
    int global_giWideTime;
    mat4 global_giWideProj;
    mat4 global_giWideInvProj;
    mat4 global_giWideView;
    #ifdef VOXY
    ivec3 global_giModelOrigin;
    uint global_giModelEpoch;
    // The original 16 KiB global block also serves passes with other option layouts.
    layout(offset = 16384) uvec2 global_giObservedCells[32768];
    uint global_giLightGroupEpoch[512];
    uint global_giLightEpoch;
    // An observed emitter retains independent samples across its face. A
    // partially visible edge must not recolor the entire off-screen block.
    layout(offset = 280592) uvec2 global_giLightSamples[32768 * 64];
    uint global_giLightSampleMaterial[32768 * 64];
    #endif
};

layout(std430, binding = 1) GLOBAL_DATA_MODIFIER IndirectComputeData {
    uint indirectComputeData[];
};
