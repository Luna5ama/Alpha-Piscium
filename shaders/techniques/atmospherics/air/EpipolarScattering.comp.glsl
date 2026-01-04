/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere
        [INT17] Intel Corporation. "Outdoor Light Scattering Sample". 2017.
            Apache License 2.0. Copyright (c) 2017 Intel Corporation.
            https://github.com/GameTechDev/OutdoorLightScattering

        You can find full license texts in /licenses
*/
#include "Common.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"
#include "/util/Coords.glsl"
#include "/util/Lighting.glsl"
#include "/util/Rand.glsl"

#if SETTING_SLICE_SAMPLES == 128
#define WORK_GROUP_SIZE 128
#define LOOP_COUNT 1
#elif SETTING_SLICE_SAMPLES == 256
#define WORK_GROUP_SIZE 256
#define LOOP_COUNT 1
#elif SETTING_SLICE_SAMPLES == 512
#define WORK_GROUP_SIZE 256
#define LOOP_COUNT 2
#elif SETTING_SLICE_SAMPLES == 1024
#define WORK_GROUP_SIZE 256
#define LOOP_COUNT 4
#endif
layout(local_size_x = 1, local_size_y = WORK_GROUP_SIZE) in;
const ivec3 workGroups = ivec3(SETTING_EPIPOLAR_SLICES, 1, 1);

#define SHARED_MEMORY_SHADOW_SAMPLE a
#include "RaymarchScreenViewAtmosphere.glsl"

layout(rgba32ui) uniform restrict uimage2D uimg_epipolarData;

void main() {
    ivec2 imgSizei = ivec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    vec2 imgSize = vec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    uint sliceIndex = gl_WorkGroupID.x;
    vec4 sliceEndPoints = uintBitsToFloat(imageLoad(uimg_epipolarData, ivec2(sliceIndex, 0)));

    uint cond = uint(isValidScreenLocation(sliceEndPoints.xy)) | uint(isValidScreenLocation(sliceEndPoints.zw));

    if (bool(cond)) {
        {
            vec2 screenPos = mix(sliceEndPoints.xy, sliceEndPoints.zw, 0.5) * 0.5 + 0.5;
            screenViewRaymarch_init(screenPos);
        }
        for (uint i = 0; i < LOOP_COUNT; i++) {
            uint sliceSampleIndex = gl_LocalInvocationID.y + i * WORK_GROUP_SIZE;
            float sliceSampleP = float(sliceSampleIndex)  / float(SETTING_SLICE_SAMPLES - 1);
            vec2 screenPos = mix(sliceEndPoints.xy, sliceEndPoints.zw, sliceSampleP) * 0.5 + 0.5;

            vec2 texelPos = screenPos * uval_mainImageSize;
            texelPos = clamp(texelPos, vec2(0.5), vec2(uval_mainImageSize - 0.5));
            ivec2 texelPosI = ivec2(texelPos);
            float noiseV = rand_stbnVec1(texelPosI, frameCounter);

            ivec2 baseWriteTexelPos = ivec2(gl_GlobalInvocationID.x, sliceSampleIndex + 1u);

            for (int layerIndex = 0; layerIndex < 2; layerIndex++) {
                int actualIndex = layerIndex * 2;
                vec2 layerViewZ;
                if (actualIndex == 0) {
                    layerViewZ = uintBitsToFloat(transient_translucentZLayer1_fetch(texelPosI).xy);
                } else if (actualIndex == 1) {
                    layerViewZ = uintBitsToFloat(transient_translucentZLayer2_fetch(texelPosI).xy);
                } else {
                    layerViewZ = uintBitsToFloat(transient_translucentZLayer3_fetch(texelPosI).xy);
                }
                layerViewZ = -abs(layerViewZ);
                ScatteringResult result = scatteringResult_init();

                if (layerViewZ.x > -FLT_MAX) {
                    result = raymarchScreenViewAtmosphere(
                        texelPosI,
                        layerViewZ.x,
                        layerViewZ.y,
                        SETTING_LIGHT_SHAFT_SAMPLES,
                        noiseV
                    );
                }

                uvec4 outputData;
                packEpipolarData(outputData, result, texelPosI);
                ivec2 writeTexelPos = baseWriteTexelPos;
                writeTexelPos.y += actualIndex * int(SETTING_SLICE_SAMPLES);
                imageStore(uimg_epipolarData, writeTexelPos, outputData);
            }
        }
    } else {
        ivec2 texelPos = ivec2(65535);
        ScatteringResult result = scatteringResult_init();
        uvec4 outputData;
        packEpipolarData(outputData, result, texelPos);
        for (uint i = 0; i < LOOP_COUNT; i++) {
            ivec2 baseWriteTexelPos = ivec2(gl_GlobalInvocationID.xy);
            baseWriteTexelPos.y += int(1u + i * WORK_GROUP_SIZE);
            for (int layerIndex = 0; layerIndex < 2; layerIndex++) {
                int actualIndex = layerIndex * 2;
                ivec2 writeTexelPos = baseWriteTexelPos;
                writeTexelPos.y += actualIndex * int(SETTING_SLICE_SAMPLES);
                imageStore(uimg_epipolarData, writeTexelPos, outputData);
            }
        }
    }
}