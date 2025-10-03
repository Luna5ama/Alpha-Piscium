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
            uint sliceSampleIndex = gl_LocalInvocationID.y + (LOOP_COUNT - 1) * WORK_GROUP_SIZE;
            float sliceSampleP = float(sliceSampleIndex);
            vec2 screenPos = mix(sliceEndPoints.xy, sliceEndPoints.zw, sliceSampleP) * 0.5 + 0.5;
            screenViewRaymarch_init(screenPos);
        }
        for (uint i = 0; i < LOOP_COUNT; i++) {
            uint sliceSampleIndex = gl_LocalInvocationID.y + i * WORK_GROUP_SIZE;
            float sliceSampleP = float(sliceSampleIndex);
            sliceSampleP /= float(SETTING_SLICE_SAMPLES - 1);

            vec2 screenPos = mix(sliceEndPoints.xy, sliceEndPoints.zw, sliceSampleP) * 0.5 + 0.5;

            vec2 texelPos = screenPos * uval_mainImageSize;
            texelPos = clamp(texelPos, vec2(0.5), vec2(uval_mainImageSize - 0.5));
            ivec2 texelPosI = ivec2(texelPos);
            float noiseV = rand_stbnVec1(texelPosI, frameCounter);

            float viewZ = texelFetch(usam_gbufferViewZ, texelPosI, 0).r;
            ScatteringResult result = raymarchScreenViewAtmosphere(texelPosI, viewZ, SETTING_LIGHT_SHAFT_SAMPLES, noiseV);
            uvec4 outputData;
            packEpipolarData(outputData, result, viewZ);
            ivec2 writeTexelPos = ivec2(gl_GlobalInvocationID.x, sliceSampleIndex + 1u);
            imageStore(uimg_epipolarData, writeTexelPos, outputData);
        }
    } else {
        float viewZ = 65536.0;
        ScatteringResult result = scatteringResult_init();
        uvec4 outputData;
        packEpipolarData(outputData, result, viewZ);
        for (uint i = 0; i < LOOP_COUNT; i++) {
            ivec2 writeTexelPos = ivec2(gl_GlobalInvocationID.xy);
            writeTexelPos.y += int(1u + i * WORK_GROUP_SIZE);
            imageStore(uimg_epipolarData, writeTexelPos, outputData);
        }
    }
}