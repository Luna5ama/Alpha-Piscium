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
#define WORK_GROUP_COUNT 1
#elif SETTING_SLICE_SAMPLES == 256
#define WORK_GROUP_SIZE 256
#define WORK_GROUP_COUNT 1
#elif SETTING_SLICE_SAMPLES == 512
#define WORK_GROUP_SIZE 256
#define WORK_GROUP_COUNT 2
#elif SETTING_SLICE_SAMPLES == 1024
#define WORK_GROUP_SIZE 256
#define WORK_GROUP_COUNT 4
#endif
layout(local_size_x = 1, local_size_y = WORK_GROUP_SIZE) in;
const ivec3 workGroups = ivec3(SETTING_EPIPOLAR_SLICES, WORK_GROUP_COUNT, 1);

#define SHARED_MEMORY_SHADOW_SAMPLE a
#include "RaymarchScreenViewAtmosphere.glsl"

layout(rgba32ui) uniform restrict uimage2D uimg_epipolarData;

void main() {
    ivec2 imgSizei = ivec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    vec2 imgSize = vec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    uint sliceIndex = gl_WorkGroupID.x;
    uint sliceSampleIndex = gl_GlobalInvocationID.y;
    vec4 sliceEndPoints = uintBitsToFloat(imageLoad(uimg_epipolarData, ivec2(sliceIndex, 0)));

    ScatteringResult result = scatteringResult_init();
    float viewZ = 65536.0;

    uint cond = uint(all(lessThan(texelPos, imgSizei)));
    cond &= uint(isValidScreenLocation(sliceEndPoints.xy)) | uint(isValidScreenLocation(sliceEndPoints.zw));

    if (bool(cond)) {
        float sliceSampleP = float(sliceSampleIndex);
        sliceSampleP /= float(SETTING_SLICE_SAMPLES - 1);

        vec2 screenPos = mix(sliceEndPoints.xy, sliceEndPoints.zw, sliceSampleP) * 0.5 + 0.5;
        screenViewRaymarch_init(screenPos);

        vec2 texelPos = screenPos * uval_mainImageSize;
        texelPos = clamp(texelPos, vec2(0.5), vec2(uval_mainImageSize - 0.5));
        ivec2 texelPosI = ivec2(texelPos);
        float noiseV = rand_stbnVec1(texelPosI, frameCounter);

        viewZ = texelFetch(usam_gbufferViewZ, texelPosI, 0).r;
        result = raymarchScreenViewAtmosphere(texelPosI, viewZ, SETTING_LIGHT_SHAFT_SAMPLES, noiseV);
    }

    uvec4 outputData;
    packEpipolarData(outputData, result, viewZ);
    ivec2 writeTexelPos = texelPos;
    writeTexelPos.y += 1;
    imageStore(uimg_epipolarData, writeTexelPos, outputData);
}