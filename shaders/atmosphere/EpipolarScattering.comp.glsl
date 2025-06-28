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
#include "/rtwsm/RTWSM.glsl"
#include "/util/Coords.glsl"
#include "/util/Lighting.glsl"
#include "/util/Rand.glsl"

layout(local_size_x = 1, local_size_y = SETTING_SLICE_SAMPLES) in;
const ivec3 workGroups = ivec3(SETTING_EPIPOLAR_SLICES, 1, 1);


#include "Scattering.glsl"

layout(rgba32ui) uniform restrict uimage2D uimg_epipolarData;

void main() {
    ivec2 imgSizei = ivec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    vec2 imgSize = vec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    uint sliceIndex = gl_WorkGroupID.x;
    uint sliceSampleIndex = gl_LocalInvocationID.y;
    vec4 sliceEndPoints = uintBitsToFloat(imageLoad(uimg_epipolarData, ivec2(sliceIndex, 0)));

    ScatteringResult result = scatteringResult_init();
    float viewZ = 65536.0;

    uint cond = uint(all(lessThan(texelPos, imgSizei)));
    cond &= uint(isValidScreenLocation(sliceEndPoints.xy)) | uint(isValidScreenLocation(sliceEndPoints.zw));

    if (bool(cond)) {
        vec2 noiseV = rand_stbnVec2(ivec2(gl_GlobalInvocationID.xy), frameCounter);
        float sliceSampleP = float(sliceSampleIndex);
        sliceSampleP += noiseV.x - 0.5;
        sliceSampleP /= float(SETTING_SLICE_SAMPLES - 1);
        sliceSampleP = sliceSampleP * sliceSampleP;

        vec2 screenPos = mix(sliceEndPoints.xy, sliceEndPoints.zw, sliceSampleP) * 0.5 + 0.5;
        vec2 texelPos = screenPos * global_mainImageSize;
        texelPos = clamp(texelPos, vec2(0.5), vec2(global_mainImageSize - 0.5));
        screenPos = texelPos * global_mainImageSizeRcp;

        viewZ = texelFetch(usam_gbufferViewZ, ivec2(texelPos), 0).r;
        result = computeSingleScattering(screenPos, viewZ, noiseV.y);
    }

    uvec4 outputData;
    packEpipolarData(outputData, result, viewZ);
    ivec2 writeTexelPos = texelPos;
    writeTexelPos.y += 1;
    imageStore(uimg_epipolarData, writeTexelPos, outputData);
}