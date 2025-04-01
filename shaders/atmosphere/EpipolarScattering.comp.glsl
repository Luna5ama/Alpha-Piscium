// Contains code adopted from:
// https://github.com/sebh/UnrealEngineSkyAtmosphere
// MIT License
// Copyright (c) 2020 Epic Games, Inc.
//
// https://github.com/GameTechDev/OutdoorLightScattering
// Apache License 2.0
// Copyright (c) 2017 Intel Corporation
//
// You can find full license texts in /licenses
#include "Common.glsl"
#include "Scattering.glsl"
#include "/rtwsm/RTWSM.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"

layout(local_size_x = 1, local_size_y = SETTING_SLICE_SAMPLES) in;
const ivec3 workGroups = ivec3(SETTING_EPIPOLAR_SLICES, 1, 1);

uniform sampler2D usam_gbufferViewZ;

layout(rgba32f) uniform readonly image2D uimg_epipolarSliceEnd;
layout(rgba32ui) uniform writeonly uimage2D uimg_epipolarData;

void main() {
    ivec2 imgSizei = ivec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    vec2 imgSize = vec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    ivec2 pixelPos = ivec2(gl_GlobalInvocationID.xy);
    uint sliceIndex = gl_WorkGroupID.x;
    uint sliceSampleIndex = gl_LocalInvocationID.y;
    vec4 sliceEndPoints = imageLoad(uimg_epipolarSliceEnd, ivec2(sliceIndex, 0));

    uint cond = uint(all(lessThan(pixelPos, imgSizei)));
    cond &= uint(isValidScreenLocation(sliceEndPoints.xy)) | uint(isValidScreenLocation(sliceEndPoints.zw));

    if (bool(cond)) {
        float ignValue = rand_stbnVec1(ivec2(gl_GlobalInvocationID.xy), frameCounter);
        float sliceSampleP = float(sliceSampleIndex);
        sliceSampleP += ignValue - 0.5;
        sliceSampleP /= float(SETTING_SLICE_SAMPLES - 1);
        sliceSampleP = sliceSampleP * sliceSampleP;

        vec2 texCoord = mix(sliceEndPoints.xy, sliceEndPoints.zw, sliceSampleP) * 0.5 + 0.5;
        texCoord = (round(texCoord * global_mainImageSize) + 0.5) * global_mainImageSizeRcp;

        AtmosphereParameters atmosphere = getAtmosphereParameters();
        float viewZ = textureLod(usam_gbufferViewZ, texCoord, 0.0).r;
        vec3 viewCoord = coords_toViewCoord(texCoord, viewZ, gbufferProjectionInverse);

        ScatteringResult result = computeSingleScattering(atmosphere, vec3(0.0), viewCoord, ignValue);

        uvec4 outputData;
        packEpipolarData(outputData, result, viewZ);
        imageStore(uimg_epipolarData, pixelPos, outputData);
    }
}