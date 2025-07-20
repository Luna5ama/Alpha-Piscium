/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere

        You can find full license texts in /licenses
*/
#include "Common.glsl"
#include "/util/Celestial.glsl"
#include "/util/Coords.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(16, 16, 1);

#define ATMOSPHERE_RAYMARCHING_SKY a
#include "Raymarching.glsl"

layout(rgba16f) restrict uniform image2D uimg_skyViewLUT_scattering;
layout(rgba16f) restrict uniform image2D uimg_skyViewLUT_transmittance;

void main() {
    ivec2 imgSize = ivec2(256, 256);
    ivec2 pixelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(pixelPos, imgSize))) {
        vec2 texCoord = (pixelPos + 0.5) / vec2(imgSize);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        vec3 rayDir = coords_octDecode01(texCoord);

        ScatteringResult result = computeSingleScattering(atmosphere, rayDir, rayDir * shadowDistance);
        imageStore(uimg_skyViewLUT_scattering, pixelPos, vec4(result.inScattering, 1.0));
        imageStore(uimg_skyViewLUT_transmittance, pixelPos, vec4(result.transmittance, 1.0));
    }
}