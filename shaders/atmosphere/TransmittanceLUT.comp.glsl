/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere

        You can find full license texts in /licenses
*/
#include "Common.glsl"

layout(rgba16f) uniform image2D uimg_transmittanceLUT;
const ivec3 workGroups = ivec3(2, 64, 1);

layout(local_size_x = 128) in;

#define ATMOSPHERE_RAYMARCHING_TRANSMITTANCE a
#include "Raymarching.glsl"

void main() {
    ivec2 ipixPos = ivec2(gl_GlobalInvocationID.xy);
    vec2 pixPos = vec2(ipixPos + 0.5);
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    // Compute camera position from LUT coords
    vec2 uv = (pixPos) / imageSize(uimg_transmittanceLUT);
    float altitude;
    float cosZenith;
    uvToLutTransmittanceParams(atmosphere, altitude, cosZenith, uv);

    vec3 rayDir = vec3(0.0, sqrt(1.0 - pow2(cosZenith)), cosZenith);
    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = vec3(0.0, 0.0, altitude);
    setupRayEnd(atmosphere, params, rayDir);
    params.steps = 64u;

    vec3 transmittance = raymarchTransmittance(atmosphere, params);

    imageStore(uimg_transmittanceLUT, ipixPos, vec4(transmittance, 1.0));
}
