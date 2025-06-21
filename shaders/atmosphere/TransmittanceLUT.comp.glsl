/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere

        You can find full license texts in /licenses
*/
#include "Common.glsl"
#include "/util/Coords.glsl"
#include "/util/Rand.glsl"

layout(rgba16f) uniform restrict image2D uimg_transmittanceLUT;
const ivec3 workGroups = ivec3(2, 64, 1);

layout(local_size_x = 128) in;

#define ATMOSPHERE_RAYMARCHING_TRANSMITTANCE a
#include "Raymarching.glsl"

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    // Compute camera position from LUT coords
    vec2 screenPos = coords_texelToScreen(texelPos, TRANSMITTANCE_TEXEL_SIZE);
    float altitude;
    float cosZenith;
    uvToLutTransmittanceParams(atmosphere, altitude, cosZenith, screenPos);

    vec3 rayDir = vec3(0.0, sqrt(1.0 - pow2(cosZenith)), cosZenith);
    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = vec3(0.0, 0.0, altitude);
    setupRayEnd(atmosphere, params, rayDir);
    params.steps = 32u;

    float jitter = rand_stbnVec1(texelPos, frameCounter);
    vec3 transmittance = raymarchTransmittance(atmosphere, params, jitter);

    vec4 prevData = imageLoad(uimg_transmittanceLUT, texelPos);
    imageStore(uimg_transmittanceLUT, texelPos, temporalUpdate(prevData, transmittance, 32.0));
}
