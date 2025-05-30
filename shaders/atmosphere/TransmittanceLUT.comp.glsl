/*
    Contains code adopted from:
        https://github.com/sebh/UnrealEngineSkyAtmosphere
        MIT License
        Copyright (c) 2020 Epic Games, Inc.

        You can find full license texts in /licenses
*/
#include "Common.glsl"

layout(rgba16f) uniform image2D uimg_transmittanceLUT;
const ivec3 workGroups = ivec3(2, 64, 1);

layout(local_size_x = 128) in;

void main() {
    ivec2 ipixPos = ivec2(gl_GlobalInvocationID.xy);
    vec2 pixPos = vec2(ipixPos + 0.5);
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    // Compute camera position from LUT coords
    vec2 uv = (pixPos) / imageSize(uimg_transmittanceLUT);
    float altitude;
    float cosZenith;
    uvToLutTransmittanceParams(atmosphere, altitude, cosZenith, uv);

    //  A few extra needed constants
    vec3 pos = vec3(0.0f, 0.0f, altitude);
    vec3 dir = vec3(0.0f, sqrt(1.0 - cosZenith * cosZenith), cosZenith);

    vec3 transmittance = raymarchTransmittance(atmosphere, pos, dir, 64u);

    imageStore(uimg_transmittanceLUT, ipixPos, vec4(transmittance, 1.0));
}
