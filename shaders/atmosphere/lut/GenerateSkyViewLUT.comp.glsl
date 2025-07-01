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

layout(local_size_x = 8, local_size_y = 16) in;

#if SETTING_SKYVIEW_RES == 128
#define SKYVIEW_RES_D16 8
#elif SETTING_SKYVIEW_RES == 256
#define SKYVIEW_RES_D16 16
#elif SETTING_SKYVIEW_RES == 512
#define SKYVIEW_RES_D16 32
#elif SETTING_SKYVIEW_RES == 1024
#define SKYVIEW_RES_D16 64
#endif
const ivec3 workGroups = ivec3(SKYVIEW_RES_D16, SKYVIEW_RES_D16, 1);

#define ATMOSPHERE_RAYMARCHING_SKY a
#include "../Raymarching.glsl"

layout(rgba16f) restrict uniform image2D uimg_skyViewLUT_scattering;
layout(rgba16f) restrict uniform image2D uimg_skyViewLUT_transmittance;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec2 screenPos = (texelPos + 0.5) / SKYVIEW_LUT_SIZE_F;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    vec3 rayStart = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    float viewHeight = length(rayStart);

    float viewZenithCosAngle, lightViewCosAngle;
    uvToSkyViewLutParams(atmosphere, viewZenithCosAngle, lightViewCosAngle, viewHeight, screenPos);

    float viewZenithSinAngle = sqrt(1.0 - pow2(viewZenithCosAngle));
    float lightViewSinAngle = sqrt(1.0 - pow2(lightViewCosAngle));
    vec3 rayDir = vec3(
        viewZenithSinAngle * lightViewCosAngle,
        viewZenithCosAngle,
        viewZenithSinAngle * lightViewSinAngle
    );

    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = rayStart;
    params.steps = SETTING_SKY_SAMPLES;

    ScatteringResult result = scatteringResult_init();
    if (setupRayEnd(atmosphere, params, rayDir)) {
        vec3 upVector = params.rayStart / viewHeight;
        float sunZenithCosAngle = dot(upVector, uval_sunDirWorld);
        float sunZenithSinAngle = sqrt(1.0 - pow2(sunZenithCosAngle));
        vec3 sunDir = normalize(vec3(sunZenithSinAngle, sunZenithCosAngle, 0.0));

        LightParameters sunParam = lightParameters_init(atmosphere, SUN_ILLUMINANCE * PI, sunDir, rayDir);
        LightParameters moonParams = lightParameters_init(atmosphere, vec3(0.0), vec3(0.0), rayDir);
        ScatteringParameters scatteringParams = scatteringParameters_init(sunParam, moonParams, 1.0);

        params.rayStart = params.rayStart + rayDir * (shadowDistance / SETTING_ATM_D_SCALE);
        result = raymarchSky(atmosphere, params, scatteringParams);
    }

    imageStore(uimg_skyViewLUT_scattering, texelPos, vec4(result.inScattering, 1.0));
    imageStore(uimg_skyViewLUT_transmittance, texelPos, vec4(result.transmittance, 1.0));
}