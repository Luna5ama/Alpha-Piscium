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
#include "/util/Rand.glsl"

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
const ivec3 workGroups = ivec3(SKYVIEW_RES_D16, SKYVIEW_RES_D16, 2);

#define ATMOSPHERE_RAYMARCHING_SKY_SINGLE a
#include "../Raymarching.glsl"

layout(rgba8) restrict uniform writeonly image3D uimg_skyViewLUT;

bool setupRayEndC(AtmosphereParameters atmosphere, inout RaymarchParameters params, vec3 rayDir, float bottomOffset) {
    const vec3 earthCenter = vec3(0.0);
    float rayStartHeight = length(params.rayStart);

    // Check if ray origin is outside the techniques.atmosphere
    if (rayStartHeight > atmosphere.top) {
        float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
        if (tTop < 0.0) {
            return false; // No intersection with techniques.atmosphere: stop right away
        }
        vec3 upVector = params.rayStart / rayStartHeight;
        vec3 upOffset = upVector * -PLANET_RADIUS_OFFSET;
        params.rayStart += rayDir * tTop + upOffset;
    }

    float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom - bottomOffset);
    float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
    float rayLen = 0.0;

    if (tBottom < 0.0) {
        if (tTop < 0.0) {
            return false; // No intersection with earth nor techniques.atmosphere: stop right away
        } else {
            rayLen = tTop;
        }
    } else {
        if (tTop > 0.0) {
            rayLen = min(tTop, tBottom);
        }
    }

    params.rayEnd = params.rayStart + rayDir * rayLen;

    return true;
}

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec2 screenPos = (texelPos + 0.5) / SKYVIEW_LUT_SIZE_F;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    vec3 rayStart = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    float viewHeight = length(rayStart);

    float viewZenithCosAngle, lightViewCosAngle;
    _atmospherics_air_lut_uvToSkyViewLutParams(atmosphere, viewZenithCosAngle, lightViewCosAngle, viewHeight, screenPos);

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
    int workGroupZI = int(gl_WorkGroupID.z);
    int isMoonI = workGroupZI & 1;

    float groundFactor = exp2(-4.0 * (viewHeight - atmosphere.bottom));
    float bottomOffset = groundFactor * 10.0;

    if (setupRayEndC(atmosphere, params, rayDir, bottomOffset)) {
        vec3 upVector = params.rayStart / viewHeight;

        LightParameters lightParam;
        if (bool(isMoonI)) {
            float moonZenithCosAngle = dot(upVector, uval_moonDirWorld);
            float moonZenithSinAngle = sqrt(1.0 - pow2(moonZenithCosAngle));
            vec3 moonDir = normalize(vec3(moonZenithSinAngle, moonZenithCosAngle, 0.0));
            lightParam = lightParameters_init(atmosphere, MOON_ILLUMINANCE, moonDir, rayDir);
        } else {
            float sunZenithCosAngle = dot(upVector, uval_sunDirWorld);
            float sunZenithSinAngle = sqrt(1.0 - pow2(sunZenithCosAngle));
            vec3 sunDir = normalize(vec3(sunZenithSinAngle, sunZenithCosAngle, 0.0));
            lightParam = lightParameters_init(atmosphere, SUN_ILLUMINANCE * PI, sunDir, rayDir);
        }

        params.rayStart = params.rayStart + rayDir * (shadowDistance / SETTING_ATM_D_SCALE);
        result = raymarchSkySingle(atmosphere, params, lightParam, bottomOffset);
    }

    uint layerIndex = workGroupZI >> 1u;
    ivec3 writePos = ivec3(texelPos, layerIndex * 3);
    writePos.z += isMoonI;
    imageStore(uimg_skyViewLUT, writePos, colors_sRGBToLogLuv32(result.inScattering));
    if (isMoonI == 0) {
        writePos.z += 2;
        imageStore(uimg_skyViewLUT, writePos, colors_sRGBToLogLuv32(result.transmittance));
    }
}