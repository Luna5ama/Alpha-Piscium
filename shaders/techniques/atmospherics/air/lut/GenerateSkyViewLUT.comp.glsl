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

const ivec3 workGroups = ivec3(SKYVIEW_RES_D16, SKYVIEW_RES_D16, 6);

#define ATMOSPHERE_RAYMARCHING_SKY_SINGLE a
#include "../Raymarching.glsl"

layout(rgba8) restrict uniform writeonly image3D uimg_skyViewLUT;

bool setupRayEndLayered(AtmosphereParameters atmosphere, inout RaymarchParameters params, vec3 rayDir, vec2 layerBound, float bottomOffset) {
    const vec3 earthCenter = vec3(0.0);
    float rayStartHeight = length(params.rayStart);

    if (rayStartHeight > layerBound.y) {
        float tLayerTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, layerBound.y);
        if (tLayerTop < 0.0) {
            return false; // No intersection with atmosphere: stop right away
        }
        vec3 upVector = params.rayStart / rayStartHeight;
        vec3 upOffset = upVector * -0.001;
        params.rayStart += rayDir * tLayerTop + upOffset;
    } else if (rayStartHeight < layerBound.x) {
        float tLayerBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, layerBound.x);
        float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom - bottomOffset);
        if (tBottom >= 0.0 || tLayerBottom < 0.0) {
            return false;
        }
        vec3 upVector = params.rayStart / rayStartHeight;
        vec3 upOffset = upVector * 0.001;
        params.rayStart += rayDir * tLayerBottom + upOffset;
    }

    float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom - bottomOffset);
    float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);

    float rayLen = 0.0;

    if (tBottom < 0.0) {
        if (tTop < 0.0) {
            return false; // No intersection with earth nor atmosphere: stop right away
        } else {
            rayLen = tTop;
        }
    } else {
        if (tTop > 0.0) {
            rayLen = min(tTop, tBottom);
        }
    }

    float tLayerBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, layerBound.x - bottomOffset);
    float tLayerTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, layerBound.y);

    uint inLayer = uint(rayStartHeight > layerBound.x - 0.001) & uint(rayStartHeight < layerBound.y);

    if (bool(inLayer)) {
        rayLen = min(rayLen, tLayerBottom > 0.0 ? tLayerBottom : rayLen);
        rayLen = min(rayLen, tLayerTop > 0.0 ? tLayerTop : rayLen);
    } else {
        if (rayStartHeight < layerBound.x) {
            // Below the layer
            if (tLayerTop >= 0.0) {
                rayLen = min(rayLen, tLayerTop);
            } else {
                return false;
            }
        } else {
            // Above the layer
            if (rayDir.y >= 0.0) return false;
            rayLen = min(rayLen, tLayerBottom > 0.0 ? tLayerBottom : rayLen);
        }
    }

    params.rayEnd = params.rayStart + rayDir * rayLen;

    return true;
}

const vec2[3] LAYER_BOUNDS = {
    vec2(-100.0, SETTING_CLOUDS_CU_HEIGHT),
    vec2(SETTING_CLOUDS_CU_HEIGHT, SETTING_CLOUDS_CI_HEIGHT),
    vec2(SETTING_CLOUDS_CI_HEIGHT, 100.0)
};

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
    int layerIndex = workGroupZI >> 1;
    int isMoonI = workGroupZI & 1;

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

    float groundFactor = exp2(-4.0 * (viewHeight - atmosphere.bottom));
    float bottomOffset = groundFactor * 10.0;

    vec2 layerBound = LAYER_BOUNDS[layerIndex] + atmosphere.bottom;
    layerBound = min(layerBound, vec2(atmosphere.top));

    if (setupRayEndLayered(atmosphere, params, rayDir, layerBound, bottomOffset)) {
        result = raymarchSkySingle(atmosphere, params, lightParam, bottomOffset);
    }

    ivec3 writePos = ivec3(texelPos, layerIndex * 3);
    writePos.z += isMoonI;
    imageStore(uimg_skyViewLUT, writePos, colors_sRGBToLogLuv32(result.inScattering));
    if (isMoonI == 0) {
        writePos.z += 2;
        imageStore(uimg_skyViewLUT, writePos, colors_sRGBToLogLuv32(result.transmittance));
    }
}