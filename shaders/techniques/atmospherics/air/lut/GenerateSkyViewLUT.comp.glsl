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

layout(local_size_x = 16, local_size_y = 16) in;

const ivec3 workGroups = ivec3(SKYVIEW_RES_D16, SKYVIEW_RES_D16, 4);

/*const*/
#define ATMOSPHERE_RAYMARCHING_SKY a
/*const*/
#include "../Raymarching.glsl"

layout(rgba16f) restrict uniform writeonly image3D uimg_skyViewLUT;

const float BOTTOM_OFFSET = 8.0;

bool setupRayEndC(AtmosphereParameters atmosphere, inout RaymarchParameters params, vec3 rayDir) {
    const vec3 earthCenter = vec3(0.0);
    float rayStartHeight = length(params.rayStart);

    // Check if ray origin is outside the atmosphere
    if (rayStartHeight > atmosphere.top) {
        float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
        if (tTop < 0.0) {
            return false; // No intersection with atmosphere: stop right away
        }
        vec3 upVector = params.rayStart / rayStartHeight;
        vec3 upOffset = upVector * -PLANET_RADIUS_OFFSET;
        params.rayStart += rayDir * tTop + upOffset;
    }

    float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom - BOTTOM_OFFSET);
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

    params.rayEnd = params.rayStart + rayDir * rayLen;

    return true;
}

bool setupRayEndLayered(AtmosphereParameters atmosphere, inout RaymarchParameters params, vec3 rayDir, vec2 layerBound) {
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
        float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom - BOTTOM_OFFSET);
        if (tBottom >= 0.0 || tLayerBottom < 0.0) {
            return false;
        }
        vec3 upVector = params.rayStart / rayStartHeight;
        vec3 upOffset = upVector * 0.001;
        params.rayStart += rayDir * tLayerBottom + upOffset;
    }

    float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom - BOTTOM_OFFSET);
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

    float tLayerBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, layerBound.x);
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
    vec2(-100.0, SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS * 0.5),
    vec2(SETTING_CLOUDS_CU_HEIGHT + SETTING_CLOUDS_CU_THICKNESS * 0.5, SETTING_CLOUDS_CI_HEIGHT),
    vec2(SETTING_CLOUDS_CI_HEIGHT, 100.0)
};

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec2 screenPos = (texelPos + 0.5) / SKYVIEW_LUT_SIZE_F;

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    vec3 rayStart = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    float viewHeight = length(rayStart);

    // Decode UV -> lat/lon -> world-space ray direction
    float lat, lon;
    _atmospherics_air_lut_uvToSkyViewLonLat(screenPos, lat, lon);
    vec3 rayDir;
    _atmospherics_air_lut_skyViewLonLatToRayDir(lat, lon, rayDir);

    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = rayStart;
    params.steps = SETTING_SKY_SAMPLES;

    int layerIndex = int(gl_WorkGroupID.z);

    LightParameters sunParams = lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, rayDir);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, rayDir);
    ScatteringParameters scatteringParams = scatteringParameters_init(sunParams, moonParams, 1.0);

    ScatteringResult result = scatteringResult_init();

    if (layerIndex == 0) {
        if (setupRayEndC(atmosphere, params, rayDir)) {
            params.rayStart = params.rayStart + rayDir * (shadowDistance / SETTING_ATM_D_SCALE);
            result = raymarchSky(atmosphere, params, scatteringParams, BOTTOM_OFFSET);
        }
    } else {
        vec2 layerBound = LAYER_BOUNDS[layerIndex - 1] + atmosphere.bottom;
        layerBound = min(layerBound, vec2(atmosphere.top));

        if (setupRayEndLayered(atmosphere, params, rayDir, layerBound)) {
            result = raymarchSky(atmosphere, params, scatteringParams, BOTTOM_OFFSET);
        }
    }

    result.inScattering = result.inScattering;
    result.transmittance = result.transmittance;

    ivec3 writePos = ivec3(texelPos, layerIndex * 2);
    imageStore(uimg_skyViewLUT, writePos, vec4(result.inScattering, 0.0));
    writePos.z += 1;
    imageStore(uimg_skyViewLUT, writePos, vec4(result.transmittance, 0.0));
}