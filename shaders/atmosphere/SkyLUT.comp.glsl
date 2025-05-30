/*
    Contains code adopted from:
        https://github.com/sebh/UnrealEngineSkyAtmosphere
        MIT License
        Copyright (c) 2020 Epic Games, Inc.

        You can find full license texts in /licenses
*/
#include "Common.glsl"
#include "/util/Celestial.glsl"
#include "/util/Coords.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(8, 8, 1);

layout(rgba16f) restrict uniform image2D uimg_skyLUT;

// originView: ray origin in view space
// endView: ray end in view space
ScatteringResult computeSingleScattering(AtmosphereParameters atmosphere, vec3 rayDir) {
    ScatteringResult result = ScatteringResult(vec3(1.0), vec3(0.0));
    if (all(equal(rayDir, vec3(0.0)))) {
        return result;
    }

    vec3 originView = vec3(0.0, 0.0, 0.0);

    RaymarchParameters params;
    params.rayStart = atmosphere_viewToAtm(atmosphere, originView);
    params.rayStart.y = max(params.rayStart.y, atmosphere.bottom + 0.001);

    LightParameters sunParams = lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, rayDir);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, rayDir);
    ScatteringParameters scatteringParams = scatteringParameters_init(sunParams, moonParams, 1.0);

    params.steps = SETTING_SKY_SAMPLES;

    vec3 earthCenter = vec3(0.0);

    // Check if ray origin is outside the atmosphere
    if (length(params.rayStart) > atmosphere.top) {
        float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
        if (tTop < 0.0) {
            return result; // No intersection with atmosphere: stop right away
        }
        params.rayStart += rayDir * (tTop + PLANET_RADIUS_OFFSET);
    }

    float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom);
    float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
    float rayLen = 0.0;

    if (tBottom < 0.0) {
        if (tTop < 0.0) {
            return result; // No intersection with earth nor atmosphere: stop right away
        } else {
            rayLen = tTop;
        }
    } else {
        if (tTop > 0.0) {
            rayLen = min(tTop, tBottom);
        }
    }

    params.rayEnd = params.rayStart + rayDir * rayLen;

    result = raymarchSingleScattering(atmosphere, params, scatteringParams);

    return result;
}

void main() {
    ivec2 imgSize = imageSize(uimg_skyLUT);
    ivec2 pixelPos = ivec2(gl_GlobalInvocationID.xy);

    if (all(lessThan(pixelPos, imgSize))) {
        vec2 texCoord = (pixelPos + 0.5) / vec2(imgSize);
        AtmosphereParameters atmosphere = getAtmosphereParameters();
        vec3 rayDir = coords_octDecode01(texCoord);

        ScatteringResult result = computeSingleScattering(atmosphere, rayDir);
        imageStore(uimg_skyLUT, pixelPos, vec4(result.inScattering * 4.0, 1.0));
    }
}