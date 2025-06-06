#ifndef INCLUDE_atmosphere_RaymarchingBase_glsl
#define INCLUDE_atmosphere_RaymarchingBase_glsl a

#include "Common.glsl"

struct MultiScatteringResult {
    vec3 inScattering;
    vec3 multiSctrAs1;
};

MultiScatteringResult multiScatteringResult_init() {
    MultiScatteringResult result;
    result.inScattering = vec3(0.0);
    result.multiSctrAs1 = vec3(0.0);
    return result;
}

struct RaymarchParameters {
    vec3 rayStart;
    vec3 rayEnd;
    float stepJitter;
    uint steps;
};

RaymarchParameters raymarchParameters_init() {
    RaymarchParameters params;
    params.rayStart = vec3(0.0);
    params.rayEnd = vec3(0.0);
    params.stepJitter = 0.5;
    params.steps = 0u;
    return params;
}

bool setupRayEnd(AtmosphereParameters atmosphere, inout RaymarchParameters params, vec3 rayDir) {
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

    float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom);
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

struct LightParameters {
    vec3 lightDir;
    float rayleighPhase;
    float miePhase;
    vec3 irradiance;
};

LightParameters lightParameters_init(AtmosphereParameters atmosphere, vec3 irradiance, vec3 lightDir, vec3 rayDir) {
    LightParameters lightParams;
    lightParams.irradiance = irradiance * PI;
    lightParams.lightDir = lightDir;
    float cosLightTheta = -dot(rayDir, lightDir);
    lightParams.rayleighPhase = rayleighPhase(cosLightTheta);
    lightParams.miePhase = miePhase(cosLightTheta, atmosphere.miePhaseG);
    return lightParams;
}

struct ScatteringParameters {
    LightParameters sunParams;
    LightParameters moonParams;
    float multiSctrFactor;
};

ScatteringParameters scatteringParameters_init(LightParameters sunParams, LightParameters moonParams, float multiSctrFactor) {
    ScatteringParameters params;
    params.sunParams = sunParams;
    params.moonParams = moonParams;
    params.multiSctrFactor = multiSctrFactor;
    return params;
}

vec3 computeOpticalDepth(AtmosphereParameters atmosphere, vec3 density) {
    vec3 result = vec3(0.0);
    result += atmosphere.rayleighExtinction * density.x;
    result += atmosphere.mieExtinction * density.y;
    result += atmosphere.ozoneExtinction * density.z;
    return result;
}

vec3 computeTotalInSctr(AtmosphereParameters atmosphere, LightParameters lightParams, vec3 sampleDensity) {
    vec3 rayleighInSctr = (sampleDensity.x * lightParams.rayleighPhase) * atmosphere.rayleighSctrCoeff;
    vec3 mieInSctr = (sampleDensity.y * lightParams.miePhase) * atmosphere.mieSctrCoeff;
    return rayleighInSctr + mieInSctr;
}

#endif