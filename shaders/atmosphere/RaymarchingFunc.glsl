#include "Common.glsl"
#include "RaymarchingBase.glsl"
// Type 0: transmittance lut
// Type 1: multi-scattering lut
// Type 2: sky
// Type 3: aerial perspective

#if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 0
#define ATMOSPHERE_RAYMARCHING_FUNC_NAME raymarchTransmittance
#define ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE vec3
#define ATMOSPHERE_RAYMARCHING_FUNC_PARAMS \
AtmosphereParameters atmosphere, \
RaymarchParameters params, \
float stepJitter

#elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 1
#define ATMOSPHERE_RAYMARCHING_FUNC_NAME raymarchMultiScattering
#define ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE MultiScatteringResult
#define ATMOSPHERE_RAYMARCHING_FUNC_PARAMS \
AtmosphereParameters atmosphere, \
RaymarchParameters params, \
LightParameters lightParams

#elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 2
#define ATMOSPHERE_RAYMARCHING_FUNC_NAME raymarchSky
#define ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE ScatteringResult
#define ATMOSPHERE_RAYMARCHING_FUNC_PARAMS \
AtmosphereParameters atmosphere, \
RaymarchParameters params, \
ScatteringParameters scatteringParams

#elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 3
#define ATMOSPHERE_RAYMARCHING_FUNC_NAME raymarchAerialPerspective
#define ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE ScatteringResult
#define ATMOSPHERE_RAYMARCHING_FUNC_PARAMS \
AtmosphereParameters atmosphere, \
RaymarchParameters params, \
ScatteringParameters scatteringParams, \
vec3 shadowStart, \
vec3 shadowEnd, \
float stepJitter

#endif

ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE ATMOSPHERE_RAYMARCHING_FUNC_NAME(ATMOSPHERE_RAYMARCHING_FUNC_PARAMS) {
    float rcpSteps = 1.0 / float(params.steps);
    vec3 rayStepDelta = (params.rayEnd - params.rayStart) * rcpSteps;
    float rayStepLength = length(rayStepDelta);

    #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 0
    vec3 totalDensity = vec3(0.0);
    #else
    vec3 totalInSctr = vec3(0.0);
    vec3 tSampleToOrigin = vec3(1.0);
    #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 1
    vec3 totalMultiSctrAs1 = vec3(0.0);
    #elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 3
    vec3 shaodwStepDelta = (shadowEnd - shadowStart) * rcpSteps;
    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));
    #endif
    #endif

    for (uint stepIndex = 0u; stepIndex < params.steps; stepIndex++) {
        float stepIndexF = float(stepIndex);
        #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 0
        vec3 samplePos = params.rayStart + (stepIndexF + stepJitter) * rayStepDelta;
        #else
        vec3 samplePos = params.rayStart + stepIndexF * rayStepDelta;
        #endif
        float sampleHeight = length(samplePos);

        vec3 sampleDensity = sampleParticleDensity(atmosphere, sampleHeight);

        #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE != 0
        vec3 upVector = samplePos / sampleHeight;

        vec3 sampleExtinction = computeOpticalDepth(atmosphere, sampleDensity);
        vec3 sampleOpticalDepth = sampleExtinction * rayStepLength;
        vec3 sampleTransmittance = exp(-sampleOpticalDepth);

        vec3 sampleRayleighInSctr = sampleDensity.x * atmosphere.rayleighSctrCoeff;
        vec3 sampleMieInSctr = sampleDensity.y * atmosphere.mieSctrCoeff;
        vec3 sampleTotalInSctr = sampleRayleighInSctr + sampleMieInSctr;
        #endif

        const vec3 earthCenter = vec3(0.0);

        #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 0
        {
            totalDensity += sampleDensity * rayStepLength;
        }
        #elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 1
        {
            float cosZenith = dot(upVector, lightParams.lightDir);
            vec3 tLightToSample = sampleTransmittanceLUT(atmosphere, cosZenith, sampleHeight);

            float tEarth = raySphereIntersectNearest(samplePos, lightParams.lightDir, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom);
            float earthShadow = float(tEarth < 0.0);

            vec3 sampleInSctr = tLightToSample * computeTotalInSctr(atmosphere, lightParams, sampleDensity);
            // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
            vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
            totalInSctr += tSampleToOrigin * sampleInSctrInt;

            vec3 sampleMultiSctr = earthShadow * sampleTotalInSctr;
            vec3 sampleMultiSctrInt = (sampleMultiSctr - sampleMultiSctr * sampleTransmittance) / sampleExtinction;
            totalMultiSctrAs1 += tSampleToOrigin * sampleMultiSctrInt;
        }
        #else
        {
            #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 3
            vec3 sampleShadowPos = shadowStart + (stepIndexF + stepJitter) * shaodwStepDelta;
            float shadowSample = atmosphere_sample_shadow(sampleShadowPos);
            #endif

            {
                float cosZenith = dot(upVector, scatteringParams.sunParams.lightDir);
                vec3 tSunToSample = sampleTransmittanceLUT(atmosphere, cosZenith, sampleHeight);
                vec3 multiSctrLuminance = sampleMultiSctrLUT(atmosphere, cosZenith, sampleHeight);

                #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 3
                float shadow = mix(1.0, shadowSample, shadowIsSun);
                #else
                float shadow = 1.0;
                #endif

                float tEarth = raySphereIntersectNearest(samplePos, scatteringParams.sunParams.lightDir, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom);
                float earthShadow = float(tEarth < 0.0);

                vec3 sampleInSctr = earthShadow * shadow * tSunToSample * computeTotalInSctr(atmosphere, scatteringParams.sunParams, sampleDensity);
                sampleInSctr += scatteringParams.multiSctrFactor * multiSctrLuminance * (sampleTotalInSctr);

                // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
                vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
                totalInSctr += tSampleToOrigin * sampleInSctrInt * scatteringParams.sunParams.irradiance;
            }

            {
                float cosZenith = dot(upVector, scatteringParams.moonParams.lightDir);
                vec3 tMoonToSample = sampleTransmittanceLUT(atmosphere, cosZenith, sampleHeight);
                vec3 multiSctrLuminance = sampleMultiSctrLUT(atmosphere, cosZenith, sampleHeight);

                #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 3
                float shadow = mix(shadowSample, 1.0, shadowIsSun);
                #else
                float shadow = 1.0;
                #endif

                float tEarth = raySphereIntersectNearest(samplePos, scatteringParams.moonParams.lightDir, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom);
                float earthShadow = float(tEarth < 0.0);

                vec3 sampleInSctr = earthShadow * shadow * tMoonToSample * computeTotalInSctr(atmosphere, scatteringParams.moonParams, sampleDensity);
                sampleInSctr += scatteringParams.multiSctrFactor * multiSctrLuminance * (sampleTotalInSctr);

                vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
                totalInSctr += tSampleToOrigin * sampleInSctrInt * scatteringParams.moonParams.irradiance;
            }
        }
        #endif

        #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE != 0
        tSampleToOrigin *= sampleTransmittance;
        #endif
    }

    ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE result;

    #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 0
    vec3 totalOpticalDepth = computeOpticalDepth(atmosphere, totalDensity);
    result = exp(-totalOpticalDepth);
    #elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 1
    result.inScattering = totalInSctr;
    result.multiSctrAs1 = totalMultiSctrAs1;
    #else ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 2
    result.transmittance = tSampleToOrigin;
    result.inScattering = totalInSctr;
    #endif

    return result;
}

#undef ATMOSPHERE_RAYMARCHING_FUNC_NAME
#undef ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE
#undef ATMOSPHERE_RAYMARCHING_FUNC_PARAMS