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
LightParameters lightParams, \
float stepJitter

#elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 2
#define ATMOSPHERE_RAYMARCHING_FUNC_NAME raymarchSkySingle
#define ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE ScatteringResult
#define ATMOSPHERE_RAYMARCHING_FUNC_PARAMS \
AtmosphereParameters atmosphere, \
RaymarchParameters params, \
LightParameters lightParams, \
float bottomOffset

#elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 3
#define ATMOSPHERE_RAYMARCHING_FUNC_NAME raymarchSky
#define ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE ScatteringResult
#define ATMOSPHERE_RAYMARCHING_FUNC_PARAMS \
AtmosphereParameters atmosphere, \
RaymarchParameters params, \
ScatteringParameters scatteringParams

#elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
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
    const vec3 earthCenter = vec3(0.0);

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
    #elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
    vec3 shaodwStepDelta = (shadowEnd - shadowStart) * rcpSteps;
    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));

    vec3 samplePosMid = (params.rayEnd + params.rayStart) * 0.5;
    float sampleHeightMid = length(samplePosMid);
    vec3 upVectorMid = samplePosMid / sampleHeightMid;

    float cosZenithSun = dot(upVectorMid, scatteringParams.sunParams.lightDir);
    vec3 tSunToSampleMid = atmospherics_air_lut_sampleTransmittance(atmosphere, cosZenithSun, sampleHeightMid);
    vec3 earthShadowPos = vec3(0.0, 0.0, max(sampleHeightMid, atmosphere.bottom + PLANET_RADIUS_OFFSET));
    float tEarthSun = raySphereIntersectNearest(earthShadowPos, scatteringParams.sunParams.lightDir, earthCenter, atmosphere.bottom);
    tSunToSampleMid *= float(tEarthSun < 0.0);
    tSunToSampleMid *= scatteringParams.sunParams.irradiance;

    vec3 multiSctrLuminanceSun = atmospherics_air_lut_sampleMultiSctr(atmosphere, cosZenithSun, sampleHeightMid);
    multiSctrLuminanceSun *= scatteringParams.multiSctrFactor * scatteringParams.sunParams.irradiance;

    float cosZenithMoon = dot(upVectorMid, scatteringParams.moonParams.lightDir);
    vec3 tMoonToSampleMid = atmospherics_air_lut_sampleTransmittance(atmosphere, cosZenithMoon, sampleHeightMid);
    float tEarthMoon = raySphereIntersectNearest(earthShadowPos, scatteringParams.moonParams.lightDir, earthCenter, atmosphere.bottom);
    tMoonToSampleMid *= float(tEarthMoon < 0.0);
    tMoonToSampleMid *= scatteringParams.moonParams.irradiance;

    vec3 multiSctrLuminanceMoon = atmospherics_air_lut_sampleMultiSctr(atmosphere, cosZenithMoon, sampleHeightMid);
    multiSctrLuminanceMoon *= scatteringParams.multiSctrFactor * scatteringParams.moonParams.irradiance;

    #endif

    #endif

    for (uint stepIndex = 0u; stepIndex < params.steps; stepIndex++) {
        float stepIndexF = float(stepIndex);
        #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 0 || ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 1
        vec3 samplePos = params.rayStart + (stepIndexF + stepJitter) * rayStepDelta;
        #else
        vec3 samplePos = params.rayStart + (stepIndexF + 0.5) * rayStepDelta;
        #endif
        float sampleHeight = length(samplePos);
        #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 2
        sampleHeight = max(sampleHeight, atmosphere.bottom);
        #endif

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

        #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 0
        {
            totalDensity += sampleDensity * rayStepLength;
        }
        #elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 1
        {
            float cosZenith = dot(upVector, lightParams.lightDir);
            vec3 tLightToSample = atmospherics_air_lut_sampleTransmittance(atmosphere, cosZenith, sampleHeight);

            float tEarth = raySphereIntersectNearest(samplePos, lightParams.lightDir, earthCenter, atmosphere.bottom);
            float earthShadow = float(tEarth < 0.0);

            vec3 sampleInSctr = tLightToSample * computeTotalInSctr(atmosphere, lightParams, sampleDensity);
            // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
            vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
            totalInSctr += tSampleToOrigin * sampleInSctrInt;

            vec3 sampleMultiSctr = earthShadow * sampleTotalInSctr;
            vec3 sampleMultiSctrInt = (sampleMultiSctr - sampleMultiSctr * sampleTransmittance) / sampleExtinction;
            totalMultiSctrAs1 += tSampleToOrigin * sampleMultiSctrInt;
        }
        #elif ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 2
        {
            float cosZenith = dot(upVector, lightParams.lightDir);
            vec3 tLightToSample = atmospherics_air_lut_sampleTransmittance(atmosphere, cosZenith, sampleHeight);
            vec3 multiSctrLuminance = atmospherics_air_lut_sampleMultiSctr(atmosphere, cosZenith, sampleHeight);

            float tEarth = raySphereIntersectNearest(samplePos, lightParams.lightDir, earthCenter, atmosphere.bottom - bottomOffset);
            float earthShadow = float(tEarth < 0.0);

            vec3 sampleInSctr = earthShadow * tLightToSample * computeTotalInSctr(atmosphere, lightParams, sampleDensity);
            sampleInSctr += multiSctrLuminance * (sampleTotalInSctr);

            vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
            totalInSctr += tSampleToOrigin * sampleInSctrInt * lightParams.irradiance;
        }
        #else
        {
            #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
            float startShadowT = max(stepIndexF + stepJitter - 0.5, 0.0);
            float endShadowT = min(stepIndexF + stepJitter + 0.5, float(params.steps));
            vec3 startShadowPos = shadowStart + startShadowT * shaodwStepDelta;
            vec3 endShadowPos = shadowStart + endShadowT * shaodwStepDelta;
            float shadowSample = atmosphere_sample_shadow(startShadowPos, endShadowPos);
            #endif

            vec3 sampleInSctr = vec3(0.0);

            {
                #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
                float shadowTerm = mix(1.0, shadowSample, shadowIsSun);
                #else
                float tEarth = raySphereIntersectNearest(samplePos, scatteringParams.sunParams.lightDir, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom);
                float shadowTerm = float(tEarth < 0.0);
                #endif

                #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
                vec3 tSunToSample = tSunToSampleMid;
                #else
                float cosZenithSun = dot(upVector, scatteringParams.sunParams.lightDir);
                vec3 tSunToSample = atmospherics_air_lut_sampleTransmittance(atmosphere, cosZenithSun, sampleHeight);
                vec3 multiSctrLuminanceSun = atmospherics_air_lut_sampleMultiSctr(atmosphere, cosZenithSun, sampleHeight);
                multiSctrLuminanceSun *= scatteringParams.multiSctrFactor;
                #endif
                tSunToSample *= shadowTerm;

                vec3 sampleInSctrC = sampleRayleighInSctr * scatteringParams.sunParams.rayleighPhase;
                sampleInSctrC += sampleMieInSctr * scatteringParams.sunParams.miePhase;


                #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
                sampleInSctr += sampleInSctrC * tSunToSample;
                sampleInSctr += multiSctrLuminanceSun * sampleTotalInSctr;
                #else
                sampleInSctrC *= tSunToSample;
                sampleInSctrC += multiSctrLuminanceSun * sampleTotalInSctr;
                sampleInSctr += sampleInSctrC * scatteringParams.sunParams.irradiance;
                #endif
            }

            {
                #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
                float shadowTerm = mix(shadowSample, 1.0, shadowIsSun);
                #else
                float tEarth = raySphereIntersectNearest(samplePos, scatteringParams.moonParams.lightDir, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom);
                float shadowTerm = float(tEarth < 0.0);
                #endif

                #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
                vec3 tMoonToSample = tMoonToSampleMid;
                #else
                float cosZenithMoon = dot(upVector, scatteringParams.moonParams.lightDir);
                vec3 tMoonToSample = atmospherics_air_lut_sampleTransmittance(atmosphere, cosZenithMoon, sampleHeight);
                vec3 multiSctrLuminanceMoon = atmospherics_air_lut_sampleMultiSctr(atmosphere, cosZenithMoon, sampleHeight);
                multiSctrLuminanceMoon *= scatteringParams.multiSctrFactor;
                #endif

                tMoonToSample *= shadowTerm;
                vec3 sampleInSctrC = sampleRayleighInSctr * scatteringParams.moonParams.rayleighPhase;
                sampleInSctrC += sampleMieInSctr * scatteringParams.moonParams.miePhase;

                #if ATMOSPHERE_RAYMARCHING_FUNC_TYPE == 4
                sampleInSctr += sampleInSctrC * tMoonToSample;
                sampleInSctr += multiSctrLuminanceMoon * sampleTotalInSctr;
                #else
                sampleInSctrC *= tMoonToSample;
                sampleInSctrC += multiSctrLuminanceMoon * sampleTotalInSctr;
                sampleInSctr += sampleInSctrC * scatteringParams.moonParams.irradiance;
                #endif
            }

            // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
            vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
            totalInSctr += tSampleToOrigin * sampleInSctrInt;
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
    #else
    result.transmittance = tSampleToOrigin;
    result.inScattering = totalInSctr;
    #endif

    return result;
}

#undef ATMOSPHERE_RAYMARCHING_FUNC_NAME
#undef ATMOSPHERE_RAYMARCHING_FUNC_RESULT_TYPE
#undef ATMOSPHERE_RAYMARCHING_FUNC_PARAMS