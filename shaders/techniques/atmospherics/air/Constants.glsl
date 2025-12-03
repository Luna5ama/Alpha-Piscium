/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere
        [HIL20] Hillaire, Sébastien. "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
            EGSR 2020. 2020.
            https://sebh.github.io/publications/egsr2020.pdf
        [INT17] Intel Corporation. "Outdoor Light Scattering Sample". 2017.
            Apache License 2.0. Copyright (c) 2017 Intel Corporation.
            https://github.com/GameTechDev/OutdoorLightScattering
        [PRE99] Preetham, A. J. et al. "A Practical Analytic Model for Daylight"
            SIGGRAPH 1999. 1999.
            https://courses.cs.duke.edu/fall01/cps124/resources/p91-preetham.pdf

        You can find full license texts in /licenses

    Credits:
        Jessie - Helping with Mie coefficients (https://github.com/Jessie-LC)
*/
#ifndef INCLUDE_atmosphere_Constants_glsl
#define INCLUDE_atmosphere_Constants_glsl a

#include "/util/Colors2.glsl"

#if SETTING_EPIPOLAR_SLICES == 256

#define EPIPOLAR_SLICE_D16 16
#define EPIPOLAR_SLICE_D128 2

#elif SETTING_EPIPOLAR_SLICES == 512

#define EPIPOLAR_SLICE_D16 32
#define EPIPOLAR_SLICE_D128 4

#elif SETTING_EPIPOLAR_SLICES == 1024

#define EPIPOLAR_SLICE_D16 64
#define EPIPOLAR_SLICE_D128 8

#elif SETTING_EPIPOLAR_SLICES == 2048

#define EPIPOLAR_SLICE_D16 128
#define EPIPOLAR_SLICE_D128 16

#endif

#if SETTING_SLICE_SAMPLES == 128

#define EPIPOLAR_DATA_Y_SIZE 385
#define SLICE_SAMPLE_D16 8

#elif SETTING_SLICE_SAMPLES == 256

#define EPIPOLAR_DATA_Y_SIZE 769
#define SLICE_SAMPLE_D16 16

#elif SETTING_SLICE_SAMPLES == 512

#define EPIPOLAR_DATA_Y_SIZE 1537
#define SLICE_SAMPLE_D16 32

#elif SETTING_SLICE_SAMPLES == 1024

#define EPIPOLAR_DATA_Y_SIZE 3073
#define SLICE_SAMPLE_D16 64

#endif

#define MULTI_SCTR_LUT_SIZE 32
#define PLANET_RADIUS_OFFSET 0.01

// Every length is in KM!!!

struct AtmosphereParameters {
    float bottom;
    float top;

    float mieHeight;

    vec3 rayleighSctrCoeff;
    vec3 rayleighExtinction;

    vec3 mieSctrCoeff;
    vec3 mieExtinction;

    float miePhaseG;
    float miePhaseE;

    vec3 ozoneExtinction;
};

// [PRE99], see also https://www.desmos.com/calculator/giz0uiar7k
vec3 atmosphere_mieCoefficientsPreetham(float turbidity) {
    const vec3 a0 = vec3(-0.00767542206226, -0.00822772032997, -0.0121707541321);
    const vec3 a1 = vec3(0.00771550875198, 0.00827069152678, 0.0122343187466);
    return colors2_constants_toWorkSpace(a0 + a1 * turbidity);
}

AtmosphereParameters getAtmosphereParameters() {
    const float ATMOSPHERE_BOTTOM = 6378.137;
    const float ATMOSPHERE_TOP = ATMOSPHERE_BOTTOM + 100.0;

    const float MIE_HEIGHT = 1.2;

    // https://www.desmos.com/calculator/1qbdlareew
    // Already in km
    const vec3 RAYLEIGH_SCATTERING_BASE = colors2_constants_toWorkSpace(vec3(0.0120766817597, 0.0129498634753, 0.0275704559807));

    vec3 RAYLEIGH_SCATTERING = RAYLEIGH_SCATTERING_BASE * SETTING_ATM_RAY_SCT_MUL;

    // Constants from [BRU08]
    //    const vec3 MIE_SCATTERING_BASE = vec3(2.10e-5) * 1000.0;

    // Constants from [HIL20]
    //    const vec3 MIE_SCATTERING_BASE = vec3(3.996e-6) * 1000.0;

    vec3 MIE_SCATTERING_BASE = atmosphere_mieCoefficientsPreetham(global_turbidity);
    vec3 MIE_SCATTERING = MIE_SCATTERING_BASE * SETTING_ATM_MIE_SCT_MUL;
    vec3 MIE_ABOSORPTION = MIE_SCATTERING_BASE * SETTING_ATM_MIE_ABS_MUL;

    const float MIE_PHASE_G = 0.7034;
    const float MIE_PHASE_E = 2500.0; // For Klein-Nishina phase function

    // https://www.desmos.com/calculator/ykoihjoqdm
    // cm to km conversion
    const vec3 OZONE_ABOSORPTION_BASE = colors2_constants_toWorkSpace(vec3(3.2964135827e-10, 2.9538443418e-10, 3.7326149468e-11) * 100000.0);
    const vec3 OZONE_ABOSORPTION = OZONE_ABOSORPTION_BASE * SETTING_ATM_OZO_ABS_MUL;

    AtmosphereParameters atmosphere;
    atmosphere.bottom = ATMOSPHERE_BOTTOM;
    atmosphere.top = ATMOSPHERE_TOP;

    atmosphere.mieHeight = MIE_HEIGHT;

    atmosphere.rayleighSctrCoeff = RAYLEIGH_SCATTERING;
    atmosphere.rayleighExtinction = RAYLEIGH_SCATTERING;

    atmosphere.miePhaseG = MIE_PHASE_G;
    atmosphere.miePhaseE = MIE_PHASE_E;
    atmosphere.mieSctrCoeff = MIE_SCATTERING;
    atmosphere.mieExtinction = MIE_SCATTERING + MIE_ABOSORPTION;

    atmosphere.ozoneExtinction = OZONE_ABOSORPTION;

    return atmosphere;
}

#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
const vec2 TRANSMITTANCE_TEXTURE_SIZE = vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
const vec2 TRANSMITTANCE_TEXEL_SIZE = 1.0 / TRANSMITTANCE_TEXTURE_SIZE;

#endif