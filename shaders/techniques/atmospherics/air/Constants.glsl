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

#include "/Base.glsl"

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

#define EPIPOLAR_DATA_Y_SIZE 129
#define EPIPOLAR_SLICE_END_POINTS_V (0.5 / 129.0)
#define SLICE_SAMPLE_D16 8

#elif SETTING_SLICE_SAMPLES == 256

#define EPIPOLAR_DATA_Y_SIZE 257
#define EPIPOLAR_SLICE_END_POINTS_V (0.5 / 257.0)
#define SLICE_SAMPLE_D16 16

#elif SETTING_SLICE_SAMPLES == 512

#define EPIPOLAR_DATA_Y_SIZE 513
#define EPIPOLAR_SLICE_END_POINTS_V (0.5 / 513.0)
#define SLICE_SAMPLE_D16 32

#elif SETTING_SLICE_SAMPLES == 1024

#define EPIPOLAR_DATA_Y_SIZE 1025
#define EPIPOLAR_SLICE_END_POINTS_V (0.5 / 1025.0)
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

// [PRE99], see also https://www.desmos.com/calculator/2e43734e0c
vec3 atmosphere_mieCoefficientsPreetham(float turbidity) {
    const vec3 a0 = vec3(-0.00565597102796, -0.00796791817965, -0.0118034994284);
    const vec3 a1 = vec3(0.00568551066159, 0.00800953249887, 0.0118651459692);
    return a0 + a1 * turbidity;
}

AtmosphereParameters getAtmosphereParameters() {
    const float ATMOSPHERE_BOTTOM = 6378.137;
    const float ATMOSPHERE_TOP = ATMOSPHERE_BOTTOM + 100.0;

    const float MIE_HEIGHT = 1.2;

    // https://www.desmos.com/calculator/8zep6vmnxa
    // Already in km
    const vec3 RAYLEIGH_SCATTERING_BASE = vec3(0.00559495220371, 0.0117551946648, 0.02767445204); // CIE 1931 2 deg
    //    const vec3 RAYLEIGH_SCATTERING_BASE = vec3(0.00523321397326, 0.0127899562336, 0.0279251882303); // CIE 1964 2 deg
    //    const vec3 RAYLEIGH_SCATTERING_BASE = vec3(0.00472928809669, 0.0122555400301, 0.0282925884685); // CIE 2006 2 deg
    //    const vec3 RAYLEIGH_SCATTERING_BASE = vec3(0.00500767075505, 0.013021188889, 0.0280120803159); // CIE 2006 10 deg

    const vec3 RAYLEIGH_SCATTERING = RAYLEIGH_SCATTERING_BASE * SETTING_ATM_RAY_SCT_MUL;

    // Constants from [BRU08]
    //    const vec3 MIE_SCATTERING_BASE = vec3(2.10e-5) * 1000.0;

    // Constants from [HIL20]
    //    const vec3 MIE_SCATTERING_BASE = vec3(3.996e-6) * 1000.0;

    vec3 MIE_SCATTERING_BASE = atmosphere_mieCoefficientsPreetham(global_turbidity);
    vec3 MIE_SCATTERING = MIE_SCATTERING_BASE * SETTING_ATM_MIE_SCT_MUL;
    vec3 MIE_ABOSORPTION = MIE_SCATTERING_BASE * SETTING_ATM_MIE_ABS_MUL;

    const float MIE_PHASE_G = 0.7034;
    const float MIE_PHASE_E = 2500.0; // For Klein-Nishina phase function

    // https://www.desmos.com/calculator/fumphpur14
    // cm to km conversion
    const vec3 OZONE_ABOSORPTION_BASE = vec3(4.9799463143e-10, 3.0842607592e-10, -9.1714404502e-12) * 100000.0;
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