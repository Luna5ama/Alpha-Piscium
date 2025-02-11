// References:
//     [BRU08] Bruneton, Eric. "Precomputed Atmospheric Scattering". EGSR 2008.
//         https://hal.inria.fr/inria-00290084/document
//     [HIL20] Hillaire, Sébastien. "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
//         EGSR 2020. https://sebh.github.io/publications/egsr2020.pdf
//     [YUS13] Yusov, Egor. “Practical Implementation of Light Scattering Effects Using Epipolar Sampling and
//         1D Min/Max Binary Trees”. GDC 2013.
//         http://gdcvault.com/play/1018227/Practical-Implementation-of-Light-Scattering
//
// Contains code adopted from:
// https://github.com/GameTechDev/OutdoorLightScattering
// Apache License 2.0
// Copyright (c) 2017 Intel Corporation
//
// Contains code adopted from:
// https://github.com/sebh/UnrealEngineSkyAtmosphere
// MIT License
// Copyright (c) 2020 Epic Games, Inc.
//
// You can find full license texts in /licenses
#ifndef INCLUDE_atmosphere_Common.glsl
#define INCLUDE_atmosphere_Common.glsl

#include "../_Util.glsl"

#define MULTI_SCTR_LUT_SIZE 32
#define PLANET_RADIUS_OFFSET 0.001

uniform sampler2D usam_transmittanceLUT;
uniform sampler2D usam_multiSctrLUT;

// Every length is in KM!!!

struct AtmosphereParameters {
    float bottom;
    float top;

    float mieHeight;
    float ozoneCenter;
    float ozoneHalfWidth;

    vec3 rayleighSctrCoeff;
    vec3 rayleighExtinction;

    vec3 mieSctrCoeff;
    vec3 mieExtinction;

    float miePhaseG;

    vec3 ozoneExtinction;
};

AtmosphereParameters getAtmosphereParameters() {
    const float ATMOSPHERE_BOTTOM = 6378.137;
    const float ATMOSPHERE_TOP = ATMOSPHERE_BOTTOM + 100.0;

    const float MIE_HEIGHT = 1.2;
    const float OZONE_CENTER = 25.0;
    const float OZONE_HALF_WIDTH = 15.0;

    // https://www.desmos.com/calculator/8zep6vmnxa
    const vec3 RAYLEIGH_SCATTERING = vec3(0.00559495220371, 0.0117551946648, 0.02767445204);

    // Constants from [HIL20]
//    const vec3 MIE_SCATTERING = vec3(3.996e-6) * 1000.0;
//    const vec3 MIE_ABOSORPTION = vec3(4.4e-6) * 1000.0;

    // Constants from [BRU08]
//        const vec3 MIE_SCATTERING = vec3(2.10e-5) * 1000.0;
//        const vec3 MIE_ABOSORPTION = MIE_SCATTERING * 1.1;

    const vec3 MIE_SCATTERING = vec3(2.10e-5) * 500.0;
    const vec3 MIE_ABOSORPTION = MIE_SCATTERING * 1.1;

    const float MIE_PHASE_G = 0.76;

    // https://www.desmos.com/calculator/rggs64tsru
    const vec3 OZONE_ABOSORPTION = vec3(5.3136330769e-10, 3.0767786886e-10, 5.0003555457e-12) * 100000.0;

    AtmosphereParameters atmosphere;
    atmosphere.bottom = ATMOSPHERE_BOTTOM;
    atmosphere.top = ATMOSPHERE_TOP;

    atmosphere.mieHeight = MIE_HEIGHT;
    atmosphere.ozoneCenter = OZONE_CENTER;
    atmosphere.ozoneHalfWidth = OZONE_HALF_WIDTH;

    atmosphere.rayleighSctrCoeff = RAYLEIGH_SCATTERING;
    atmosphere.rayleighExtinction = RAYLEIGH_SCATTERING;

    atmosphere.miePhaseG = MIE_PHASE_G;
    atmosphere.mieSctrCoeff = MIE_SCATTERING;
    atmosphere.mieExtinction = MIE_SCATTERING + MIE_ABOSORPTION;

    atmosphere.ozoneExtinction = OZONE_ABOSORPTION;

    return atmosphere;
}

#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
const vec2 TRANSMITTANCE_TEXTURE_SIZE = vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
const vec2 TRANSMITTANCE_TEXEL_SIZE = 1.0 / TRANSMITTANCE_TEXTURE_SIZE;

// Calculate the air density ratio at a given height(km) relative to sea level
// Fitted to U.S. Standard Atmosphere 1976
// See https://www.desmos.com/calculator/8zep6vmnxa
float sampleRayleighDensity(AtmosphereParameters atmosphere, float altitude) {
    const float a0 = 0.00947927584794;
    const float a1 = -0.138528179963;
    const float a2 = -0.00235619411773;
    return exp2(a0 + a1 * altitude + a2 * altitude * altitude);
}

float sampleMieDensity(AtmosphereParameters atmosphere, float altitude) {
    return exp(-altitude / atmosphere.mieHeight);
}

// Calculate the ozone number density in 10^17 molecules/m^3
// See https://www.desmos.com/calculator/rggs64tsru
float sampleOzoneDensity(AtmosphereParameters atmosphere, float altitude) {
    float x = max(altitude, 0.0);
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;

    const float d10 = 3.14463183277;
    const float d11 = 0.0498300739688;
    const float d12 = -0.130539505915;
    const float d13 = 0.021937805503;
    const float d14 = -0.000931031499441;
    float d1 = exp2(d10 + d11 * x + d12 * x2 + d13 * x3 + d14 * x4);

    const float d20 = -15.997595602;
    const float d21 = 2.79421136325;
    const float d22 = -0.128226752553;
    const float d23 = 0.00249280242791;
    const float d24 = -0.000018555830924;
    float d2 = exp2(d20 + d21 * x + d22 * x2 + d23 * x3 + d24 * x4);

    return d1 + d2;
}

vec3 sampleParticleDensity(AtmosphereParameters atmosphere, float height) {
    float altitude = height - atmosphere.bottom;
    return vec3(
        sampleRayleighDensity(atmosphere, altitude),
        sampleMieDensity(atmosphere, altitude),
        sampleOzoneDensity(atmosphere, altitude)
    );
}

// - r0: ray origin
// - rd: normalized ray direction
// - s0: sphere center
// - sR: sphere radius
// - Returns distance from r0 to first intersecion with sphere,
//   or -1.0 if no intersection.
float raySphereIntersectNearest(vec3 r0, vec3 rd, vec3 s0, float sR) {
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sR * sR);
    float delta = b * b - 4.0*a*c;
    if (delta < 0.0 || a == 0.0) {
        return -1.0;
    }
    float sol0 = (-b - sqrt(delta)) / (2.0*a);
    float sol1 = (-b + sqrt(delta)) / (2.0*a);
    if (sol0 < 0.0 && sol1 < 0.0) {
        return -1.0;
    }
    if (sol0 < 0.0) {
        return max(0.0, sol1);
    }
    else if (sol1 < 0.0) {
        return max(0.0, sol0);
    }
    return max(0.0, min(sol0, sol1));
}

// [HIL20] https://github.com/sebh/UnrealEngineSkyAtmosphere/blob/master/Resources/RenderSkyCommon.hlsl
// Transmittance LUT function parameterisation from Bruneton 2017 https://github.com/ebruneton/precomputed_atmospheric_scattering
// uv in [0,1]
// viewZenithCosAngle in [-1,1]
// viewAltitude in [bottomRAdius, topRadius]
float fromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float fromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

vec2 fromUnitToSubUvs(vec2 uv, vec2 resolution) { return (uv + 0.5 / resolution) * (resolution / (resolution + 1.0)); }
vec2 fromSubUvsToUnit(vec2 uv, vec2 resolution) { return (uv - 0.5 / resolution) * (resolution / (resolution - 1.0)); }

float atmosphere_height(AtmosphereParameters atmosphere, vec3 worldPos) {
    float worldHeight = max(worldPos.y - 62.0, float(SETTING_ATM_ALT_SCALE) * 0.001);
    return worldHeight * (1.0 / float(SETTING_ATM_ALT_SCALE)) + atmosphere.bottom;
}

vec3 atmosphere_viewToAtm(AtmosphereParameters atmosphere, vec3 viewPos) {
    vec3 feetPlayer = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 world = feetPlayer + cameraPosition;
    float height = atmosphere_height(atmosphere, world);
    return vec3(feetPlayer.x, 0.0, feetPlayer.z) * (1.0 / float(SETTING_ATM_D_SCALE)) + vec3(0.0, height, 0.0);
}

void lutTransmittanceParamsToUv(AtmosphereParameters atmosphere, float height, float cosZenith, out vec2 uv) {
    height = clamp(height, atmosphere.bottom + 0.0001, atmosphere.top - 0.0001);
    cosZenith = clamp(cosZenith, -1.0, 1.0);
    float H = sqrt(max(0.0, atmosphere.top * atmosphere.top - atmosphere.bottom * atmosphere.bottom));
    float rho = sqrt(max(0.0, height * height - atmosphere.bottom * atmosphere.bottom));

    float discriminant = height * height * (cosZenith * cosZenith - 1.0) + atmosphere.top * atmosphere.top;
    float d = max(0.0, (-height * cosZenith + sqrt(discriminant)));// Distance to atmosphere boundary

    float d_min = atmosphere.top - height;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;

    uv = vec2(x_mu, x_r);
    //uv = vec2(fromUnitToSubUvs(uv.x, TRANSMITTANCE_TEXTURE_WIDTH), fromUnitToSubUvs(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT)); // No real impact so off
}

void uvToLutTransmittanceParams(AtmosphereParameters atmosphere, out float altitude, out float cosZenith, vec2 uv) {
    //uv = vec2(fromSubUvsToUnit(uv.x, TRANSMITTANCE_TEXTURE_WIDTH), fromSubUvsToUnit(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT)); // No real impact so off
    uv = clamp(uv, TRANSMITTANCE_TEXEL_SIZE, vec2(1.0 - TRANSMITTANCE_TEXEL_SIZE));
    float x_mu = uv.x;
    float x_r = uv.y;

    float H = sqrt(atmosphere.top * atmosphere.top - atmosphere.bottom * atmosphere.bottom);
    float rho = H * x_r;
    altitude = sqrt(rho * rho + atmosphere.bottom * atmosphere.bottom);

    float d_min = atmosphere.top - altitude;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    cosZenith = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * altitude * d);
    cosZenith = clamp(cosZenith, -1.0, 1.0);
}

float rayleighPhase(float cosTheta) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks phase function for Mie scattering
float miePhase(float cosTheta, float g) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

struct ScatteringResult {
    vec3 transmittance;
    vec3 inScattering;
};

struct RaymarchParameters {
    vec3 rayStart;
    vec3 rayEnd;
    uint steps;
};

struct LightParameters {
    float cosZenith;
    float rayleighPhase;
    float miePhase;
    vec3 radiance;
};

void lightParameters_setup(AtmosphereParameters atmosphere, out LightParameters lightParams, vec3 radiance, vec3 lightDir, vec3 rayDir) {
    lightParams.radiance = radiance;
    lightParams.cosZenith = dot(lightDir, vec3(0.0, 1.0, 0.0));
    float cosLightTheta = -dot(rayDir, lightDir);
    lightParams.rayleighPhase = rayleighPhase(cosLightTheta);
    lightParams.miePhase = miePhase(cosLightTheta, atmosphere.miePhaseG);
}

bool setupRayEnd(AtmosphereParameters atmosphere, inout RaymarchParameters params, vec3 rayDir) {
    vec3 earthCenter = vec3(0.0);

    // Check if ray origin is outside the atmosphere
    if (length(params.rayStart) > atmosphere.top) {
        float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
        if (tTop < 0.0) {
            return false; // No intersection with atmosphere: stop right away
        }
        params.rayStart += rayDir * (tTop + PLANET_RADIUS_OFFSET);
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

vec3 sampleTransmittanceLUT(AtmosphereParameters atmosphere, float cosLightZenith, float sampleAltitude) {
    vec2 tLUTUV;
    lutTransmittanceParamsToUv(atmosphere, sampleAltitude, cosLightZenith, tLUTUV);
    uint cond = uint(any(lessThan(tLUTUV, vec2(0.0))));
    cond |= uint(any(greaterThan(tLUTUV, vec2(1.0))));
    if (bool(cond)) {
        return vec3(0.0);
    }
    return texture(usam_transmittanceLUT, tLUTUV).rgb;
}

vec3 sampleMultiSctrLUT(AtmosphereParameters atmosphere, float cosLightZenith, float sampleAltitude) {
    vec2 uv = saturate(vec2(cosLightZenith * 0.5 + 0.5, sampleAltitude / (atmosphere.top - atmosphere.bottom)));
    uv = fromUnitToSubUvs(uv, vec2(MULTI_SCTR_LUT_SIZE));
    return texture(usam_multiSctrLUT, uv).rgb;
}

void unpackEpipolarData(uvec4 epipolarData, out ScatteringResult sctrResult, out float viewZ) {
    vec2 unpacked1 = unpackHalf2x16(epipolarData.x);
    vec2 unpacked2 = unpackHalf2x16(epipolarData.y);
    vec2 unpacked3 = unpackHalf2x16(epipolarData.z);
    sctrResult.inScattering = vec3(unpacked1.xy, unpacked2.x);
    sctrResult.transmittance = vec3(unpacked2.y, unpacked3.xy);
    viewZ = uintBitsToFloat(epipolarData.w);
}

void packEpipolarData(out uvec4 epipolarData, ScatteringResult sctrResult, float viewZ) {
    epipolarData.x = packHalf2x16(sctrResult.inScattering.xy);
    epipolarData.y = packHalf2x16(vec2(sctrResult.inScattering.z, sctrResult.transmittance.x));
    epipolarData.z = packHalf2x16(sctrResult.transmittance.yz);
    epipolarData.w = floatBitsToUint(viewZ);
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

ScatteringResult raymarchSingleScattering(
AtmosphereParameters atmosphere, RaymarchParameters params, LightParameters sunParams, LightParameters moonParams
) {
    ScatteringResult result = ScatteringResult(vec3(1.0), vec3(0.0));

    float rcpSteps = 1.0 / float(params.steps);
    vec3 rayStepDelta = (params.rayEnd - params.rayStart) * rcpSteps;
    float rayStepLength = length(params.rayEnd - params.rayStart) * rcpSteps;

    vec3 totalInSctr = vec3(0.0);
    vec3 tSampleToOrigin = vec3(1.0);

    for (uint stepIndex = 0u; stepIndex < params.steps; stepIndex++) {
        float stepIndexF = float(stepIndex) + 0.5;
        vec3 samplePos = params.rayStart + stepIndexF * rayStepDelta;
        float sampleHeight = length(samplePos);

        vec3 sampleDensity = sampleParticleDensity(atmosphere, sampleHeight);
        vec3 sampleExtinction = computeOpticalDepth(atmosphere, sampleDensity);
        vec3 sampleOpticalDepth = sampleExtinction * rayStepLength;
        vec3 sampleTransmittance = exp(-sampleOpticalDepth);

        vec3 rayleighInSctr = sampleDensity.x * atmosphere.rayleighSctrCoeff;
        vec3 mieInSctr = sampleDensity.y * atmosphere.mieSctrCoeff;

        {
            vec3 tSunToSample = sampleTransmittanceLUT(atmosphere, sunParams.cosZenith, sampleHeight);
            vec3 multiSctrLuminance = sampleMultiSctrLUT(atmosphere, sunParams.cosZenith, sampleHeight);

            vec3 sampleInSctr = tSunToSample * computeTotalInSctr(atmosphere, sunParams, sampleDensity);
            sampleInSctr += multiSctrLuminance * (rayleighInSctr + mieInSctr);

            // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
            vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
            totalInSctr += tSampleToOrigin * sampleInSctrInt * sunParams.radiance;
        }

        {
            vec3 tMoonToSample = sampleTransmittanceLUT(atmosphere, moonParams.cosZenith, sampleHeight);
            vec3 multiSctrLuminance = sampleMultiSctrLUT(atmosphere, moonParams.cosZenith, sampleHeight);

            vec3 sampleInSctr = tMoonToSample * computeTotalInSctr(atmosphere, moonParams, sampleDensity);
            sampleInSctr += multiSctrLuminance * (rayleighInSctr + mieInSctr);

            vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
            totalInSctr += tSampleToOrigin * sampleInSctrInt * moonParams.radiance;
        }

        tSampleToOrigin *= sampleTransmittance;
    }

    result.transmittance = tSampleToOrigin;
    result.inScattering = totalInSctr;

    return result;
}

vec3 raymarchTransmittance(AtmosphereParameters atmosphere, vec3 origin, vec3 dir, uint steps) {
    vec3 result = vec3(1.0);

    vec3 earthCenter = vec3(0.0, 0.0, 0.0);
    float tBottom = raySphereIntersectNearest(origin, dir, earthCenter, atmosphere.bottom);
    float tTop = raySphereIntersectNearest(origin, dir, earthCenter, atmosphere.top);
    float rayLenAtm = 0.0;
    if (tBottom < 0.0) {
        if (tTop < 0.0) {
            rayLenAtm = 0.0;// No intersection with earth nor atmosphere: stop right away
            return result;
        } else {
            rayLenAtm = tTop;
        }
    } else {
        if (tTop > 0.0) {
            rayLenAtm = min(tTop, tBottom);
        }
    }

    float stepLength = rayLenAtm / float(steps);
    vec3 stepDelta = dir * stepLength;

    vec3 totalDensity = vec3(0.0);

    for (uint stepIndex = 0u; stepIndex < steps; stepIndex++) {
        float stepIndexF = float(stepIndex);
        vec3 samplePos = origin + (stepIndexF + 0.5) * stepDelta;
        float sampleHeight = length(samplePos);
        vec3 sampleDensity = sampleParticleDensity(atmosphere, sampleHeight) * stepLength;
        totalDensity += sampleDensity;
    }

    vec3 totalOpticalDepth = computeOpticalDepth(atmosphere, totalDensity);
    result = exp(-totalOpticalDepth);

    return result;
}

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

#define SLICE_SAMPLE_D16 8

#elif SETTING_SLICE_SAMPLES == 256

#define SLICE_SAMPLE_D16 16

#elif SETTING_SLICE_SAMPLES == 512

#define SLICE_SAMPLE_D16 32

#elif SETTING_SLICE_SAMPLES == 1024

#define SLICE_SAMPLE_D16 64

#endif


#define INVALID_EPIPOLAR_LINE vec4(-1000.0, -1000.0, -100.0, -100.0)

bool isValidScreenLocation(vec2 f2XY) {
    const float SAFETY_EPSILON = 0.2f;
    return all(lessThanEqual(abs(f2XY), 1.0 - (1.0 - SAFETY_EPSILON) / vec2(global_mainImageSizeI)));
}

vec4 getOutermostScreenPixelCoords() {
    // The outermost visible screen pixels centers do not lie exactly on the boundary (+1 or -1), but are biased by
    // 0.5 screen pixel size inwards
    //
    //                                        2.0
    //    |<---------------------------------------------------------------------->|
    //
    //       2.0/Res
    //    |<--------->|
    //    |     X     |      X     |     X     |    ...    |     X     |     X     |
    //   -1     |                                                            |    +1
    //          |                                                            |
    //          |                                                            |
    //      -1 + 1.0/Res                                                  +1 - 1.0/Res
    //
    // Using shader macro is much more efficient than using constant buffer variable
    // because the compiler is able to optimize the code more aggressively
    return vec4(-1.0, -1.0, 1.0, 1.0) + vec4(1.0, 1.0, -1.0, -1.0) / global_mainImageSizeI.xyxy;
}

#endif
