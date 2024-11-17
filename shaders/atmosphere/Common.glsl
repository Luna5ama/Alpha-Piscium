// Contains code adopted from: https://github.com/sebh/UnrealEngineSkyAtmosphere
// MIT License
// Copyright (c) 2020 Epic Games, Inc.
// You can find the full license text in /licenses/MIT.txt
#ifndef INCLUDE_atmosphere_Common.glsl
#define INCLUDE_atmosphere_Common.glsl

#include "../_Util.glsl"

// Every length is in KM!!!

struct AtmosphereParameters {
    float bottom;
    float top;
    vec3 sunDirection;

    float mieHeight;
    float ozoneCenter;
    float ozoneHalfWidth;

    vec3 rayleighScattering;
//    vec3 rayleighAbsorption;

    vec3 mieScattering;
    vec3 mieAbsorption;
    float miePhaseG;

//    vec3 ozoneScattering;
    vec3 ozoneAbsorption;
};

AtmosphereParameters getAtmosphereParameters() {
    const float ATMOSPHERE_BOTTOM = 6378.137;
    const float ATMOSPHERE_TOP = ATMOSPHERE_BOTTOM + 100.0;

    const float MIE_HEIGHT = 1.2;
    const float OZONE_CENTER = 25.0;
    const float OZONE_HALF_WIDTH = 15.0;
    
    // https://forums.flightsimulator.com/t/replace-the-atmosphere-parameters-with-more-accurate-ones-from-arpc/607603
    const vec3 RAYLEIGH_SCATTERING = vec3(6.602e-6, 1.239e-5, 2.940e-5);
//    const vec3 RAYLEIGH_SCATTERING = vec3(7.23392672713746e-6, 1.19997766911071e-5, 2.65861007515866e-5);
    const vec3 RAYLEIGH_ABOSORPTION = vec3(0.0);

    // A Scalable and Production Ready Sky and Atmosphere Rendering Technique
    // by Sébastien Hillaire, 2020
    // https://sebh.github.io/publications/egsr2020.pdf
    const vec3 MIE_SCATTERING = vec3(3.996e-6);
    const vec3 MIE_ABOSORPTION = vec3(4.4e-6);
    const float MIE_PHASE_G = 0.76;

    const vec3 OZONE_SCATTERING = vec3(0.0);
//    const vec3 OZONE_ABOSORPTION = vec3(0.650e-6, 1.881e-6, 0.085e-6);
    const vec3 OZONE_ABOSORPTION = vec3(2.341e-6, 1.54e-6, 2.193e-25);

    AtmosphereParameters atmosphere;
    atmosphere.bottom = ATMOSPHERE_BOTTOM;
    atmosphere.top = ATMOSPHERE_TOP;

    atmosphere.mieHeight = MIE_HEIGHT;
    atmosphere.ozoneCenter = OZONE_CENTER;
    atmosphere.ozoneHalfWidth = OZONE_HALF_WIDTH;

    atmosphere.rayleighScattering = RAYLEIGH_SCATTERING * 1000.0;
    //    atmosphere.rayleighAbsorption = RAYLEIGH_ABOSORPTION * 1000.0;

    atmosphere.mieScattering = MIE_SCATTERING * 1000.0;
    atmosphere.mieAbsorption = MIE_ABOSORPTION * 1000.0;
    atmosphere.miePhaseG = MIE_PHASE_G;

    //    atmosphere.ozoneScattering = OZONE_SCATTERING * 1000.0;
    atmosphere.ozoneAbsorption = OZONE_ABOSORPTION * 1000.0;

    return atmosphere;
}

#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
const vec2 TRANSMITTANCE_TEXTURE_SIZE = vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
const vec2 TRANSMITTANCE_TEXEL_SIZE = 1.0 / TRANSMITTANCE_TEXTURE_SIZE;

// Calculate the air density ratio at a given height(km) relative to sea level
// See https://www.desmos.com/calculator/homrt1shnb
float densityRayleigh(AtmosphereParameters atmosphere, float h) {
    const float a0 = 0.00947927584794;
    const float a1 = -0.138528179963;
    const float a2 = -0.00235619411773;
    const float c = 8.163265e-6;
    return exp2(a0 + a1 * h + a2 * h * h);
}

float densityMie(AtmosphereParameters atmosphere, float h) {
    return exp(-h / atmosphere.mieHeight);
}

float densityOzone(AtmosphereParameters atmosphere, float h) {
    return max(0.0, 1.0 - abs(h - atmosphere.ozoneCenter) / atmosphere.ozoneHalfWidth);
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

// https://github.com/sebh/UnrealEngineSkyAtmosphere/blob/master/Resources/RenderSkyCommon.hlsl
// Transmittance LUT function parameterisation from Bruneton 2017 https://github.com/ebruneton/precomputed_atmospheric_scattering
// uv in [0,1]
// viewZenithCosAngle in [-1,1]
// viewAltitude in [bottomRAdius, topRadius]
float fromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float fromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

float calcViewAltitude(AtmosphereParameters atmosphere, vec3 worldPos) {
    float viewAltitude = worldPos.y;
    viewAltitude /= SETTING_ATM_ALT_SCALE;
    viewAltitude += 1.0;
    viewAltitude += atmosphere.bottom;
    return viewAltitude;
}

float calcCosSunZenith(AtmosphereParameters atmosphere, vec3 sunDirection) {
    return dot(sunDirection, upPosition * 0.01);
}

void lutTransmittanceParamsToUv(AtmosphereParameters atmosphere, float viewAltitude, float cosSunZenith, out vec2 uv) {
    viewAltitude = clamp(viewAltitude, atmosphere.bottom + 0.0001, atmosphere.top - 0.0001);
    cosSunZenith = clamp(cosSunZenith, -1.0, 1.0);
    float H = sqrt(max(0.0, atmosphere.top * atmosphere.top - atmosphere.bottom * atmosphere.bottom));
    float rho = sqrt(max(0.0, viewAltitude * viewAltitude - atmosphere.bottom * atmosphere.bottom));

    float discriminant = viewAltitude * viewAltitude * (cosSunZenith * cosSunZenith - 1.0) + atmosphere.top * atmosphere.top;
    float d = max(0.0, (-viewAltitude * cosSunZenith + sqrt(discriminant)));// Distance to atmosphere boundary

    float d_min = atmosphere.top - viewAltitude;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;

    uv = vec2(x_mu, x_r);
    //uv = vec2(fromUnitToSubUvs(uv.x, TRANSMITTANCE_TEXTURE_WIDTH), fromUnitToSubUvs(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT)); // No real impact so off
}

void uvToLutTransmittanceParams(AtmosphereParameters atmosphere, out float viewAltitude, out float cosSunZenith, vec2 uv) {
    //uv = vec2(fromSubUvsToUnit(uv.x, TRANSMITTANCE_TEXTURE_WIDTH), fromSubUvsToUnit(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT)); // No real impact so off
    uv = clamp(uv, TRANSMITTANCE_TEXEL_SIZE, vec2(1.0 - TRANSMITTANCE_TEXEL_SIZE));
    float x_mu = uv.x;
    float x_r = uv.y;

    float H = sqrt(atmosphere.top * atmosphere.top - atmosphere.bottom * atmosphere.bottom);
    float rho = H * x_r;
    viewAltitude = sqrt(rho * rho + atmosphere.bottom * atmosphere.bottom);

    float d_min = atmosphere.top - viewAltitude;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    cosSunZenith = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * viewAltitude * d);
    cosSunZenith = clamp(cosSunZenith, -1.0, 1.0);
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

    vec3 opticalDepth = vec3(0.0);
    for (uint stepIndex = 0u; stepIndex < steps; stepIndex++) {
        float stepIndexF = float(stepIndex);
        vec3 samplePos = origin + (stepIndexF + 0.5) * stepDelta;
        float sampleHeight = length(samplePos) - atmosphere.bottom;

        float sampleDensityRayleigh = densityRayleigh(atmosphere, sampleHeight);
        float sampleDensityMie = densityMie(atmosphere, sampleHeight);
        float sampleDensityOzone = densityOzone(atmosphere, sampleHeight);

        vec3 sampleExtinction = vec3(0.0);
        vec3 rayleighExtinction = atmosphere.rayleighScattering;
        vec3 mieExtinction = atmosphere.mieScattering + atmosphere.mieAbsorption;
        vec3 ozoneExtinction = atmosphere.ozoneAbsorption;
        sampleExtinction += rayleighExtinction * sampleDensityRayleigh;
        sampleExtinction += mieExtinction * sampleDensityMie;
        sampleExtinction += ozoneExtinction * sampleDensityOzone;

        vec3 sampleOpticalDepth = sampleExtinction * stepLength;
        opticalDepth += sampleOpticalDepth;
    }
    result = exp(-opticalDepth);
    return result;
}

float rayleighPhase(float cosTheta) {
    float factor = 3.0f / (16.0f * PI_CONST);
    return factor * (1.0f + cosTheta * cosTheta);
}

// Cornette-Shanks phase function for Mie scattering
float miePhase(float cosTheta, float g) {
    float k = 3.0 / (8.0 * PI_CONST) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

struct ScatteringResult {
    vec3 transmittance;
    vec3 inScattering;
};

struct RaymarchParameters {
    vec3 origin;
    vec3 rayDir;
    float rayLen;
    float cosSunZenith;
    float rayleighPhase;
    float miePhase;
    uint steps;
};

ScatteringResult raymarchSingleScattering(AtmosphereParameters atmosphere, RaymarchParameters params, sampler2D usam_transmittanceLUT) {
    ScatteringResult result = ScatteringResult(vec3(1.0), vec3(0.0));

    float stepLength = params.rayLen / float(params.steps);
    vec3 stepDelta = params.rayDir * stepLength;

    vec3 opticalDepth = vec3(0.0);

    for (uint stepIndex = 0u; stepIndex < params.steps; stepIndex++) {
        float stepIndexF = float(stepIndex);
        vec3 samplePos = params.origin + (stepIndexF + 0.5) * stepDelta;
        float sampleHeight = length(samplePos) - atmosphere.bottom;

        float sampleDensityRayleigh = densityRayleigh(atmosphere, sampleHeight);
        float sampleDensityMie = densityMie(atmosphere, sampleHeight);
        float sampleDensityOzone = densityOzone(atmosphere, sampleHeight);

        vec3 rayleighExtinction = atmosphere.rayleighScattering;
        vec3 mieExtinction = atmosphere.mieScattering + atmosphere.mieAbsorption;
        vec3 ozoneExtinction = atmosphere.ozoneAbsorption;
        vec3 sampleExtinction = vec3(0.0);
        sampleExtinction += rayleighExtinction * sampleDensityRayleigh;
        sampleExtinction += mieExtinction * sampleDensityMie;
        sampleExtinction += ozoneExtinction * sampleDensityOzone;

        vec3 sampleOpticalDepth = sampleExtinction * stepLength;
        vec3 sampleTransmittance = exp(-sampleOpticalDepth);

        vec3 rayleighScattering = atmosphere.rayleighScattering;
        vec3 mieScattering = atmosphere.mieScattering;
        vec3 sampleScattering = vec3(0.0);
        sampleScattering += params.rayleighPhase * rayleighScattering * sampleDensityRayleigh;
        sampleScattering += params.miePhase * mieScattering * sampleDensityMie;

        // TODO: shadowed in-scattering
        float shadow = 1.0;

        float sampleAltitude = length(samplePos);
        vec2 tLUTUV;
        lutTransmittanceParamsToUv(atmosphere, sampleAltitude, params.cosSunZenith, tLUTUV);
        vec3 tSunToSample = texture(usam_transmittanceLUT, tLUTUV).rgb;
        vec3 tSampleToOrigin = exp(-opticalDepth);

        vec3 scattering = sampleScattering * tSunToSample;
        // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
        vec3 scatteringInt = (scattering - scattering * sampleTransmittance) / sampleExtinction;
        result.inScattering += tSampleToOrigin * scatteringInt;

        opticalDepth += sampleOpticalDepth;
    }
    result.transmittance = exp(-opticalDepth);
    return result;
}

#endif