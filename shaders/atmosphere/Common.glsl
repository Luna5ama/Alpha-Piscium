/*
    References:
        [BRU08] Bruneton, Eric. "Precomputed Atmospheric Scattering". EGSR 2008.
            https://hal.inria.fr/inria-00290084/document
        [HIL20] Hillaire, Sébastien. "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
            EGSR 2020. https://sebh.github.io/publications/egsr2020.pdf
            Code: https://github.com/sebh/UnrealEngineSkyAtmosphere (MIT License)
        [YUS13] Yusov, Egor. “Practical Implementation of Light Scattering Effects Using Epipolar Sampling and
            1D Min/Max Binary Trees”. GDC 2013.
            http://gdcvault.com/play/1018227/Practical-Implementation-of-Light-Scattering
            Code: https://github.com/GameTechDev/OutdoorLightScattering (Apache-2.0 License)
        [BIO23] Biology2394. “Replace the atmosphere parameters with more accurate ones from ARPC”. 2023.
            https://forums.flightsimulator.com/t/replace-the-atmosphere-parameters-with-more-accurate-ones-from-arpc/607603

    You can find full license texts in /licenses
*/
#ifndef INCLUDE_atmosphere_Common.glsl
#define INCLUDE_atmosphere_Common.glsl

#include "../_Util.glsl"
#include "Epipolar.glsl"

#define TRAPEZOIDAL_INTEGRATION 1

// Every length is in KM!!!

struct AtmosphereParameters {
    float bottom;
    float top;

    float mieHeight;
    float ozoneCenter;
    float ozoneHalfWidth;

    vec3 rayleighSctrCoeffTotal;
// Angular Rayleigh scattering coefficient contains all the terms exepting 1 + cos^2(Theta):
// See [YUS13] https://github.com/GameTechDev/OutdoorLightScattering/blob/master/fx/Structures.fxh
    vec3 rayleighSctrCoeffAngular;
    vec3 rayleighExtinction;

    vec3 mieSctrCoeffTotal;
    vec3 mieSctrCoeffAngular;
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

    // https://www.desmos.com/calculator/ugi2cb8qyj
    const vec3 RAYLEIGH_SCATTERING = vec3(0.00000718336687547, 0.0000119270553606, 0.0000264247319705);

    // Constants from [HIL20]
    const vec3 MIE_SCATTERING = vec3(3.996e-6);
    const vec3 MIE_ABOSORPTION = vec3(4.4e-6);

    // Constants from [BRU08]
    //    const vec3 MIE_SCATTERING = vec3(2.10e-5);
    //    const vec3 MIE_ABOSORPTION = MIE_SCATTERING * 1.1;

    const float MIE_PHASE_G = 0.76;

    // https://www.desmos.com/calculator/ntnh7wwbd8
    const vec3 OZONE_ABOSORPTION = vec3(6.2240628315e-16, 2.6898614109e-16, -2.2351831487e-18);

    AtmosphereParameters atmosphere;
    atmosphere.bottom = ATMOSPHERE_BOTTOM;
    atmosphere.top = ATMOSPHERE_TOP;

    atmosphere.mieHeight = MIE_HEIGHT;
    atmosphere.ozoneCenter = OZONE_CENTER;
    atmosphere.ozoneHalfWidth = OZONE_HALF_WIDTH;

    atmosphere.rayleighSctrCoeffTotal = RAYLEIGH_SCATTERING * 1000.0;
    atmosphere.rayleighSctrCoeffAngular = atmosphere.rayleighSctrCoeffTotal * (3.0 / (16.0 * PI));
    atmosphere.rayleighExtinction = atmosphere.rayleighSctrCoeffTotal;

    atmosphere.miePhaseG = MIE_PHASE_G;
    const float k = 3.0 / (8.0 * PI) * (1.0 - atmosphere.miePhaseG * atmosphere.miePhaseG) / (2.0 + atmosphere.miePhaseG * atmosphere.miePhaseG);
    atmosphere.mieSctrCoeffTotal = MIE_SCATTERING * 1000.0;
    atmosphere.mieSctrCoeffAngular = atmosphere.mieSctrCoeffTotal * k;
    atmosphere.mieExtinction = atmosphere.mieSctrCoeffTotal + (MIE_ABOSORPTION * 1000.0);

    atmosphere.ozoneExtinction = OZONE_ABOSORPTION;

    return atmosphere;
}

#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
const vec2 TRANSMITTANCE_TEXTURE_SIZE = vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
const vec2 TRANSMITTANCE_TEXEL_SIZE = 1.0 / TRANSMITTANCE_TEXTURE_SIZE;

// Calculate the air density ratio at a given height(km) relative to sea level
// Fitted to U.S. Standard Atmosphere 1976
// See https://www.desmos.com/calculator/ugi2cb8qyj
float sampleRayleighDensity(AtmosphereParameters atmosphere, float altitude) {
    const float a0 = 0.00947927584794;
    const float a1 = -0.138528179963;
    const float a2 = -0.00235619411773;
    return exp2(a0 + a1 * altitude + a2 * altitude * altitude);
}

float sampleMieDensity(AtmosphereParameters atmosphere, float altitude) {
    return exp(-altitude / atmosphere.mieHeight);
}

// Calculate the ozone number density in molecules/cm^3
// See https://www.desmos.com/calculator/ntnh7wwbd8
float sampleOzoneDensity(AtmosphereParameters atmosphere, float altitude) {
    float x = max(altitude, 0.0);
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;

    const float d10 = 5.07148407148;
    const float d11 = -0.713837088837;
    const float d12 = 0.0518405205905;
    const float d13 = -0.00185347060347;
    const float d14 = 0.0000546328671329;
    float d1 = d10 + d11 * x + d12 * x2 + d13 * x3 + d14 * x4;

    const float d20 = -21.150982082;
    const float d21 = 3.16169710887;
    const float d22 = -0.141187419433;
    const float d23 = 0.00261622221548;
    const float d24 = -0.0000175765474848;
    float d2 = exp2(d20 + d21 * x + d22 * x2 + d23 * x3 + d24 * x4);

    const float l0 = 14.5;
    const float l1 = 18.5;
    float s = smoothstep(l0, l1, x);

    float mPa = mix(d1, d2, s);
    return mPa * 3.260396361e11;
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
    float factor = 3.0 / (16.0 * PI);
    return factor * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks phase function for Mie scattering
float miePhase(float cosTheta, float g) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

// [YUS13] https://github.com/GameTechDev/OutdoorLightScattering/blob/master/fx/LightScattering.fx
float rayleighPhaseAngular(float cosTheta) {
    return 1.0f + cosTheta * cosTheta;
}

// [YUS13] https://github.com/GameTechDev/OutdoorLightScattering/blob/master/fx/LightScattering.fx
float miePhaseAngular(float cosTheta, float g) {
    return (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

struct ScatteringResult {
    vec3 transmittance;
    vec3 inScattering;
};

struct RaymarchParameters {
    vec3 rayStart;
    vec3 rayEnd;
    float cosZenith;
    float rayleighPhaseAngular;
    float miePhaseAngular;
    uint steps;
};

vec3 sampleTransmittanceLUT(
AtmosphereParameters atmosphere, float cosLightZenith,
float sampleAltitude, sampler2D transmittanceLUT
) {
    vec2 tLUTUV;
    lutTransmittanceParamsToUv(atmosphere, sampleAltitude, cosLightZenith, tLUTUV);
    uint cond = uint(any(lessThan(tLUTUV, vec2(0.0))));
    cond |= uint(any(greaterThan(tLUTUV, vec2(1.0))));
    if (bool(cond)) {
        return vec3(0.0);
    }
    return texture(transmittanceLUT, tLUTUV).rgb;
}

vec3 computeOpticalDepth(AtmosphereParameters atmosphere, vec3 density) {
    vec3 result = vec3(0.0);
    result += atmosphere.rayleighExtinction * density.x;
    result += atmosphere.mieExtinction * density.y;
    result += atmosphere.ozoneExtinction * density.z;
    return result;
}

void computePointDiffInSctr(
vec3 sampleUnitDensity,
vec3 tSampleToOrigin,
vec3 tSunToSample,
out vec3 rayleighInSctr,
out vec3 mieInSctr
) {
    vec3 totalTransmittance = tSunToSample * tSampleToOrigin;

    rayleighInSctr = sampleUnitDensity.x * totalTransmittance;
    mieInSctr = sampleUnitDensity.y * totalTransmittance;
}

ScatteringResult raymarchSingleScattering(
AtmosphereParameters atmosphere, RaymarchParameters params,
sampler2D transmittanceLUT
) {
    ScatteringResult result = ScatteringResult(vec3(1.0), vec3(0.0));

    float rcpSteps = 1.0 / float(params.steps);
    vec3 stepDelta = (params.rayEnd - params.rayStart) * rcpSteps;
    float stepLength = length(params.rayEnd - params.rayStart) * rcpSteps;

    vec3 prevDensity;
    vec3 prevRayleighInSctr;
    vec3 prevMieInSctr;

    vec3 totalDensity = vec3(0.0);
    vec3 totalRayleighInSctr = vec3(0.0);
    vec3 totalMieInSctr = vec3(0.0);
    {
        vec3 samplePos = params.rayStart;
        float sampleHeight = length(samplePos);
        vec3 sampleDensity = sampleParticleDensity(atmosphere, sampleHeight);

        prevDensity = sampleDensity;

        vec3 tSunToSample = sampleTransmittanceLUT(atmosphere, params.cosZenith, sampleHeight, transmittanceLUT);
        vec3 tSampleToOrigin = vec3(1.0);

        computePointDiffInSctr(sampleDensity, tSampleToOrigin, tSunToSample, prevRayleighInSctr, prevMieInSctr);
    }

    for (uint stepIndex = 1u; stepIndex <= params.steps; stepIndex++) {
        float stepIndexF = float(stepIndex);
        vec3 samplePos = params.rayStart + stepIndexF * stepDelta;
        float sampleHeight = length(samplePos);
        vec3 sampleDensity = sampleParticleDensity(atmosphere, sampleHeight);

        totalDensity += (prevDensity + sampleDensity) * (stepLength * 0.5);
        prevDensity = sampleDensity;

        vec3 tSunToSample = sampleTransmittanceLUT(atmosphere, params.cosZenith, sampleHeight, transmittanceLUT);
        vec3 tSampleToOrigin = exp(-computeOpticalDepth(atmosphere, totalDensity));

        vec3 sampleRayleightInSctr;
        vec3 sampleMieInSctr;
        computePointDiffInSctr(sampleDensity, tSampleToOrigin, tSunToSample, sampleRayleightInSctr, sampleMieInSctr);

        totalRayleighInSctr += (prevRayleighInSctr + sampleRayleightInSctr) * (stepLength * 0.5);
        totalMieInSctr += (prevMieInSctr + sampleMieInSctr) * (stepLength * 0.5);

        prevRayleighInSctr = sampleRayleightInSctr;
        prevMieInSctr = sampleMieInSctr;
    }

    vec3 totalOpticalDepth = computeOpticalDepth(atmosphere, totalDensity);
    result.transmittance = exp(-totalOpticalDepth);

    vec3 totalInSctr = vec3(0.0);
    totalInSctr += params.rayleighPhaseAngular * atmosphere.rayleighSctrCoeffAngular * totalRayleighInSctr;
    totalInSctr += params.miePhaseAngular * atmosphere.mieSctrCoeffAngular * totalMieInSctr;
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

    vec3 prevDensity = vec3(0.0);
    {
        vec3 samplePos = origin;
        float sampleHeight = length(samplePos);

        prevDensity = sampleParticleDensity(atmosphere, sampleHeight);
    }
    for (uint stepIndex = 1u; stepIndex <= steps; stepIndex++) {
        float stepIndexF = float(stepIndex);
        vec3 samplePos = origin + stepIndexF * stepDelta;
        float sampleHeight = length(samplePos);
        vec3 sampleDensity = sampleParticleDensity(atmosphere, sampleHeight);

        totalDensity += (prevDensity + sampleDensity) * (stepLength * 0.5);
        prevDensity = sampleDensity;
    }

    vec3 totalOpticalDepth = computeOpticalDepth(atmosphere, totalDensity);
    result = exp(-totalOpticalDepth);

    return result;
}


#endif