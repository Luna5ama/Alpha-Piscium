/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere
        [HIL20] Hillaire, Sébastien. "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
            EGSR 2020. 2020.
            https://sebh.github.io/publications/egsr2020.pdf

        You can find full license texts in /licenses
*/
#ifndef INCLUDE_atmosphere_lut_Common_glsl
#define INCLUDE_atmosphere_lut_Common_glsl a

#include "../Common.glsl"
#include "/util/Colors.glsl"
#include "/util/Math.glsl"

#define SKYVIEW_LUT_WIDTH (SETTING_SKYVIEW_RES / 2)
#define SKYVIEW_LUT_HEIGHT SETTING_SKYVIEW_RES
#define SKYVIEW_LUT_DEPTH 3
#define SKYVIEW_LUT_SIZE ivec2(SKYVIEW_LUT_WIDTH, SKYVIEW_LUT_HEIGHT)
#define SKYVIEW_LUT_SIZE_F vec2(SKYVIEW_LUT_WIDTH, SKYVIEW_LUT_HEIGHT)

// [HIL20] https://github.com/sebh/UnrealEngineSkyAtmosphere/blob/master/Resources/RenderSkyCommon.hlsl
// Transmittance LUT function parameterisation from Bruneton 2017 https://github.com/ebruneton/precomputed_atmospheric_scattering
// uv in [0,1]
// viewZenithCosAngle in [-1,1]
// viewAltitude in [bottomRAdius, topRadius]
float fromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float fromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

vec2 fromUnitToSubUvs(vec2 uv, vec2 resolution) { return (uv + 0.5 / resolution) * (resolution / (resolution + 1.0)); }
vec2 fromSubUvsToUnit(vec2 uv, vec2 resolution) { return (uv - 0.5 / resolution) * (resolution / (resolution - 1.0)); }



void lutTransmittanceParamsToUv(AtmosphereParameters atmosphere, float height, float cosZenith, out vec2 uv) {
    height = clamp(height, atmosphere.bottom + 0.0001, atmosphere.top - 0.0001);
    cosZenith = clamp(cosZenith, -1.0, 1.0);
    float H = sqrt(max(0.0, pow2(atmosphere.top) - pow2(atmosphere.bottom)));
    float rho = sqrt(max(0.0, pow2(height) - pow2(atmosphere.bottom)));

    float discriminant = pow2(height) * (cosZenith * cosZenith - 1.0) + pow2(atmosphere.top);
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
    // Hacky twilight multiple scattering fix
    return texture(usam_multiSctrLUT, uv).rgb * pow6(linearStep(-0.2, 0.0, cosLightZenith));
}

void uvToSkyViewLutParams(
    AtmosphereParameters atmosphere,
    out float viewZenithCosAngle,
    out float lightViewCosAngle,
    in float viewHeight,
    in vec2 uv
) {
    uv.y = 1.0 - uv.y;
    uv = vec2(fromSubUvsToUnit(uv.x, SKYVIEW_LUT_WIDTH), fromSubUvsToUnit(uv.y, SKYVIEW_LUT_HEIGHT));

    float vHorizon = sqrt(pow2(viewHeight) - pow2(atmosphere.bottom));
    float cosBeta = vHorizon / viewHeight;  // GroundToHorizonCos
    float beta = acos(cosBeta);
    float zenithHorizonAngle = PI - beta;

    if (uv.y < 0.5) {
        float coord = 2.0 * uv.y;
        coord = 1.0 - coord;
        coord *= coord; // Non linear sky view LUT
        coord = 1.0 - coord;
        viewZenithCosAngle = cos(zenithHorizonAngle * coord);
    } else {
        float coord = uv.y * 2.0 - 1.0;
        coord *= coord; // Non linear sky view LUT
        viewZenithCosAngle = cos(zenithHorizonAngle + beta * coord);
    }

    float coord = uv.x;
    coord *= coord;
    lightViewCosAngle = -(coord * 2.0 - 1.0);
}

void skyViewLutParamsToUv(
    in AtmosphereParameters atmosphere,
    in bool intersectGround,
    in float viewZenithCosAngle,
    in float lightViewCosAngle,
    in float viewHeight,
    out vec2 uv
) {
    float Vhorizon = sqrt(viewHeight * viewHeight - atmosphere.bottom * atmosphere.bottom);
    float CosBeta = Vhorizon / viewHeight;  // GroundToHorizonCos
    float Beta = acos(CosBeta);
    float ZenithHorizonAngle = PI - Beta;

    if (!intersectGround) {
        float coord = acos(viewZenithCosAngle) / ZenithHorizonAngle;
        coord = 1.0 - coord;
        coord = sqrt(saturate(coord));    // Non linear sky view LUT
        coord = 1.0 - coord;
        uv.y = coord * 0.5;
    } else {
        float coord = (acos(viewZenithCosAngle) - ZenithHorizonAngle) / Beta;
        coord = sqrt(saturate(coord));    // Non linear sky view LUT
        uv.y = coord * 0.5 + 0.5;
    }

    {
        float coord = -lightViewCosAngle * 0.5 + 0.5;
        coord = sqrt(coord);
        uv.x = coord;
    }

    uv = vec2(fromUnitToSubUvs(uv.x, SKYVIEW_LUT_WIDTH), fromUnitToSubUvs(uv.y, SKYVIEW_LUT_HEIGHT));
    uv.y = 1.0 - uv.y;
}

vec3 sampleSkyViewLUTSlice(vec2 sliceUV, float sliceIndex) {
    vec3 sampleUV = vec3(sliceUV, (sliceIndex + 0.5) / SKYVIEW_LUT_DEPTH);
    return colors_LogLuv32ToSRGB(texture(usam_skyViewLUT, sampleUV));
}

ScatteringResult sampleSkyViewLUT(
    AtmosphereParameters atmosphere,
    bool intersectGround,
    float viewZenithCosAngle,
    float sunViewCosAngle,
    float MoonViewCosAngle,
    float viewHeight,
    float layerIndex
) {
    vec2 sunSliceUV;
    skyViewLutParamsToUv(
        atmosphere,
        intersectGround,
        viewZenithCosAngle,
        sunViewCosAngle,
        viewHeight,
        sunSliceUV
    );
    vec2 moonSliceUV;
    skyViewLutParamsToUv(
        atmosphere,
        intersectGround,
        viewZenithCosAngle,
        MoonViewCosAngle,
        viewHeight,
        moonSliceUV
    );
    float sunSlice = layerIndex * 3;
    float moonSlice = sunSlice + 1;
    float tSlice = sunSlice + 2;

    ScatteringResult result = scatteringResult_init();
    result.inScattering = sampleSkyViewLUTSlice(sunSliceUV, sunSlice);
    result.inScattering += sampleSkyViewLUTSlice(moonSliceUV, moonSlice);
    result.transmittance = sampleSkyViewLUTSlice(sunSliceUV, tSlice);
    return result;
}

#endif