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

#include "../Constants.glsl"
#include "/util/Math.glsl"

// [HIL20] https://github.com/sebh/UnrealEngineSkyAtmosphere/blob/master/Resources/RenderSkyCommon.hlsl
// Transmittance LUT function parameterisation from Bruneton 2017 https://github.com/ebruneton/precomputed_atmospheric_scattering
// uv in [0,1]
// viewZenithCosAngle in [-1,1]
// viewAltitude in [bottomRAdius, topRadius]
float fromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float fromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

vec2 fromUnitToSubUvs(vec2 uv, vec2 resolution) { return (uv + 0.5 / resolution) * (resolution / (resolution + 1.0)); }
vec2 fromSubUvsToUnit(vec2 uv, vec2 resolution) { return (uv - 0.5 / resolution) * (resolution / (resolution - 1.0)); }

uniform sampler2D usam_transmittanceLUT;
uniform sampler2D usam_multiSctrLUT;


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

#endif