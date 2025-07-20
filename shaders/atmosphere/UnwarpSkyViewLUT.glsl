#include "lut/Common.glsl"
#include "Common.glsl"

void uvToSkyViewLutParams(
    AtmosphereParameters Atmosphere,
    out float viewZenithCosAngle,
    out float lightViewCosAngle,
    in float viewHeight,
    in vec2 uv
) {
    uv = vec2(fromSubUvsToUnit(uv.x, 256.0), fromSubUvsToUnit(uv.y, 256.0));

    float Vhorizon = sqrt(viewHeight * viewHeight - Atmosphere.bottom * Atmosphere.bottom);
    float CosBeta = Vhorizon / viewHeight;  // GroundToHorizonCos
    float Beta = acos(CosBeta);
    float ZenithHorizonAngle = PI - Beta;

    if (uv.y < 0.5) {
        float coord = 2.0 * uv.y;
        coord = 1.0 - coord;
        coord *= coord; // Non linear sky view LUT
        coord = 1.0 - coord;
        viewZenithCosAngle = cos(ZenithHorizonAngle * coord);
    } else {
        float coord = uv.y * 2.0 - 1.0;
        coord *= coord; // Non linear sky view LUT
        viewZenithCosAngle = cos(ZenithHorizonAngle + Beta * coord);
    }

    float coord = uv.x;
    coord *= coord;
    lightViewCosAngle = -(coord * 2.0 - 1.0);
}

void skyViewLutParamsToUv(
    in AtmosphereParameters Atmosphere,
    in bool IntersectGround,
    in float viewZenithCosAngle,
    in float lightViewCosAngle,
    in float viewHeight,
    out vec2 uv
) {
    float Vhorizon = sqrt(viewHeight * viewHeight - Atmosphere.bottom * Atmosphere.bottom);
    float CosBeta = Vhorizon / viewHeight;  // GroundToHorizonCos
    float Beta = acos(CosBeta);
    float ZenithHorizonAngle = PI - Beta;

    if (!IntersectGround) {
        float coord = acos(viewZenithCosAngle) / ZenithHorizonAngle;
        coord = 1.0 - coord;
        coord = sqrt(coord);    // Non linear sky view LUT
        coord = 1.0 - coord;
        uv.y = coord * 0.5;
    } else {
        float coord = (acos(viewZenithCosAngle) - ZenithHorizonAngle) / Beta;
        coord = sqrt(coord);    // Non linear sky view LUT
        uv.y = coord * 0.5 + 0.5;
    }

    {
        float coord = -lightViewCosAngle * 0.5 + 0.5;
        coord = sqrt(coord);
        uv.x = coord;
    }

    uv = vec2(fromUnitToSubUvs(uv.x, 256.0), fromUnitToSubUvs(uv.y, 256.0));
}

ScatteringResult sampleSkyViewLUT(
    vec2 screenPos, vec3 viewPos, float viewZ
) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    ScatteringResult result = scatteringResult_init();

    // TODO: Implement sky view LUT sampling

    return result;
}