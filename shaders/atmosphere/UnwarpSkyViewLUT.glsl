
#include "Common.glsl"

ScatteringResult unwarpSkyView(vec2 screenPos, vec3 viewPos, float viewZ) {
    ScatteringResult result = scatteringResult_init();

    AtmosphereParameters atmosphere = getAtmosphereParameters();

    vec3 feetPlayer = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec4 worldPos = vec4(feetPlayer + cameraPosition, 1.0);
    vec3 rayDir = normalize(worldPos.xyz - cameraPosition);

    float viewHeight = length(cameraPosition);

    vec3 zenithDir = normalize(cameraPosition);

    float viewZenithCosAngle = dot(zenithDir, rayDir);

    vec3 sunDir = normalize(sunPosition - cameraPosition);
    float lightViewCosAngle = dot(sunDir, rayDir);

    float Vhorizon = sqrt(max(0.0, viewHeight * viewHeight - atmosphere.bottom * atmosphere.bottom));
    float CosBeta = Vhorizon / viewHeight;
    float Beta = acos(CosBeta);
    float ZenithHorizonAngle = PI - Beta;

    float viewZenithAngle = acos(viewZenithCosAngle);
    bool IntersectGround = (viewZenithAngle > ZenithHorizonAngle);

    vec2 uv;
    if (!IntersectGround) {
        float coord = viewZenithAngle / ZenithHorizonAngle;
        coord = 1.0 - coord;
        coord = sqrt(coord); // Non linear mapping
        coord = 1.0 - coord;
        uv.y = coord * 0.5f;
    } else {
        float coord = (viewZenithAngle - ZenithHorizonAngle) / Beta;
        coord = sqrt(coord); // Non linear mapping
        uv.y = coord * 0.5f + 0.5f;
    }

    float coord = -lightViewCosAngle * 0.5 + 0.5;
    coord = sqrt(coord);
    uv.x = coord;

    const vec2 skyViewLutSize = vec2(192.0, 108.0);
    uv.x = (uv.x * skyViewLutSize.x + 0.5) / skyViewLutSize.x;
    uv.y = (uv.y * skyViewLutSize.y + 0.5) / skyViewLutSize.y;

    vec3 scattering = texture(usam_skyViewLUT_scattering, uv).rgb;
    float transmittance = texture(usam_skyViewLUT_transmittance, uv).r;

    result.inScattering = scattering;
    result.transmittance = vec3(transmittance);
    return result;
}
