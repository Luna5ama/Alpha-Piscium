#include "Common.glsl"

float circle(vec3 rayDir, vec3 objDir, float objAngularRadius, float bloomPower) {
    float objCosTheta = cos(objAngularRadius);
    float cosTheta = saturate(dot(rayDir, objDir));

    float x = saturate(objCosTheta - cosTheta);
    float invBloom = (bloomPower / (bloomPower + x)) * 1.0 * cosTheta * cosTheta;
    return invBloom;
}

vec3 renderSunMoon(ivec2 texelPos) {
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;
    vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;
    vec3 viewCoord = coords_toViewCoord(screenPos, -far, gbufferProjectionInverse);

    vec3 viewDir = normalize(viewCoord);
    vec3 viewDirWorld = mat3(gbufferModelViewInverse) * viewDir;

    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 origin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    origin.y = max(origin.y, atmosphere.bottom + 0.5);
    vec3 earthCenter = vec3(0.0);
    float earthIntersect = raySphereIntersectNearest(origin, viewDirWorld, earthCenter, atmosphere.bottom);

    const float moonAngularRadius = 0.528611 * PI / 180.0;

    float sunV = circle(viewDir, uval_sunDirView, uval_sunAngularRadius, 0.000005);
    float moonV = circle(viewDir, uval_moonDirView, moonAngularRadius, 0.000001);

    vec3 result = vec3(0.0);
    result += sunV * sunRadiance * 16.0;
    result += moonV * sunRadiance * MOON_RADIANCE_MUL * 16.0;
    result *= step(earthIntersect, 0.0);

    return result;
}