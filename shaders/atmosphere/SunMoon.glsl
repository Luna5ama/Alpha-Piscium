#include "Common.glsl"
#include "/util/CelestialObjects.glsl"
#include "/util/Coords.glsl"

float circle(vec3 rayDir, vec3 objDir, float objAngularRadius) {
    float objCosTheta = cos(objAngularRadius);
    float cosTheta = saturate(dot(rayDir, objDir));
    return float(cosTheta >= objCosTheta);
}

vec4 renderSunMoon(ivec2 texelPos) {
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;
    vec3 viewCoord = coords_toViewCoord(screenPos, -far, gbufferProjectionInverse);

    vec3 viewDir = normalize(viewCoord);
    vec3 viewDirWorld = mat3(gbufferModelViewInverse) * viewDir;

    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 origin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    origin.y = max(origin.y, atmosphere.bottom + 0.5);
    vec3 earthCenter = vec3(0.0);
    float earthIntersect = raySphereIntersectNearest(origin, viewDirWorld, earthCenter, atmosphere.bottom);

    float sunV = circle(viewDir, uval_sunDirView, SUN_ANGULAR_RADIUS * 2.0);
    float moonV = circle(viewDir, uval_moonDirView, MOON_ANGULAR_RADIUS * 2.0);

    vec4 result = vec4(0.0);
    result += sunV * vec4(SUN_LUMINANCE, 2.0);
    result += moonV * vec4(MOON_LUMINANCE, 2.0);
    result *= step(earthIntersect, 0.0);

    return result;
}