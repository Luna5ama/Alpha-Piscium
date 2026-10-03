#ifndef INCLUDE_techniques_gi_HitDirectLighting_glsl
#define INCLUDE_techniques_gi_HitDirectLighting_glsl a

#include "/techniques/Lighting.glsl"
#include "/techniques/atmospherics/air/lut/API.glsl"

vec3 gi_hitDirectLighting(Material material, vec3 worldPos, vec3 V, vec3 N, vec3 geomN) {
    material.sss = 0.0;
    GBufferData gData = gbufferData_init();
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 scenePos = worldPos - cameraPosition;
    vec3 viewPos = coords_pos_worldToView(scenePos, gbufferModelView);
    vec3 atmPos = atmosphere_viewToAtm(atmosphere, viewPos);
    atmPos.y = max(atmPos.y, atmosphere.bottom + 0.1);
    float viewAltitude = length(atmPos);
    vec3 upVector = atmPos / viewAltitude;
    const vec3 earthCenter = vec3(0.0);

    vec4 shadowPos = global_shadowProj * global_shadowRotationMatrix * global_shadowView * vec4(scenePos + geomN * 0.02, 1.0);
    vec3 shadowTexCoord = shadowPos.xyz / shadowPos.w * 0.5 + 0.5;
    shadowTexCoord.xy = rtwsm_warpTexCoord(shadowTexCoord.xy);
    float shadow = rtwsm_sampleShadowDepth(shadowtex1HW, shadowTexCoord, 0.0);
    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));

    float cosSunZenith = dot(uval_sunDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tSun = atmospherics_air_lut_sampleTransmittance(atmosphere, cosSunZenith, viewAltitude);
    tSun *= float(raySphereIntersectNearest(atmPos, uval_sunDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
    vec3 sunShadow = mix(vec3(1.0), vec3(shadow), shadowIsSun);
    LightingResult sunLighting = directLighting(
        gData,
        material,
        SUN_ILLUMINANCE * tSun,
        vec4(sunShadow, 0.0),
        V,
        uval_sunDirWorld,
        N
    );

    float cosMoonZenith = dot(uval_moonDirWorld, vec3(0.0, 1.0, 0.0));
    vec3 tMoon = atmospherics_air_lut_sampleTransmittance(atmosphere, cosMoonZenith, viewAltitude);
    tMoon *= float(raySphereIntersectNearest(atmPos, uval_moonDirWorld, earthCenter + PLANET_RADIUS_OFFSET * upVector, atmosphere.bottom) < 0.0);
    vec3 moonShadow = mix(vec3(shadow), vec3(1.0), shadowIsSun);
    LightingResult moonLighting = directLighting(
        gData,
        material,
        MOON_ILLUMINANCE * tMoon,
        vec4(moonShadow, 0.0),
        V,
        uval_moonDirWorld,
        N
    );

    LightingResult lighting = lightingResult_add(sunLighting, moonLighting);
    return lighting.diffuse + lighting.specular;
}

#endif
