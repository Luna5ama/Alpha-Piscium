#ifndef INCLUDE_util_Lighting_glsl
#define INCLUDE_util_Lighting_glsl a
/*
    References:
        [BEL25] Belmu. Noble.
            GPL v3.0 License. Copyright (c) 2025 Belmu
            https://github.com/BelmuTM/Noble

        You can find full license texts in /licenses

    Other Credits:
        Jessie - Skylight falloff function (https://github.com/Jessie-LC)
*/

float lighting_skyLightFalloff(float lmCoordSky) {
    return lmCoordSky * exp2(8.0 * (lmCoordSky - 1.0));
}

#endif