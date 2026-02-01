#ifndef INCLUDE_util_Translucent_glsl
#define INCLUDE_util_Translucent_glsl a

#include "/util/Colors2.glsl"

vec4 translucent_albedoToTransmittance(vec3 materialColor, float alpha, uint materialID) {
    if (materialID == 3u) {
        return vec4(1.0);
    }
    vec3 t = materialColor;
    float lumaT = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, t);
    t *= saturate(0.3 / lumaT);

    t = pow(t, vec3(1.0 / 2.2));
    lumaT = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, t);
    float sat = SETTING_TRANSLUCENT_ABSORPTION_SATURATION;
    t = lumaT + sat * (t - lumaT);
    t = pow(t, vec3(2.2));

    float absorptionMultiplier = alpha * sqrt(alpha);
    vec3 absorption = -log(t) * absorptionMultiplier;
    absorption = max(absorption, 0.0);

    vec3 transmittance = exp(-absorption);

    return vec4(transmittance, 0.0);
}

#endif