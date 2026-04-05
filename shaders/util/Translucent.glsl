#ifndef INCLUDE_util_Translucent_glsl
#define INCLUDE_util_Translucent_glsl a

#include "/util/Colors2.glsl"
#include "/util/MaterialIDConst.glsl"

vec4 translucent_albedoToTransmittance(vec4 inputAlbedo, uint materialID) {
    if (materialID == MATERIAL_ID_WATER) {
        return vec4(1.0);
    }
    vec3 t = inputAlbedo.rgb;
    float lumaT = colors2_colorspaces_luma(COLORS2_MATERIAL_COLORSPACE, t);
    t *= saturate(0.6 / lumaT); // Fix for white glasses

    lumaT = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, t);
    float sat = SETTING_TRANSLUCENT_ABSORPTION_SATURATION;
    t = lumaT + sat * (t - lumaT);
    t = colors2_material_toWorkSpace(t);
    t = pow(t, vec3(SETTING_TRANSLUCENT_ABSORPTION_GAMMA));

    float absorptionMultiplier = pow(inputAlbedo.a, SETTING_TRANSLUCENT_ABSORPTION_ALPHA_CURVE);
    vec3 absorption = log(t) * absorptionMultiplier * SETTING_TRANSLUCENT_ABSORPTION_MULTIPLIER;

    vec3 transmittance = saturate(exp(absorption));
    return vec4(transmittance, 0.0);
}

#endif