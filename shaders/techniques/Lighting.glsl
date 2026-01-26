#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"

GBufferData lighting_gData = gbufferData_init();

struct LightingResult {
    vec3 diffuse;
    vec3 diffuseLambertian;
    vec3 specular;
    vec3 sss;
};

LightingResult lightingResult_add(LightingResult a, LightingResult b) {
    LightingResult result;
    result.diffuse = a.diffuse + b.diffuse;
    result.diffuseLambertian = a.diffuseLambertian + b.diffuseLambertian;
    result.specular = a.specular + b.specular;
    result.sss = a.sss + b.sss;
    return result;
}

LightingResult directLighting(Material material, vec3 irradiance, float surfaceDepth, vec3 V, vec3 L, vec3 N) {
    vec3 H = normalize(L + V);
    float LDotV = clamp(dot(L, V), -1.0, 1.0);
    float LDotH = clamp(dot(L, H), -1.0, 1.0);
    float NDotL = clamp(dot(N, L), -1.0, 1.0);
    float NDotV = clamp(dot(N, V), -1.0, 1.0);
    float NDotH = clamp(dot(N, H), -1.0, 1.0);

    vec3 fresnel = fresnel_evalMaterial(material, saturate(LDotH));

    LightingResult result;

    float diffuseBaseF = 1.0 - material.metallic;
    vec3 diffuseBaseVec3 = diffuseBaseF * (irradiance * (1.0 - fresnel) * material.albedo);

    result.diffuse = diffuseBaseVec3 * bsdf_diffuseHammon(material, NDotL, NDotV, LDotH, LDotV);
    result.diffuseLambertian = diffuseBaseVec3 * (RCP_PI * saturate(NDotL));

    result.sss = vec3(0.0);

    if (material.sss > 0.0) {
        const float ABSORPTION_MULTIPLIER = 1.0;
        const float SCATTERING_MULTIPLIER = 1.5;
        const float DENSITY_MULTIPLIER = 16.0;
        const float ABSO_POW = 1.2;
        const float SCTR_POW = 0.6;

        vec3 albedoSRGB = colors2_colorspaces_convert(COLORS2_WORKING_COLORSPACE, COLORS2_COLORSPACES_SRGB, material.albedo);

        vec3 tCoeff = pow(albedoSRGB, vec3(ABSO_POW));
        tCoeff = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_WORKING_COLORSPACE, tCoeff);
        vec3 aCoeff = -log(tCoeff);

        vec3 sCoeff = pow(albedoSRGB, vec3(SCTR_POW));
        sCoeff = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_WORKING_COLORSPACE, sCoeff);

        float sss2 = pow2(material.sss);
        aCoeff = max(aCoeff, 0.0) * ABSORPTION_MULTIPLIER / sss2;
        sCoeff *= SCATTERING_MULTIPLIER * sss2;

        float density = DENSITY_MULTIPLIER;
        vec3 sampleScattering = sCoeff * density;
        vec3 sampleExtinction = (aCoeff + sCoeff) * density;
        vec3 sampleOpticalDepth = sampleExtinction * surfaceDepth;
        vec3 sampleTransmittance = exp(-sampleOpticalDepth);

        vec3 sampleIrradiance = irradiance;

        float msRadius = sqrt(surfaceDepth) + 1.0;
        vec3 fMS = (sampleScattering / sampleExtinction) * (1.0 - exp(-msRadius * sampleExtinction));
        vec3 sampleMSIrradiance = irradiance;
        sampleMSIrradiance *= UNIFORM_PHASE;
        sampleMSIrradiance *= fMS / (1.0 - fMS);
        sampleIrradiance += sampleMSIrradiance;

        vec3 sampleInSctr = sampleIrradiance * sampleScattering;
        vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;

        // This fakes lights from sun disk + mie haze
        float sunMiePhase = phasefunc_KleinNishinaE(-LDotV, 1e5);
        vec3 sheenTransmittance = max(exp(-sampleOpticalDepth), exp(-sampleOpticalDepth * 0.25) * 0.7);

        float sssFresnel = frenel_schlick(1.0 - abs(LDotV), 0.04);
        float phase = phasefunc_BiLambertianPlate(-LDotV, 0.3);
        result.sss = phase * sampleInSctrInt;
        result.sss += sunMiePhase * sheenTransmittance * irradiance * (1.0 - sssFresnel);
    }

    result.specular = irradiance * fresnel * bsdf_ggx(material, NDotL, NDotV, NDotH);
    result.specular = min(result.specular, SETTING_MAXIMUM_SPECULAR_LUMINANCE);

    return result;
}

// TODO: Cleanup
LightingResult directLighting2(Material material, vec4 irradiance, vec3 V, vec3 L, vec3 N, float ior) {
    vec3 H = normalize(L + V);
    float LDotV = clamp(dot(L, V), -1.0, 1.0);
    float LDotH = clamp(dot(L, H), -1.0, 1.0);
    float NDotL = clamp(dot(N, L), -1.0, 1.0);
    float NDotV = clamp(dot(N, V), -1.0, 1.0);
    float NDotH = clamp(dot(N, H), -1.0, 1.0);

    float fresnel = fresnel_dielectricDielectric_reflection(saturate(LDotH), AIR_IOR, ior);

    LightingResult result;

    float diffuseBaseF = 1.0 - material.metallic;
    vec3 diffuseBaseVec3 = diffuseBaseF * (irradiance.rgb * (vec3(1.0) - fresnel) * material.albedo);

    result.diffuse = diffuseBaseVec3 * bsdf_diffuseHammon(material, NDotL, NDotV, LDotH, LDotV);
    result.diffuseLambertian = diffuseBaseVec3 * (RCP_PI * saturate(NDotL));

    float shadowPow = saturate(1.0 - irradiance.a);
    shadowPow = (1.0 - SETTING_SSS_HIGHLIGHT * 0.5) + pow4(shadowPow) * SETTING_SSS_SCTR_FACTOR;

    float phase = phasefunc_BiLambertianPlate(-LDotV, 0.3);
    float sssV = material.sss * phase * SETTING_SSS_STRENGTH;
    result.sss = sssV * pow(material.albedo, vec3(shadowPow)) * irradiance.rgb;

    result.specular = irradiance.rgb * fresnel * bsdf_ggx(material, NDotL, NDotV, NDotH);
    result.specular = min(result.specular, SETTING_MAXIMUM_SPECULAR_LUMINANCE);

    return result;
}