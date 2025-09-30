#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"

layout(rgba16f) uniform writeonly image2D uimg_temp3;

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

LightingResult directLighting(Material material, vec4 irradiance, vec3 V, vec3 L, vec3 N) {
    imageStore(uimg_temp3, ivec2(gl_GlobalInvocationID.xy), vec4(V, 1.0));

    vec3 H = normalize(L + V);
    float LDotV = clamp(dot(L, V), -1.0, 1.0);
    float LDotH = clamp(dot(L, H), -1.0, 1.0);
    float NDotL = clamp(dot(N, L), -1.0, 1.0);
    float NDotV = clamp(dot(N, V), -1.0, 1.0);
    float NDotH = clamp(dot(N, H), -1.0, 1.0);

    vec3 fresnel = fresnel_evalMaterial(material, saturate(LDotH));

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