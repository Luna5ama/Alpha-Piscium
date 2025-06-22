#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"
#include "/util/Material.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Fresnel.glsl"
#include "/util/BSDF.glsl"
#include "/rtwsm/RTWSM.glsl"

uniform sampler2D usam_skyLUT;

GBufferData lighting_gData = gbufferData_init();
vec3 lighting_viewCoord;
vec3 lighting_viewDir;
ivec2 lighting_texelPos;

void lighting_init(vec3 viewCoord, ivec2 texelPos) {
    lighting_viewCoord = viewCoord;
    lighting_viewDir = normalize(-viewCoord);
    lighting_texelPos = texelPos;
}

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

vec3 skyReflection(Material material, float lmCoordSky, vec3 N) {
    vec3 V = lighting_viewDir;
    float NDotV = dot(N, V);
    vec3 fresnelReflection = fresnel_evalMaterial(material, saturate(NDotV));

    vec3 reflectDirView = reflect(-V, N);
    vec3 reflectDir = normalize(mat3(gbufferModelViewInverse) * reflectDirView);
    vec2 reflectLUTUV = coords_octEncode01(reflectDir);
    vec3 reflectRadiance = texture(usam_skyLUT, reflectLUTUV).rgb;

    vec3 result = fresnelReflection;
    result *= material.albedo;
    result *= linearStep(1.5 / 16.0, 15.5 / 16.0, lmCoordSky);
    result *= texture(usam_skyLUT, reflectLUTUV).rgb;

    return result;
}

LightingResult directLighting(Material material, vec4 irradiance, vec3 L, vec3 N) {
    vec3 V = lighting_viewDir;
    vec3 H = normalize(L + V);
    float LDotV = dot(L, V);
    float LDotH = dot(L, H);
    float NDotL = dot(N, L);
    float NDotV = dot(N, V);
    float NDotH = dot(N, H);

    vec3 fresnel = fresnel_evalMaterial(material, saturate(LDotH));

    LightingResult result;

    float diffuseBaseF = 1.0 - material.metallic;
    vec3 diffuseBaseVec3 = diffuseBaseF * (irradiance.rgb * (vec3(1.0) - fresnel) * material.albedo);

    result.diffuse = diffuseBaseVec3 * bsdf_diffuseHammon(material, NDotL, NDotV, LDotH, LDotV);
    result.diffuseLambertian = diffuseBaseVec3 * (RCP_PI * saturate(NDotL));

    float shadowPow = saturate(1.0 - irradiance.a);
    shadowPow = (1.0 - SETTING_SSS_HIGHLIGHT * 0.5) + pow4(shadowPow) * SETTING_SSS_SCTR_FACTOR;

    float phase = biLambertianPlatePhase(LDotV, 0.3);
    float sssV = material.sss * phase * SETTING_SSS_STRENGTH;
    result.sss = sssV * pow(material.albedo, vec3(shadowPow)) * irradiance.rgb;

    result.specular = irradiance.rgb * fresnel * bsdf_ggx(material, NDotL, NDotV, NDotH);
    result.specular = min(result.specular, SETTING_MAXIMUM_SPECULAR_LUMINANCE);

    return result;
}