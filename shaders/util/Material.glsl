#ifndef INCLUDE_util_Material_glsl
#define INCLUDE_util_Material_glsl a
#include "BlackBody.glsl"
#include "Colors.glsl"
#include "GBufferData.glsl"
#include "Math.glsl"
#include "Rand.glsl"

struct Material {
    vec3 albedo;
    float roughness;
    float f0;
    float metallic;
    vec3 emissive;
    float porosity;
    float sss;
};

const float _MATERIAL_F0_EPSILON = exp2(-SETTING_SPECULAR_MAPPING_MINIMUM_F0_FACTOR);
const float _MATERIAL_MINIMUM_ROUGHNESS = exp2(-SETTING_SPECULAR_MAPPING_MINIMUM_ROUGHNESS_FACTOR);
const float _MATERIAL_MAXIMUM_ROUGHNESS = 1.0 - exp2(-SETTING_SPECULAR_MAPPING_MAXIMUM_ROUGHNESS_FACTOR);
const float _MATERIAL_LAVA_LUMINANCE = colors_sRGB_luma(blackBody_evalRadiance(SETTING_LAVA_TEMPERATURE));
const float _MATERIAL_FIRE_LUMINANCE = colors_sRGB_luma(blackBody_evalRadiance(SETTING_FIRE_TEMPERATURE));

Material material_decode(GBufferData gData) {
    Material material;

    material.albedo = colors_srgbToLinear(gData.albedo);

    material.roughness = 1.0 - gData.pbrSpecular.r;
    material.roughness *= material.roughness;
    material.roughness = clamp(material.roughness, _MATERIAL_MINIMUM_ROUGHNESS, _MATERIAL_MAXIMUM_ROUGHNESS);
    material.f0 = gData.pbrSpecular.g;
    #if SETTING_SPECULAR_MAPPING_MINIMUM_F0_FACTOR > 0
    material.f0 = max(material.f0, _MATERIAL_F0_EPSILON);
    #endif
    material.metallic = float(material.f0 >= (229.5 / 255.0));

    float emissivePBR = pow(gData.pbrSpecular.a, SETTING_EMISSIVE_PBR_VALUE_CURVE);
    vec4 emissiveAlbedoCurve = vec4(vec3(SETTING_EMISSIVE_ALBEDO_COLOR_CURVE), SETTING_EMISSIVE_ALBEDO_LUM_CURVE);
    float albedoLuminanceAlternative = dot(material.albedo, (vec3(0.33) + vec3(0.2126, 0.7152, 0.0722)) * 0.5);
    vec4 emissiveAlbedo = pow(vec4(material.albedo, albedoLuminanceAlternative), emissiveAlbedoCurve);

    float emissiveValue = emissivePBR * 0.2;
    emissiveValue = gData.materialID == 1u ? _MATERIAL_LAVA_LUMINANCE : emissiveValue;
    emissiveValue = gData.materialID == 2u ? _MATERIAL_FIRE_LUMINANCE : emissiveValue;
    emissiveValue *= SETTING_EMISSIVE_STRENGTH;

    material.emissive = emissiveValue * emissiveAlbedo.a * emissiveAlbedo.rgb;

    const float _64o255 = 64.0 / 255.0;
    const float _65o255 = 65.0 / 255.0;
    float step64 = step(_65o255, gData.pbrSpecular.b);
    material.porosity = linearStep(0.0, _64o255, gData.pbrSpecular.b);
    material.porosity *= 1.0 - step64;

    material.sss = linearStep(_65o255, 1.0, gData.pbrSpecular.b);
    material.sss *= step64;
    material.sss = sqrt(material.sss);

    return material;
}

#endif