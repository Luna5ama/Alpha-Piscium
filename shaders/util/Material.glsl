#ifndef INCLUDE_util_Material_glsl
#define INCLUDE_util_Material_glsl a
#include "GBufferData.glsl"
#include "Math.glsl"
#include "Colors.glsl"
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

Material material_decode(GBufferData gData) {
    Material material;

    material.albedo = colors_srgbToLinear(gData.albedo);

    material.roughness = 1.0 - gData.pbrSpecular.r;
    material.roughness *= material.roughness;
    material.roughness = max(material.roughness, 0.01);
    material.f0 = gData.pbrSpecular.g;
    material.metallic = float(material.f0 >= (229.5 / 255.0));

    float emissivePBR = pow(gData.pbrSpecular.a, SETTING_EMISSIVE_PBR_VALUE_CURVE);
    vec4 emissiveAlbedoCurve = vec4(vec3(SETTING_EMISSIVE_ALBEDO_COLOR_CURVE), SETTING_EMISSIVE_ALBEDO_LUM_CURVE);
    float albedoLuminanceAlternative = dot(material.albedo, (vec3(0.33) + vec3(0.2126, 0.7152, 0.0722)) * 0.5);
    vec4 emissiveAlbedo = pow(vec4(material.albedo, albedoLuminanceAlternative), emissiveAlbedoCurve);

    float emissiveValue = emissivePBR * 64.0;
    emissiveValue = gData.materialID == 1u ? colors_blackBodyRadiation(SETTING_LAVA_TEMPERATURE, 1.0).a : emissiveValue;
    emissiveValue = gData.materialID == 2u ? colors_blackBodyRadiation(SETTING_FIRE_TEMPERATURE, 1.0).a : emissiveValue;
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