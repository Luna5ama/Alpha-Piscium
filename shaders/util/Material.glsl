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

    const float _1o255 = 1.0 / 255.0;
    float emissiveS = pow(gData.pbrSpecular.a, SETTING_EMISSIVE_PBR_CURVE);
    material.emissive = emissiveS * 64.0 * pow(material.albedo, vec3(SETTING_EMISSIVE_COLOR_CURVE));
    material.emissive = mix(material.emissive, colors_blackBodyRadiation(SETTING_LAVA_TEMPERATURE, 1.0).a * material.albedo, float(gData.materialID == 1u));
    material.emissive = mix(material.emissive, colors_blackBodyRadiation(SETTING_FIRE_TEMPERATURE, 1.0).a * material.albedo, float(gData.materialID == 2u));
    material.emissive *= SETTING_EMISSIVE_STRENGTH;

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