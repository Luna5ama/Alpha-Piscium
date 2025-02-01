#ifndef INCLUDE_util_Material.glsl
#define INCLUDE_util_Material.glsl

#include "GBuffers.glsl"
#include "Math.glsl"
#include "Colors.glsl"
#include "Rand.glsl"

struct Material {
    vec3 albedo;
    float materialAO;
    float roughness;
    float f0;
    vec3 emissive;
    float porosity;
    float sss;
};

Material material_decode(GBufferData gData) {
    Material material;

    material.albedo = colors_srgbToLinear(gData.albedo);

    material.materialAO = gData.materialAO;
    material.roughness = 1.0 - gData.pbrSpecular.r;
    material.f0 = gData.pbrSpecular.g;

    const float _1o255 = 1.0 / 255.0;
    float emissiveS = gData.pbrSpecular.a;
    material.emissive = mix(vec3(0.0), emissiveS * 64.0 * material.albedo, saturate(float(gData.materialID == 65535u) + float(gData.materialID == 65534u)));
    material.emissive = mix(material.emissive, colors_blackBodyRadiation(SETTING_LAVA_TEMPERATURE, 1.0).a * material.albedo, float(gData.materialID == 1u));
    material.emissive = mix(material.emissive, colors_blackBodyRadiation(SETTING_FIRE_TEMPERATURE, 1.0).a * material.albedo, float(gData.materialID == 2u));

    const float _64o255 = 64.0 / 255.0;
    const float _65o255 = 65.0 / 255.0;
    float step64 = step(_65o255, gData.pbrSpecular.b);
    material.porosity = linearStep(0.0, _64o255, gData.pbrSpecular.b);
    material.porosity *= 1.0 - step64;

    material.sss = linearStep(_65o255, 1.0, gData.pbrSpecular.b);
    material.sss *= step64;

    return material;
}

#endif