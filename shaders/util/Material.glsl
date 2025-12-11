#ifndef INCLUDE_util_Material_glsl
#define INCLUDE_util_Material_glsl a
#include "BlackBody.glsl"
#include "Colors.glsl"
#include "Colors2.glsl"
#include "GBufferData.glsl"
#include "Math.glsl"
#include "Rand.glsl"

struct Material {
    vec3 albedo; // Working space
    float roughness;
    float f0;
    float metallic;
    vec3 emissive;
    float porosity;
    float sss;
    float hardCodedIOR;
    mat3 tbn;
    mat3 tbnInv;
};

#ifdef MATERIAL_TRANSLUCENT
const float _MATERIAL_F0_EPSILON = exp2(-SETTING_MINIMUM_F0);
const float _MATERIAL_ROUGHNESS_MULTIPLIER = exp2(-SETTING_TRANSLUCENT_ROUGHNESS_REDUCTION);
const float _MATERIAL_MINIMUM_ROUGHNESS = exp2(-SETTING_TRANSLUCENT_MINIMUM_ROUGHNESS);
const float _MATERIAL_MAXIMUM_ROUGHNESS = exp2(-SETTING_TRANSLUCENT_MAXIMUM_ROUGHNESS);
const float _MATERIAL_WATER_ROUGHNESS = exp2(-SETTING_WATER_ROUGHNESS);
#else
const float _MATERIAL_F0_EPSILON = exp2(-SETTING_MINIMUM_F0);
const float _MATERIAL_ROUGHNESS_MULTIPLIER = 1.0;
const float _MATERIAL_MINIMUM_ROUGHNESS = exp2(-SETTING_SOLID_MINIMUM_ROUGHNESS);
const float _MATERIAL_MAXIMUM_ROUGHNESS = 1.0 - exp2(-SETTING_SOLID_MAXIMUM_ROUGHNESS);
#endif

Material material_decode(GBufferData gData) {
    Material material;

    material.albedo = colors2_material_toWorkSpace(gData.albedo);

    material.roughness = 1.0 - gData.pbrSpecular.r;
    material.roughness *= material.roughness;
    material.roughness *= _MATERIAL_ROUGHNESS_MULTIPLIER;
    material.roughness = mix(_MATERIAL_MINIMUM_ROUGHNESS, _MATERIAL_MAXIMUM_ROUGHNESS, smoothstep(_MATERIAL_MINIMUM_ROUGHNESS, _MATERIAL_MAXIMUM_ROUGHNESS, material.roughness));
    #ifdef MATERIAL_TRANSLUCENT
    material.roughness = gData.materialID == 3u ? _MATERIAL_WATER_ROUGHNESS : material.roughness;
    #endif
    material.f0 = gData.pbrSpecular.g;

    #if SETTING_MINIMUM_F0_FACTOR > 0
    material.f0 = max(material.f0, _MATERIAL_F0_EPSILON);
    #endif
    material.metallic = float(material.f0 >= (229.5 / 255.0));

    float emissivePBR = pow(gData.pbrSpecular.a, SETTING_EMISSIVE_PBR_VALUE_CURVE);
    vec4 emissiveAlbedoCurve = vec4(vec3(SETTING_EMISSIVE_ALBEDO_COLOR_CURVE), SETTING_EMISSIVE_ALBEDO_LUM_CURVE);
    float albedoLuminanceAlternative = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, material.albedo);
    vec4 emissiveAlbedo = pow(vec4(material.albedo, albedoLuminanceAlternative), emissiveAlbedoCurve);

    float MATERIAL_LAVA_LUMINANCE = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, colors2_constants_toWorkSpace(blackBody_evalRadiance_AP0(SETTING_LAVA_TEMPERATURE)));
    float MATERIAL_FIRE_LUMINANCE = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, colors2_constants_toWorkSpace(blackBody_evalRadiance_AP0(SETTING_FIRE_TEMPERATURE)));

    float emissiveValue = emissivePBR * 0.5;
    emissiveValue = gData.materialID == 1u ? MATERIAL_LAVA_LUMINANCE : emissiveValue;
    emissiveValue = gData.materialID == 2u ? MATERIAL_FIRE_LUMINANCE : emissiveValue;
    emissiveValue *= exp2(SETTING_EMISSIVE_STRENGTH);

    material.emissive = emissiveValue * emissiveAlbedo.a * emissiveAlbedo.rgb;

    const float _64o255 = 64.0 / 255.0;
    const float _65o255 = 65.0 / 255.0;
    float step64 = step(_65o255, gData.pbrSpecular.b);
    material.porosity = linearStep(0.0, _64o255, gData.pbrSpecular.b);
    material.porosity *= 1.0 - step64;

    material.sss = linearStep(_65o255, 1.0, gData.pbrSpecular.b);
    material.sss *= step64;
    material.sss = sqrt(material.sss);

    #ifdef MATERIAL_TRANSLUCENT
    material.hardCodedIOR = gData.materialID == 3u ? 1.3333 : 1.5;
    #else
    material.hardCodedIOR = 1.0;
    #endif

    vec3 bitangent = cross(gData.geomTangent, gData.normal) * float(gData.bitangentSign);
    material.tbn = mat3(gData.geomTangent, bitangent, gData.normal);
    material.tbnInv = inverse(material.tbn);

    return material;
}

#endif