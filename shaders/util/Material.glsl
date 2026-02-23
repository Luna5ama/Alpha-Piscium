#ifndef INCLUDE_util_Material_glsl
#define INCLUDE_util_Material_glsl a
#include "BlackBody.glsl"
#include "Colors.glsl"
#include "Colors2.glsl"
#include "GBufferData.glsl"
#include "Math.glsl"
#include "Rand.glsl"
#include "HardcodedPBR.glsl"
#include "MaterialIDConst.glsl"

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
    mat3 geomTbn;
    mat3 geomTbnInv;
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
    HardcodedPBR hardcoded = hardcodedpbr_decode(gData.materialID);
    bool isWater = gData.materialID == MATERIAL_ID_WATER;

    material.albedo = colors2_material_toWorkSpace(gData.albedo);
    vec4 emissiveAlbedoCurve = vec4(vec3(SETTING_EMISSIVE_ALBEDO_COLOR_CURVE), SETTING_EMISSIVE_ALBEDO_LUM_CURVE);
    float albedoLuma = colors2_colorspaces_luma(COLORS2_MATERIAL_COLORSPACE, gData.albedo);

    #if defined(MC_TEXTURE_FORMAT_LAB_PBR) && SETTING_PBR_MATERIAL == 1 || SETTING_PBR_MATERIAL == 2
    bool useBuiltInPBR = gData.forceBuiltInPBR;
    #else
    bool useBuiltInPBR = true;
    #endif

    float roughness;
    float emissivePBR;

    if (useBuiltInPBR) {
        roughness = hardcoded.roughness;
        emissivePBR = hardcoded.emissive;
        emissiveAlbedoCurve.a += 1.0;
        albedoLuma = smoothstep(0.0, 1.0, albedoLuma);
    } else {
        roughness = 1.0 - gData.pbrSpecular.r;
        emissivePBR = gData.pbrSpecular.a;
        emissivePBR = pow(emissivePBR, SETTING_EMISSIVE_PBR_VALUE_CURVE);
    }

    roughness = pow2(roughness);
    roughness *= _MATERIAL_ROUGHNESS_MULTIPLIER;

    #ifdef MATERIAL_TRANSLUCENT
    roughness = isWater ? _MATERIAL_WATER_ROUGHNESS : roughness;
    #endif

    emissivePBR = pow(emissivePBR, SETTING_EMISSIVE_PBR_VALUE_CURVE);

    material.roughness = roughness;
    material.f0 = gData.pbrSpecular.g;

    #if SETTING_MINIMUM_F0_FACTOR > 0
    material.f0 = max(material.f0, _MATERIAL_F0_EPSILON);
    #endif
    material.metallic = float(material.f0 >= (229.5 / 255.0));

    vec4 emissiveAlbedo = pow(vec4(gData.albedo, albedoLuma), emissiveAlbedoCurve);
    emissiveAlbedo.rgb = colors2_material_toWorkSpace(emissiveAlbedo.rgb);

    float emissiveValue = emissivePBR * 0.5;
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
    material.hardCodedIOR = isWater ? 1.3333 : hardcoded.ior;
    #else
    material.hardCodedIOR = hardcoded.ior;
    #endif

    vec3 newTangent = normalize(gData.geomTangent - gData.normal * dot(gData.normal, gData.geomTangent));
    vec3 newBitangent = cross(newTangent, gData.normal) * float(gData.bitangentSign);
    material.tbn = mat3(newTangent, newBitangent, gData.normal);
    material.tbnInv = inverse(material.tbn);

    vec3 geomBitangent = cross(gData.geomTangent, gData.geomNormal) * float(gData.bitangentSign);
    material.geomTbn = mat3(gData.geomTangent, geomBitangent, gData.geomNormal);
    material.geomTbnInv = inverse(material.geomTbn);

    return material;
}

#endif