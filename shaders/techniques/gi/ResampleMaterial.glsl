#ifndef INCLUDE_techniques_gi_ResampleMaterial_glsl
#define INCLUDE_techniques_gi_ResampleMaterial_glsl a

#include "/util/Material.glsl"
#include "/util/Math.glsl"
#include "/util/SplitSumSpecular.glsl"

struct ResampleMaterial {
    float f0;
    float dielectric;
    float roughness;
};

struct ResampleBRDF {
    float diffuse;
    float specular;
    float full;
};

ResampleMaterial resampleMaterial_init() {
    ResampleMaterial material;
    material.f0 = 0.0;
    material.dielectric = 0.0;
    material.roughness = 1.0;
    return material;
}

ResampleMaterial resampleMaterial_fromMaterial(Material material) {
    ResampleMaterial resampleMaterial;
    resampleMaterial.f0 = material.f0;
    resampleMaterial.dielectric = material.dielectric;
    resampleMaterial.roughness = material.roughness;
    return resampleMaterial;
}

vec4 resampleMaterial_pack(ResampleMaterial material) {
    return vec4(sqrt(max(material.f0, 0.0)), material.dielectric, sqrt(material.roughness), 0.0);
}

ResampleMaterial resampleMaterial_unpack(vec4 packedData) {
    ResampleMaterial material;
    material.f0 = pow2(packedData.x);
    material.dielectric = packedData.y;
    material.roughness = pow2(packedData.z);
    return material;
}

float resampleMaterial_fresnel(ResampleMaterial material, float cosTheta) {
    return material.f0 + (1.0 - material.f0) * pow5(1.0 - cosTheta);
}

float resampleMaterial_ggx(ResampleMaterial material, float NDotL, float NDotV, float NDotH) {
    float result = 0.0;
    if (NDotL > 0.0) {
        float NDotH2 = pow2(NDotH);
        float a2 = pow2(material.roughness);
        float dDenominator = (1.0 - NDotH2) + a2 * NDotH2;
        float d = a2 / max(PI * pow2(dDenominator), 1e-16);
        float k = material.roughness * 0.5;
        float vL = NDotL * (1.0 - k) + k;
        float vV = saturate(NDotV) * (1.0 - k) + k;
        result = NDotL * d * rcp(vL * vV) * 0.25;
    }
    return result;
}

ResampleBRDF resampleMaterial_evalBRDF(
    ResampleMaterial material,
    float NDotL,
    float NDotV,
    float NDotH,
    float LDotH
) {
    ResampleBRDF brdf;
    float fresnel = resampleMaterial_fresnel(material, LDotH);
    float diffuseBRDF = material.dielectric * (1.0 - fresnel) * NDotL * RCP_PI;
    float specularBRDF = fresnel * resampleMaterial_ggx(material, NDotL, NDotV, NDotH);
    brdf.diffuse = diffuseBRDF;
    brdf.specular = specularBRDF;
    brdf.full = diffuseBRDF + specularBRDF;
    return brdf;
}

ResampleBRDF resampleMaterial_evalBRDF(
    ResampleMaterial material,
    vec3 normal,
    vec3 lightDir,
    vec3 viewDir
) {
    ResampleBRDF brdf;
    brdf.diffuse = 0.0;
    brdf.specular = 0.0;
    brdf.full = 0.0;

    float NDotL = dot(normal, lightDir);
    float NDotV = dot(normal, viewDir);
    vec3 halfVector = lightDir + viewDir;
    float halfLength2 = dot(halfVector, halfVector);
    if (NDotL <= 0.0 || NDotV <= 0.0 || halfLength2 <= 1e-12) {
        return brdf;
    }

    vec3 H = halfVector * inversesqrt(halfLength2);
    return resampleMaterial_evalBRDF(
        material,
        NDotL,
        NDotV,
        saturate(dot(normal, H)),
        saturate(dot(lightDir, H))
    );
}

vec3 resampleMaterial_resolveNormal(
    vec3 geomNormal,
    vec3 normal,
    vec3 viewDir
) {
    return dot(normal, viewDir) > 0.0 ? normal : geomNormal;
}

vec3 resampleMaterial_specularDenoiseFactor(ResampleMaterial material, float NDotV) {
    return splitSumSpecularDenoiseFactor(material.f0, NDotV, material.roughness);
}

#endif
