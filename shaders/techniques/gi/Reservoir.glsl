/*
    References:
        [WYM23] Wyman, Chris, et al. "A Gentle Introduction to ReSTIR". SIGGRAPH 2023.
            https://intro-to-restir.cwyman.org/
        [ANA23] Anagnostou, Kostas. "A Gentler Introduction to ReSTIR". Interplay of Light. 2023.
            https://interplayoflight.wordpress.com/2023/12/17/a-gentler-introduction-to-restir/
        [ALE22] Alegruz. "Screen-Space-ReSTIR-GI". GitHub. 2022.
            https://github.com/Alegruz/Screen-Space-ReSTIR-GI
            BSD 3-Clause License. Copyright (c) 2022, Alegruz.

        You can find full license texts in /licenses

    Other Credits:
        - Belmu (https://github.com/BelmuTM) - Advice on ReSTIR.
*/
#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Hash.glsl"
#include "/util/Math.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/BSDF.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/ResampleMaterial.glsl"

#define RESTIR_REUSE_TILE_SIZE 256
#define RESTIR_REUSE_TILE_SIZE_HALF 128
#define RESTIR_REUSE_TILE_BITS 8
#define RESTIR_REUSE_TILE_MASK 255

struct SpatialSampleData {
    vec3 geomNormal;
    vec3 normal;
    vec3 hitNormal;
    vec4 sampleValue;
};

SpatialSampleData spatialSampleData_init() {
    SpatialSampleData data;
    data.geomNormal = vec3(0.0);
    data.normal = vec3(0.0);
    data.hitNormal = vec3(0.0);
    data.sampleValue = vec4(0.0);
    return data;
}

uvec4 spatialSampleData_pack(SpatialSampleData data) {
    uvec4 packedData;
    nzpacking_packNormalOct16(packedData.x, data.geomNormal, data.hitNormal);
    packedData.y = nzpacking_packNormalOct32(data.normal);
    packedData.zw = packHalf4x16(clamp(data.sampleValue, 0.0, FP16_MAX));
    return packedData;
}

SpatialSampleData spatialSampleData_unpack(uvec4 packedData) {
    SpatialSampleData data;
    nzpacking_unpackNormalOct16(packedData.x, data.geomNormal, data.hitNormal);
    data.normal = nzpacking_unpackNormalOct32(packedData.y);
    data.sampleValue = unpackHalf4x16(packedData.zw);
    return data;
}

struct ReSTIRReservoir {
    vec4 Y;// direction and length
    float avgWY;// average unbiased contribution weight
    float m;
};

ReSTIRReservoir restir_initReservoir() {
    ReSTIRReservoir reservoir;
    reservoir.Y = vec4(0.0, 0.0, 0.0, -1.0);
    reservoir.avgWY = 0.0;
    reservoir.m = 0.0;
    return reservoir;
}

bool restir_isReservoirValid(ReSTIRReservoir reservoir) {
    return reservoir.m > 0.0;
}

const float EPSILON = 0.0000001;

float restir_updateRand(ivec2 texelPos, uint randSeed) {
    return hash_uintToFloat(hash_44_q3(uvec4(texelPos, frameCounter, randSeed)).x);
}

bool restir_updateReservoir(inout ReSTIRReservoir reservoir, inout float wSum, vec4 X, float wi, float m, float rand) {
    wSum += wi;
    reservoir.m += m;
    bool updateCond = rand < wi / wSum;
    if (updateCond) {
        reservoir.Y = X;
    }

    return updateCond;
}

ReSTIRReservoir restir_reservoir_unpack(uvec4 packedData) {
    ReSTIRReservoir reservoir;
    reservoir.Y.xyz = nzpacking_unpackNormalOct32(packedData.x);
    reservoir.m = uintBitsToFloat(packedData.y);
    reservoir.avgWY = uintBitsToFloat(packedData.z);
    reservoir.Y.w = uintBitsToFloat(packedData.w);
    return reservoir;
}

uvec4 restir_reservoir_pack(ReSTIRReservoir reservoir) {
    uvec4 packedData = uvec4(0u);
    packedData.x = nzpacking_packNormalOct32(reservoir.Y.xyz);
    packedData.y = floatBitsToUint(reservoir.m);
    packedData.z = floatBitsToUint(reservoir.avgWY);
    packedData.w = floatBitsToUint(reservoir.Y.w);
    return packedData;
}

float evalTargetFunction(vec3 irradiance, vec3 normal, vec3 lightDir, vec3 viewDir, ResampleMaterial material) {
    // Assumes rawNdotL is the un-clamped dot product. Ensure no pre-saturation occurred if passed from an external scope.
    float rawNdotL = dot(normal, lightDir);
    float result = 0.0;

    if (rawNdotL > 0.0) {
        float rawNdotV = dot(normal, viewDir);
        float LdotV    = dot(lightDir, viewDir);

        // Compute inverse length of ||L+V||. The 1e-5 bias prevents rsqrt(0) NaN generation when L and V are perfectly opposed (LdotV = -1.0).
        float invLen = inversesqrt(max(2.0 + 2.0 * LdotV, 1e-5));

        // Pure scalar expansion. Replaces explicit vec3 H = normalize(...) and subsequent vector dot products.
        float NdotV = saturate(rawNdotV);
        float NdotH = saturate((rawNdotL + rawNdotV) * invLen);

        // LdotH is mathematically bounded to [0, 1] (max angle between L and V is 180 deg, bounding half-angle to 90 deg). saturate() omitted.
        float LdotH = (1.0 + LdotV) * invLen;

        ResampleBRDF brdf = resampleMaterial_evalBRDF(material, rawNdotL, NdotV, NdotH, LdotH);
        vec3 radiance = irradiance * brdf.full;
        result = length(radiance);
    }
    return result;
}

struct ShiftMapping {
    vec4 Y;
    float targetPHat;
};

ShiftMapping shiftMapping_init() {
    ShiftMapping mapping;
    mapping.Y = vec4(0.0, 0.0, 0.0, -1.0);
    mapping.targetPHat = 0.0;
    return mapping;
}

bool shiftMapping_hasTarget(ShiftMapping mapping) {
    return abs(mapping.targetPHat) > 0.0;
}

bool shiftMapping_isReusable(ShiftMapping mapping) {
    return mapping.targetPHat > 0.0;
}


ShiftMapping evaluateShiftMapping(
    ReSTIRReservoir canonResSRC,
    ResampleMaterial matDST,
    SpatialSampleData sampleDST, SpatialSampleData sampleSRC,
    vec3 viewPosDST, vec3 viewPosSRC
) {
    ShiftMapping mapping = shiftMapping_init();

    vec3 hitViewPosSRC = viewPosSRC + canonResSRC.Y.xyz * canonResSRC.Y.w;
    vec3 diffSRCtoDST = hitViewPosSRC - viewPosDST;
    float dist2 = dot(diffSRCtoDST, diffSRCtoDST);
    if (dist2 > 1e-6 && canonResSRC.Y.w > 1e-6 && restir_isReservoirValid(canonResSRC)) {
        vec3 dirSRCtoDST = diffSRCtoDST * inversesqrt(dist2);
        float cosSRC = dot(sampleSRC.normal, canonResSRC.Y.xyz);
        float cosPhiSRC = -dot(canonResSRC.Y.xyz, sampleSRC.hitNormal);
        float cosPhiDST = -dot(dirSRCtoDST, sampleSRC.hitNormal);
        if (cosPhiSRC > 0.0 && cosPhiDST > 0.0) {
            vec3 VDST = normalize(-viewPosDST);
            float pHat = evalTargetFunction(sampleSRC.sampleValue.xyz, sampleDST.normal, dirSRCtoDST, VDST, matDST);
            if (pHat > 0.0) {
                float jacobian_DST = clamp(((canonResSRC.Y.w * canonResSRC.Y.w) * cosPhiDST) / (dist2 * cosPhiSRC), 0.0, 256.0);
                mapping.Y = vec4(dirSRCtoDST, sqrt(dist2));
                mapping.targetPHat = pHat * jacobian_DST;
                if (cosSRC <= 0.0) {
                    mapping.targetPHat = -mapping.targetPHat;
                }
            }
        }
    }

    return mapping;
}
