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

bool restir_isFinite(float value) {
    return !isnan(value) && !isinf(value);
}

bool restir_isFinite(vec3 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

bool restir_isFinite(vec4 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

bool restir_isPositiveFinite(float value) {
    return value > 0.0 && restir_isFinite(value);
}

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
    packedData.x = restir_isFinite(data.geomNormal)
        && dot(data.geomNormal, data.geomNormal) > 1e-8
        ? nzpacking_packNormalOct32(data.geomNormal)
        : 0u;
    packedData.y = restir_isFinite(data.hitNormal)
        && dot(data.hitNormal, data.hitNormal) > 1e-8
        ? nzpacking_packNormalOct32(data.hitNormal)
        : 0u;
    packedData.zw = packHalf4x16(clamp(data.sampleValue, 0.0, FP16_MAX));
    return packedData;
}

SpatialSampleData spatialSampleData_unpack(uvec4 packedData) {
    SpatialSampleData data;
    data.geomNormal = nzpacking_unpackNormalOct32(packedData.x);
    data.normal = vec3(0.0);
    data.hitNormal = nzpacking_unpackNormalOct32(packedData.y);
    data.sampleValue = unpackHalf4x16(packedData.zw);
    return data;
}

struct ReSTIRReservoir {
    vec4 Y;// direction and length
    float avgWY;// average unbiased contribution weight
    float m;
};

float restir_stabilizeTemporalTargetPHat(float pHat) {
    if (!restir_isFinite(pHat) || pHat <= 0.0) {
        return 0.0;
    }

    float quantizedPHat = unpackHalf2x16(packHalf2x16(vec2(pHat, 0.0))).x;
    float minStablePHat = float(SETTING_GI_TEMPORAL_REUSE_LIMIT) * 2.98023223876953125e-8;
    return restir_isFinite(quantizedPHat) && quantizedPHat >= minStablePHat ? quantizedPHat : pHat;
}

float restir_quantizeStoredTargetPHat(float pHat) {
    if (!restir_isFinite(pHat) || pHat <= 0.0) {
        return 0.0;
    }

    float quantizedPHat = unpackHalf2x16(packHalf2x16(vec2(pHat, 0.0))).x;
    float minStablePHat = float(SETTING_GI_TEMPORAL_REUSE_LIMIT) * 2.98023223876953125e-8;
    return restir_isFinite(quantizedPHat) && quantizedPHat >= minStablePHat ? quantizedPHat : 0.0;
}

ReSTIRReservoir restir_initReservoir() {
    ReSTIRReservoir reservoir;
    reservoir.Y = vec4(0.0, 0.0, 0.0, -1.0);
    reservoir.avgWY = 0.0;
    reservoir.m = 0.0;
    return reservoir;
}

bool restir_isReservoirValid(ReSTIRReservoir reservoir) {
    return reservoir.m > 0.0
        && reservoir.avgWY > 0.0
        && restir_isFinite(reservoir.m)
        && restir_isFinite(reservoir.avgWY)
        && restir_isFinite(reservoir.Y);
}

float restir_updateRand(ivec2 texelPos, uint randSeed) {
    return hash_uintToFloat(hash_44_q3(uvec4(texelPos, frameCounter, randSeed)).x);
}

bool restir_updateReservoir(inout ReSTIRReservoir reservoir, inout float wSum, vec4 X, float wi, float m, float rand) {
    if (
        !restir_isFinite(wSum)
        || wSum < 0.0
        || !restir_isFinite(reservoir.m)
        || reservoir.m < 0.0
        || !restir_isFinite(wi)
        || wi <= 0.0
        || !restir_isFinite(m)
        || m < 0.0
    ) {
        return false;
    }

    float nextWSum = wSum + wi;
    float nextM = reservoir.m + m;
    if (!restir_isFinite(nextWSum) || !restir_isFinite(nextM)) {
        return false;
    }
    wSum = nextWSum;
    reservoir.m = nextM;
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

uint restir_reservoir_packDirection(vec3 direction) {
    return restir_isFinite(direction) && dot(direction, direction) > 1e-8
        ? nzpacking_packNormalOct32(direction)
        : 0u;
}

uvec4 restir_reservoir_pack(ReSTIRReservoir reservoir) {
    uvec4 packedData = uvec4(0u);
    packedData.x = restir_reservoir_packDirection(reservoir.Y.xyz);
    packedData.y = floatBitsToUint(reservoir.m);
    packedData.z = floatBitsToUint(reservoir.avgWY);
    packedData.w = floatBitsToUint(reservoir.Y.w);
    return packedData;
}

const float RESTIR_RECONNECTION_MIN_COSINE = 0.02;
const float RESTIR_RECONNECTION_MAX_LOG2_JACOBIAN = 5.0;
const float RESTIR_RECONNECTION_MAX_DENSITY_RATIO = 32.0;

bool restir_reconnectionDensityRatioValid(
    float sourcePHat,
    float mappedTargetPHat
) {
    if (
        sourcePHat <= 0.0
        || mappedTargetPHat <= 0.0
        || !restir_isFinite(sourcePHat)
        || !restir_isFinite(mappedTargetPHat)
    ) {
        return false;
    }
    return mappedTargetPHat <= sourcePHat * RESTIR_RECONNECTION_MAX_DENSITY_RATIO
        && sourcePHat <= mappedTargetPHat * RESTIR_RECONNECTION_MAX_DENSITY_RATIO;
}

float evalTargetBRDF(
    vec3 geomNormal,
    vec3 normal,
    vec3 lightDir,
    vec3 viewDir,
    ResampleMaterial material
) {
    vec3 resolvedNormal = resampleMaterial_resolveNormal(geomNormal, normal, viewDir);
    float rawNdotL = dot(resolvedNormal, lightDir);
    float result = 0.0;

    if (
        rawNdotL > 0.0
        && dot(geomNormal, lightDir) > 0.0
        && dot(geomNormal, viewDir) > 0.0
    ) {
        ResampleBRDF brdf = resampleMaterial_evalBRDF(
            material,
            resolvedNormal,
            lightDir,
            viewDir
        );
        result = brdf.full;
    }
    return result;
}

float evalTargetFunction(
    vec3 irradiance,
    vec3 geomNormal,
    vec3 normal,
    vec3 lightDir,
    vec3 viewDir,
    ResampleMaterial material
) {
    return length(irradiance * evalTargetBRDF(
        geomNormal,
        normal,
        lightDir,
        viewDir,
        material
    ));
}

struct ShiftMapping {
    vec4 Y;
    float targetPHat;
    float unmappedTargetPHat;
};

ShiftMapping shiftMapping_init() {
    ShiftMapping mapping;
    mapping.Y = vec4(0.0, 0.0, 0.0, -1.0);
    mapping.targetPHat = 0.0;
    mapping.unmappedTargetPHat = 0.0;
    return mapping;
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
        float cosPhiSRC = -dot(canonResSRC.Y.xyz, sampleSRC.hitNormal);
        float cosPhiDST = -dot(dirSRCtoDST, sampleSRC.hitNormal);
        float sourceGeomCos = dot(sampleSRC.geomNormal, canonResSRC.Y.xyz);
        float targetGeomCos = dot(sampleDST.geomNormal, dirSRCtoDST);
        if (
            cosPhiSRC > RESTIR_RECONNECTION_MIN_COSINE
            && cosPhiDST > RESTIR_RECONNECTION_MIN_COSINE
            && sourceGeomCos > RESTIR_RECONNECTION_MIN_COSINE
            && targetGeomCos > RESTIR_RECONNECTION_MIN_COSINE
            && dot(sampleSRC.normal, canonResSRC.Y.xyz) > 0.0
        ) {
            vec3 VDST = normalize(-viewPosDST);
            float pHat = evalTargetFunction(
                sampleSRC.sampleValue.xyz,
                sampleDST.geomNormal,
                sampleDST.normal,
                dirSRCtoDST,
                VDST,
                matDST
            );
            if (pHat > 0.0 && restir_isFinite(pHat)) {
                float log2Jacobian = 2.0 * log2(canonResSRC.Y.w)
                    + log2(cosPhiDST)
                    - log2(dist2)
                    - log2(cosPhiSRC);
                if (abs(log2Jacobian) > RESTIR_RECONNECTION_MAX_LOG2_JACOBIAN) {
                    return mapping;
                }
                float jacobian_DST = exp2(log2Jacobian);
                float targetPHat = pHat * jacobian_DST;
                if (restir_reconnectionDensityRatioValid(sampleSRC.sampleValue.w, targetPHat)) {
                    mapping.Y = vec4(dirSRCtoDST, sqrt(dist2));
                    mapping.targetPHat = targetPHat;
                    mapping.unmappedTargetPHat = pHat;
                }
            }
        }
    }

    return mapping;
}
