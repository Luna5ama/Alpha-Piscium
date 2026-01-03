#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Hash.glsl"
#include "/util/Math.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/techniques/SST.glsl"
#include "/techniques/gi/Common.glsl"

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
    packedData.zw = packHalf4x16(data.sampleValue);
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
    uint m;
    uint age;
    ivec2 texelPos;
};

ReSTIRReservoir restir_initReservoir(ivec2 texelPos) {
    ReSTIRReservoir reservoir;
    reservoir.Y = vec4(0.0, 0.0, 0.0, -1.0);
    reservoir.avgWY = 0.0;
    reservoir.m = 0u;
    reservoir.age = 0u;
    reservoir.texelPos = texelPos;
    return reservoir;
}

bool restir_isReservoirValid(ReSTIRReservoir reservoir) {
    return reservoir.m > 0u;
}

const float EPSILON = 0.0000001;

bool restir_updateReservoir(inout ReSTIRReservoir reservoir, inout float wSum, vec4 X, float wi, uint m, uint age, float rand) {
    wSum += wi;
    reservoir.m += m;
    bool updateCond = rand < wi / wSum;
    if (updateCond) {
        reservoir.Y = X;
        reservoir.age = age;
    }

    return updateCond;
}

ReSTIRReservoir restir_reservoir_unpack(uvec4 packedData) {
    ReSTIRReservoir reservoir;
    reservoir.Y.xyz = nzpacking_unpackNormalOct32(packedData.x);
    uvec2 temp = unpackUInt2x16(packedData.y);
    reservoir.m = temp.x;
    reservoir.age = temp.y;
    reservoir.avgWY = uintBitsToFloat(packedData.z);
    reservoir.Y.w = uintBitsToFloat(packedData.w);
    return reservoir;
}

uvec4 restir_reservoir_pack(ReSTIRReservoir reservoir) {
    uvec4 packedData = uvec4(0u);
    packedData.x = nzpacking_packNormalOct32(reservoir.Y.xyz);
    packedData.y = packUInt2x16(uvec2(reservoir.m, reservoir.age));
    packedData.z = floatBitsToUint(reservoir.avgWY);
    packedData.w = floatBitsToUint(reservoir.Y.w);
    return packedData;
}

vec3 sampleIrradiance(ivec2 texelPos, ivec2 hitTexelPos, vec3 outgoingDirection) {
    GBufferData hitGData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, hitTexelPos, 0), hitGData);

    float hitCosTheta = saturate(dot(hitGData.geomNormal, outgoingDirection));
    vec3 hitRadiance = transient_giRadianceInput1_fetch(hitTexelPos).rgb;
    vec3 hitEmissive = transient_giRadianceInput2_fetch(hitTexelPos).rgb;
    vec3 selfHitEmissive = transient_giRadianceInput2_fetch(texelPos).rgb;

    return hitRadiance * float(hitCosTheta > 0.0) + hitEmissive * float(all(lessThan(selfHitEmissive, vec3(0.0001))));
}