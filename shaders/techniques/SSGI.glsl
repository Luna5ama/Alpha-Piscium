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
    vec3 hitRadiance;
};

SpatialSampleData spatialSampleData_init() {
    SpatialSampleData data;
    data.geomNormal = vec3(0.0);
    data.normal = vec3(0.0);
    data.hitNormal = vec3(0.0);
    data.hitRadiance = vec3(0.0);
    return data;
}

uvec4 spatialSampleData_pack(SpatialSampleData data) {
    uvec4 packedData;
    nzpacking_packNormalOct16(packedData.x, data.geomNormal, data.hitNormal);
    packedData.y = nzpacking_packNormalOct32(data.normal);
    packedData.zw = packHalf4x16(vec4(data.hitRadiance, 0.0));
    return packedData;
}

SpatialSampleData spatialSampleData_unpack(uvec4 packedData) {
    SpatialSampleData data;
    nzpacking_unpackNormalOct16(packedData.x, data.geomNormal, data.hitNormal);
    data.normal = nzpacking_unpackNormalOct32(packedData.y);
    vec4 hitRadianceAndPadding = unpackHalf4x16(packedData.zw);
    data.hitRadiance = hitRadianceAndPadding.rgb;
    return data;
}

struct ReSTIRReservoir {
    vec4 Y;// direction and length
    float avgWY;// average unbiased contribution weight
//    float wSum; // weight sum of all processed samples
    uint m;
    uint age;
    ivec2 texelPos;
};

ReSTIRReservoir restir_initReservoir(ivec2 texelPos) {
    ReSTIRReservoir reservoir;
    reservoir.Y = vec4(0.0, 0.0, 0.0, -1.0);
    reservoir.avgWY = 0.0;
    //    reservoir.wSum = 0.0;
    reservoir.m = 0u;
    reservoir.age = 0u;
    reservoir.texelPos = texelPos;
    return reservoir;
}

// x: candidate ray direction angle
// w: candidate ray sample weight
// c: candidate light contribution
//void restir_updateReservoir(inout ReSTIRReservoir reservoir, float x, float w, float c) {
//    reservoir.wSum += w;
//    reservoir.m += c;
//    reservoir.y = restir_nextRandomFloat(reservoir) < w / reservoir.wSum ? x : reservoir.y;
//}

bool restir_isReservoirValid(ReSTIRReservoir reservoir) {
    return reservoir.m > 0u;
}

const float EPSILON = 0.0000001;

// X: candidate sample texel position
// pHatX: candidate sample target function value
// pX: candidate sample pdf value
//bool restir_updateReservoir(inout ReSTIRReservoir reservoir, ivec2 X, float pHatX, float pX, uint m, float rand) {
//    if (pX > EPSILON) {
//        float WXi = rcp(pX); // WXi: unbiased contribution weight
//        reservoir.m += m;
////        float mi = float(m) / float(max(1u, reservoir.m));
//        float mi = pX / (pX + reservoir.pY);
//        float wi = mi * pHatX * WXi; // Wi: Resampling weight
//
//        reservoir.wSum += wi;
//        if (rand < wi / reservoir.wSum) {
//            reservoir.pY = pX;
//            reservoir.Y = uvec2(X);
//            reservoir.wY = rcp(pHatX) * reservoir.wSum;
//            return true;
//        }
//    }
//
//    return false;
//}
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

vec3 sampleIrradiance(ivec2 texelPos, vec3 outgoingDirection) {
    GBufferData hitGData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), hitGData);

    float hitCosTheta = saturate(dot(hitGData.normal, outgoingDirection));
    vec3 hitRadiance = transient_giRadianceInput1_fetch(texelPos).rgb;
    vec3 hitEmissive = transient_giRadianceInput2_fetch(texelPos).rgb;

    return hitCosTheta * hitRadiance + hitEmissive;
}

vec4 ssgiEvalF2(vec3 viewPos, vec3 sampleDirView) {
    vec4 result = vec4(0.0, 0.0, 0.0, -1.0);

    SSTResult sstResult = sst_trace(viewPos, sampleDirView, 0.01);

    if (sstResult.hit) {
        vec2 hitTexelPosF = floor(sstResult.hitScreenPos.xy * uval_mainImageSize);
        ivec2 hitTexelPos = ivec2(hitTexelPosF);
        vec2 hitTexelCenter = hitTexelPosF + 0.5;
        vec2 roundedHitScreenPos = hitTexelCenter * uval_mainImageSizeRcp;
        float hitViewZ = coords_reversedZToViewZ(sstResult.hitScreenPos.z, near);
        vec3 hitViewPos = coords_toViewCoord(roundedHitScreenPos, hitViewZ, global_camProjInverse);
        float hitDistance = length(hitViewPos - viewPos);
        result.w = length(hitViewPos - viewPos);
        result.xyz = sampleIrradiance(hitTexelPos, -sampleDirView);
    }

    return result;
}

vec3 ssgiEvalF(vec3 viewPos, GBufferData gData, vec3 sampleDirView, out float hitDistance) {
    hitDistance = -1.0;
    vec3 result = vec3(0.0);

    SSTResult sstResult = sst_trace(viewPos, sampleDirView, 0.01);

    if (sstResult.hit) {
        vec2 hitTexelPosF = floor(sstResult.hitScreenPos.xy * uval_mainImageSize);
        ivec2 hitTexelPos = ivec2(hitTexelPosF);
        vec2 hitTexelCenter = hitTexelPosF + 0.5;
        vec2 roundedHitScreenPos = hitTexelCenter * uval_mainImageSizeRcp;
        float hitViewZ = coords_reversedZToViewZ(sstResult.hitScreenPos.z, near);
        vec3 hitViewPos = coords_toViewCoord(roundedHitScreenPos, hitViewZ, global_camProjInverse);
        hitDistance = length(hitViewPos - viewPos);

        vec3 hitRadiance = sampleIrradiance(hitTexelPos, -sampleDirView);
        float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
        vec3 f = brdf * hitRadiance;
        result = f;
    }

    return result;
}

vec4 ssgiRef(ivec2 texelPos, uint finalIndex) {
//    uint finalIndex = RANDOM_FRAME;
    vec4 result = vec4(0.0, 0.0, 0.0, -1.0);
    GBufferData gData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
    gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
    Material material = material_decode(gData);
    #define RAND_TYPE 1
    #if RAND_TYPE == 0
    vec2 rand2 = rand_r2Seq2(finalIndex);
    #elif RAND_TYPE == 1
    vec2 rand2 = hash_uintToFloat(hash_33_q3(uvec3(texelPos, finalIndex)).xy);
    #else
    ivec2 stbnPos = texelPos + ivec2(rand_r2Seq2(finalIndex / 64) * vec2(128, 128));
    vec2 rand2 = rand_stbnVec2(stbnPos, finalIndex % 64u);
    #endif

    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).x;

    if (viewZ > -65536.0){
        vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
//        vec4 sampleDirTangentAndPdf = rand_sampleInHemisphere(rand2);
        vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

        SSTResult sstResult = sst_trace(viewPos, sampleDirView, 0.01);
        float samplePdf = sampleDirTangentAndPdf.w;

        if (sstResult.hit) {
            //        vec3 hitRadiance = texelFetch(usam_temp2, ivec2(sstResult.hitScreenPos.xy * uval_mainImageSize), 0).rgb;
            vec2 hitTexelPosF = floor(sstResult.hitScreenPos.xy * uval_mainImageSize);
            ivec2 hitTexelPos = ivec2(hitTexelPosF);
            vec2 hitTexelCenter = hitTexelPosF + 0.5;
            vec2 roundedHitScreenPos = hitTexelCenter * uval_mainImageSizeRcp;

            vec3 hitRadiance = sampleIrradiance(hitTexelPos, -sampleDirView);

            float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
            vec3 f = brdf * hitRadiance;
            result.rgb = f / samplePdf;

            float hitViewZ = coords_reversedZToViewZ(sstResult.hitScreenPos.z, near);
            vec3 hitViewPos = coords_toViewCoord(roundedHitScreenPos, hitViewZ, global_camProjInverse);
            result.w = length(hitViewPos - viewPos);
        }
    }

    return result;
}