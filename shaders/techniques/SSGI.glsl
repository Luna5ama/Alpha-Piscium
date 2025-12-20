#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Hash.glsl"
#include "/util/Math.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/techniques/SST.glsl"

#define USE_REFERENCE 1
#define SKIP_FRAMES 16
#define MAX_FRAMES 4096
#define RANDOM_FRAME (frameCounter - SKIP_FRAMES)
#define MC_SPP 1
#define SPATIAL_REUSE 1
#define SPATIAL_REUSE_SAMPLES 6
#define SPATIAL_REUSE_RADIUS 64
#define SPATIAL_REUSE_VISIBILITY_TRACE 1
#define SPATIAL_REUSE_FEEDBACK 16

/*layout(rgba32ui) uniform uimage2D uimg_csrgba32ui;

struct InitialSampleData {
    vec4 directionAndLength;
    vec3 hitRadiance;
};

InitialSampleData initialSampleData_init() {
    InitialSampleData data;
    data.directionAndLength = vec4(0.0);
    data.hitRadiance = vec3(0.0);
    return data;
}

uvec4 initialSampleData_pack(InitialSampleData data) {
    uvec4 packedData;
    packedData.x = nzpacking_packNormalOct32(data.directionAndLength.xyz);
    packedData.y = floatBitsToUint(data.directionAndLength.w);
    packedData.zw = packHalf4x16(vec4(data.hitRadiance, 0.0));
    return packedData;
}

InitialSampleData initialSampleData_unpack(uvec4 packedData) {
    InitialSampleData data;
    data.directionAndLength.xyz = nzpacking_unpackNormalOct32(packedData.x);
    data.directionAndLength.w = uintBitsToFloat(packedData.y);
    vec4 hitRadianceAndPadding = unpackHalf4x16(packedData.zw);
    data.hitRadiance = hitRadianceAndPadding.rgb;
    return data;
}

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
    reservoir.Y = vec4(0.0);
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

ReSTIRReservoir restir_loadReservoir(ivec2 texelPos, int swapIndex) {
    texelPos = clamp(texelPos, ivec2(0), uval_mainImageSizeI - 1);
    ivec2 sampleTexelPos = texelPos;
    if (swapIndex == 0) {
        sampleTexelPos = csrgba32ui_restir1_texelToTexel(texelPos);
    } else {
        sampleTexelPos = csrgba32ui_restir2_texelToTexel(texelPos);
    }
    uvec4 data1 = imageLoad(uimg_csrgba32ui, sampleTexelPos);

    ReSTIRReservoir reservoir;
    //    reservoir.Y.x = bitfieldExtract(data1.x, 0, 12);
    //    reservoir.Y.y = bitfieldExtract(data1.x, 12, 12);
    //    reservoir.m = bitfieldExtract(data1.x, 24, 8);
//    reservoir.Y = unpackUInt2x16(data1.x);
    reservoir.Y.xyz = nzpacking_unpackNormalOct32(data1.x);

    //    reservoir.pY = uintBitsToFloat(data1.y);

    uvec2 temp = unpackUInt2x16(data1.y);
    reservoir.m = temp.x;
    reservoir.age = temp.y;

    reservoir.avgWY = uintBitsToFloat(data1.z);
//    reservoir.wSum = uintBitsToFloat(data1.w);
    reservoir.Y.w = uintBitsToFloat(data1.w);
    reservoir.texelPos = texelPos;
    return reservoir;
}

void restir_storeReservoir(ivec2 texelPos, ReSTIRReservoir reservoir, int swapIndex) {
    texelPos = clamp(texelPos, ivec2(0), uval_mainImageSizeI - 1);
    ivec2 storeTexelPos = texelPos;
    if (swapIndex == 0) {
        storeTexelPos = csrgba32ui_restir1_texelToTexel(texelPos);
    } else {
        storeTexelPos = csrgba32ui_restir2_texelToTexel(texelPos);
    }
    uvec4 data1 = uvec4(0u);
    //    data1.x = bitfieldInsert(data1.x, reservoir.Y.x, 0, 12);
    //    data1.x = bitfieldInsert(data1.x, reservoir.Y.y, 12, 12);
    //    data1.x = bitfieldInsert(data1.x, min(reservoir.m, 255u), 24, 8);
//    data1.x = packUInt2x16(reservoir.Y);
    data1.x = nzpacking_packNormalOct32(reservoir.Y.xyz);

    //    data1.y = floatBitsToUint(reservoir.pY);
//    data1.y = reservoir.m;

    data1.y = packUInt2x16(uvec2(reservoir.m, reservoir.age));

    data1.z = floatBitsToUint(reservoir.avgWY);
//    data1.w = floatBitsToUint(reservoir.wSum);
    data1.w = floatBitsToUint(reservoir.Y.w);
    imageStore(uimg_csrgba32ui, storeTexelPos, data1);
}

vec4 ssgiEvalF2(vec3 viewPos, vec3 sampleDirView) {
    vec4 result = vec4(0.0);

    SSTResult sstResult = sst_trace(viewPos, sampleDirView, 0.01);

    if (sstResult.hit) {
        vec2 hitTexelPosF = floor(sstResult.hitScreenPos.xy * uval_mainImageSize);
        vec2 hitTexelCenter = hitTexelPosF + 0.5;
        vec2 roundedHitScreenPos = hitTexelCenter * uval_mainImageSizeRcp;
        float hitViewZ = coords_reversedZToViewZ(sstResult.hitScreenPos.z, near);
        vec3 hitViewPos = coords_toViewCoord(roundedHitScreenPos, hitViewZ, global_camProjInverse);
        result.w = length(hitViewPos - viewPos);

        vec3 hitRadiance = texelFetch(usam_temp2, ivec2(hitTexelPosF), 0).rgb;
        result.xyz = hitRadiance;

    }

    return result;
}

vec3 ssgiEvalF(vec3 viewPos, GBufferData gData, vec3 sampleDirView, out float hitDistance) {
    hitDistance = -1.0;
    vec3 result = vec3(0.0);

    SSTResult sstResult = sst_trace(viewPos, sampleDirView, 0.01);

    if (sstResult.hit) {
        vec2 hitTexelPosF = floor(sstResult.hitScreenPos.xy * uval_mainImageSize);
        vec2 hitTexelCenter = hitTexelPosF + 0.5;
        vec2 roundedHitScreenPos = hitTexelCenter * uval_mainImageSizeRcp;
        float hitViewZ = coords_reversedZToViewZ(sstResult.hitScreenPos.z, near);
        vec3 hitViewPos = coords_toViewCoord(roundedHitScreenPos, hitViewZ, global_camProjInverse);
        hitDistance = length(hitViewPos - viewPos);

        vec3 hitRadiance = texelFetch(usam_temp2, ivec2(hitTexelPosF), 0).rgb;
        float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
        vec3 f = brdf * hitRadiance;
        result = f;
    }

    return result;
}*/

vec3 ssgiRef(ivec2 texelPos, uint finalIndex) {
//    uint finalIndex = RANDOM_FRAME;
    vec3 result = vec3(0.0);
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
            ivec2 hitTexelPos = ivec2(sstResult.hitScreenPos.xy * uval_mainImageSize);
            vec3 hitRadiance = transient_giRadianceInput_fetch(hitTexelPos).rgb;
            GBufferData hitGData = gbufferData_init();
            gbufferData1_unpack_world(texelFetch(usam_gbufferData1, hitTexelPos, 0), hitGData);

            float hitCosTheta = saturate(dot(hitGData.normal, -sampleDirView));

            hitRadiance *= hitCosTheta;

            float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
            vec3 f = brdf * hitRadiance;
            result = f / samplePdf;
        }
    }

    return result;
}