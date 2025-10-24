#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Hash.glsl"
#include "/util/Math.glsl"
#include "/techniques/textile/CSRGBA32UI.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/techniques/SST.glsl"

#define USE_REFERENCE 0
#define SKIP_FRAMES 32
#define MAX_FRAMES 4096
#define RANDOM_FRAME (frameCounter - SKIP_FRAMES)

layout(rgba32ui) uniform uimage2D uimg_csrgba32ui;

struct ReSTIRReservoir {
    uvec2 Y; // sample hit texel position
//    float pY; // sample pdf value
    float wY; // sample unbiased contribution weight
    float wSum; // weight sum of all processed samples
    uint m;
    ivec2 texelPos;
};

ReSTIRReservoir restir_initReservoir(ivec2 texelPos) {
    ReSTIRReservoir reservoir;
    reservoir.Y = uvec2(65535u);
    reservoir.wY = 0.0;
    reservoir.wSum = 0.0;
    reservoir.m = 0u;
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
    return reservoir.Y != uvec2(65535u);
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
bool restir_updateReservoir(inout ReSTIRReservoir reservoir, ivec2 X, float wi, uint m, float rand) {
//    if (wi > EPSILON && !isnan(wi)) {
        reservoir.wSum += wi;
        reservoir.m += m;
        if (rand < wi / reservoir.wSum) {
//            reservoir.pY = pX;
            reservoir.Y = uvec2(X);
            return true;
        }
//    }

    return false;
}

void restir_updateReservoirWY(inout ReSTIRReservoir reservoir, float pHatY) {
    reservoir.wY = pHatY > EPSILON ? reservoir.wSum / pHatY / float(reservoir.m) : 0.0;
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
    reservoir.Y = unpackUInt2x16(data1.x);

//    reservoir.pY = uintBitsToFloat(data1.y);

    reservoir.m = data1.y;
    reservoir.wY = uintBitsToFloat(data1.z);
    reservoir.wSum = uintBitsToFloat(data1.w);
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
    data1.x = packUInt2x16(reservoir.Y);

//    data1.y = floatBitsToUint(reservoir.pY);
    data1.y = reservoir.m;
    data1.z = floatBitsToUint(reservoir.wY);
    data1.w = floatBitsToUint(reservoir.wSum);
    imageStore(uimg_csrgba32ui, storeTexelPos, data1);
}

vec3 ssgiEvalF(vec3 viewPos, GBufferData gData, vec3 sampleDirView, out ivec2 hitTexelPos) {
    hitTexelPos = ivec2(65535);
    vec3 result = vec3(0.0);;

    SSTResult sstResult = sst_trace(viewPos, sampleDirView, 0.01);

    if (sstResult.hit) {
        vec3 hitRadiance = texture(usam_temp2, sstResult.hitScreenPos.xy).rgb;
        float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
        vec3 f = brdf * hitRadiance;
        result = f;
        hitTexelPos = ivec2(sstResult.hitScreenPos.xy * uval_mainImageSize);
    }

    return result;
}

#define SSP 1u

vec3 ssgiRef(ivec2 texelPos) {
    uint finalIndex = RANDOM_FRAME;
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

    vec4 sampleDirTangentAndPdf = rand_sampleInCosineWeightedHemisphere(rand2);
    vec3 sampleDirView = normalize(material.tbn * sampleDirTangentAndPdf.xyz);
    float viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).x;
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

    SSTResult sstResult = sst_trace(viewPos, sampleDirView, 0.01);
    float samplePdf = sampleDirTangentAndPdf.w;

    if (sstResult.hit) {
        vec3 hitRadiance = texture(usam_temp2, sstResult.hitScreenPos.xy).rgb;
        float brdf = saturate(dot(gData.normal, sampleDirView)) / PI;
        vec3 f = brdf * hitRadiance;
        result = f / samplePdf;
    }

    return result;
}