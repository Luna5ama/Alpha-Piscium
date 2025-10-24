#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Hash.glsl"
#include "/util/Math.glsl"
#include "/techniques/textile/CSRGBA32UI.glsl"

#define USE_REFERENCE 1
#define SKIP_FRAMES 32
#define MAX_FRAMES 1
#define RANDOM_FRAME (frameCounter - SKIP_FRAMES)

layout(rgba32ui) uniform uimage2D uimg_csrgba32ui;

struct ReSTIRReservoir {
    uvec2 Y; // sample hit texel position
    float wY; // sample unbiased contribution weight
    float wSum; // weight sum of all processed samples
    ivec2 texelPos;
};

ReSTIRReservoir restir_initReservoir(ivec2 texelPos) {
    ReSTIRReservoir reservoir;
    reservoir.Y = uvec2(99999u);// initial ray direction angle
    reservoir.wY = 0.0;
    reservoir.wSum = 0.0;
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
    return reservoir.Y != uvec2(99999u);
}

// X: candidate sample texel position
// pHatX: candidate sample target function value
// pX: candidate sample pdf value
bool restir_updateReservoir(inout ReSTIRReservoir reservoir, ivec2 X, float pHatX, float pX, float rand) {
    float WXi = rcp(pX); // WXi: unbiased contribution weight
    float wi = pHatX * WXi; // Wi: Resampling weight

    reservoir.wSum += wi;
    if (rand < wi / reservoir.wSum) {
        reservoir.Y = uvec2(X);
        return true;
    }

    return false;
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
    reservoir.Y = unpackUInt2x16(data1.x);
    reservoir.wY = uintBitsToFloat(data1.y);
    reservoir.wSum = uintBitsToFloat(data1.z);
    reservoir.texelPos = texelPos;
    return reservoir;
}

void restir_storeReservoir(ivec2 texelPos, ReSTIRReservoir reservoir, int swapIndex) {
    texelPos = clamp(texelPos, ivec2(0), uval_mainImageSizeI - 1);
    ivec2 storeTexelPos = texelPos;
    if (swapIndex == 0) {
        storeTexelPos = csrgba32ui_restir2_texelToTexel(texelPos);
    } else {
        storeTexelPos = csrgba32ui_restir1_texelToTexel(texelPos);
    }
    uvec4 data1 = uvec4(0u);
    data1.x = packUInt2x16(reservoir.Y);
    data1.y = floatBitsToUint(reservoir.wY);
    data1.z = floatBitsToUint(reservoir.wSum);
    imageStore(uimg_csrgba32ui, storeTexelPos, data1);
}