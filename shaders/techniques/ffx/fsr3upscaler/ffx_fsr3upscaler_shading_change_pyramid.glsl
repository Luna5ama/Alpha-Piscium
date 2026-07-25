// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

FFX_GROUPSHARED FfxUInt32 spdCounter;

void SpdIncreaseAtomicCounter(FfxUInt32 slice) {
    SPD_IncreaseAtomicCounter(spdCounter);
}

FfxUInt32 SpdGetAtomicCounter() {
    return spdCounter;
}

void SpdResetAtomicCounter(FfxUInt32 slice) {
    SPD_ResetAtomicCounter();
}

#ifndef FFX_SPD_PACKED_ONLY
FFX_GROUPSHARED FfxFloat32 spdIntermediateR[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateG[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateB[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateA[16][16];

struct SampleSet {
    FfxFloat32 s0;
    FfxFloat32 s1;
    FfxFloat32 s2;
    FfxFloat32 s3;
    FfxFloat32 s4;
};

#define CompareSwap(a, b) \
{ \
    FfxFloat32 fTmp = ffxMin(fSet.a, fSet.b); \
    fSet.b = ffxMax(fSet.a, fSet.b); \
    fSet.a = fTmp; \
}

void SortSet(FFX_PARAMETER_INOUT SampleSet fSet) {
    CompareSwap(s0, s3);
    CompareSwap(s1, s4);
    CompareSwap(s0, s2);
    CompareSwap(s1, s3);
    CompareSwap(s0, s1);
    CompareSwap(s2, s4);
    CompareSwap(s1, s2);
    CompareSwap(s3, s4);
    CompareSwap(s2, s3);
}

FfxFloat32 GetSample(SampleSet fSet, FfxInt32 index) {
    if (index == 0) return fSet.s0;
    if (index == 1) return fSet.s1;
    if (index == 2) return fSet.s2;
    if (index == 3) return fSet.s3;
    return fSet.s4;
}

FfxFloat32 LoadLumaSample(FfxInt32x2 basePos, FfxInt32x2 offset, FfxInt32x2 renderSize) {
    const FfxInt32x2 samplePos = ClampLoad(basePos, offset, renderSize);
    FfxFloat32 fSample = LoadCurrentLuma(samplePos) * Exposure();
    fSample = ffxPow(fSample, fShadingChangeSamplePow);
    return ffxMax(fSample, FSR3UPSCALER_EPSILON);
}

FfxFloat32 LoadPreviousLumaSample(FfxInt32x2 basePos, FfxInt32x2 offset) {
    const FfxInt32x2 samplePos = ClampLoad(basePos, offset, PreviousFrameRenderSize());
    FfxFloat32 fSample = LoadPreviousLuma(samplePos) * DeltaPreExposure() * Exposure();
    fSample = ffxPow(fSample, fShadingChangeSamplePow);
    return ffxMax(fSample, FSR3UPSCALER_EPSILON);
}

FfxFloat32 ComputeMinimumDifference(SampleSet fSet0, SampleSet fSet1) {
    FfxFloat32 fMinDiff = FSR3UPSCALER_FP16_MAX - 1;
    FfxInt32 a = 0;
    FfxInt32 b = 0;

    SortSet(fSet0);
    SortSet(fSet1);

    const FfxFloat32 fMax = ffxMin(fSet0.s4, fSet1.s4);

    if (fMax > FSR3UPSCALER_FP32_MIN) {
        FFX_UNROLL
        for (FfxInt32 i = 0; i < 5 && fMinDiff < FSR3UPSCALER_FP16_MAX; i++) {
            const FfxFloat32 fSample0 = GetSample(fSet0, a);
            const FfxFloat32 fSample1 = GetSample(fSet1, b);
            FfxFloat32 fDiff = fSample0 - fSample1;

            if (abs(fDiff) > FSR3UPSCALER_FP16_MIN) {
                fDiff = sign(fDiff) * (1.0f - MinDividedByMax(fSample0, fSample1));
                fMinDiff = abs(fDiff) < abs(fMinDiff) ? fDiff : fMinDiff;
                if (fSample0 < fSample1) {
                    a++;
                    if (a < 5 && GetSample(fSet0, a) >= fSample1) {
                        b++;
                    }
                } else {
                    b++;
                }
            } else {
                fMinDiff = FSR3UPSCALER_FP16_MAX;
            }
        }
    }

    return fMinDiff * FfxFloat32(fMinDiff < FSR3UPSCALER_FP16_MAX - 1);
}

SampleSet GetCurrentLumaBilinearSamples(FfxFloat32x2 fUv) {
    const FfxFloat32x2 fUvJittered = fUv + Jitter() / RenderSize();
    const FfxInt32x2 iBasePos = FfxInt32x2(floor(fUvJittered * RenderSize()));

    SampleSet fSet;
    fSet.s0 = LoadLumaSample(iBasePos, FfxInt32x2(0, 0), RenderSize());
    fSet.s1 = LoadLumaSample(iBasePos, FfxInt32x2(-1, 0), RenderSize());
    fSet.s2 = LoadLumaSample(iBasePos, FfxInt32x2(1, 0), RenderSize());
    fSet.s3 = LoadLumaSample(iBasePos, FfxInt32x2(0, -1), RenderSize());
    fSet.s4 = LoadLumaSample(iBasePos, FfxInt32x2(0, 1), RenderSize());
    return fSet;
}

struct PreviousLumaBilinearSamplesData {
    SampleSet fSet;
    FfxBoolean bIsExistingSample;
};

PreviousLumaBilinearSamplesData GetPreviousLumaBilinearSamples(FfxFloat32x2 fUv, FfxFloat32x2 fMotionVector) {
    PreviousLumaBilinearSamplesData data;
    const FfxFloat32x2 fUvJittered = fUv + PreviousFrameJitter() / PreviousFrameRenderSize();
    const FfxFloat32x2 fReprojectedUv = fUvJittered + fMotionVector;

    data.bIsExistingSample = IsUvInside(fReprojectedUv);

    if (data.bIsExistingSample) {
        const FfxInt32x2 iBasePos = FfxInt32x2(floor(fReprojectedUv * PreviousFrameRenderSize()));
        data.fSet.s0 = LoadPreviousLumaSample(iBasePos, FfxInt32x2(0, 0));
        data.fSet.s1 = LoadPreviousLumaSample(iBasePos, FfxInt32x2(-1, 0));
        data.fSet.s2 = LoadPreviousLumaSample(iBasePos, FfxInt32x2(1, 0));
        data.fSet.s3 = LoadPreviousLumaSample(iBasePos, FfxInt32x2(0, -1));
        data.fSet.s4 = LoadPreviousLumaSample(iBasePos, FfxInt32x2(0, 1));
    }

    return data;
}

FfxFloat32 ComputeDiff(FfxFloat32x2 fUv, FfxFloat32x2 fMotionVector) {
    FfxFloat32 fMinDiff = 0.0f;
    const SampleSet fCurrentSamples = GetCurrentLumaBilinearSamples(fUv);
    const PreviousLumaBilinearSamplesData previousData = GetPreviousLumaBilinearSamples(fUv, fMotionVector);

    if (previousData.bIsExistingSample) {
        fMinDiff = ComputeMinimumDifference(fCurrentSamples, previousData.fSet);
    }

    return fMinDiff;
}

FfxFloat32x4 SpdLoadSourceImage(FfxInt32x2 iPxPos, FfxUInt32 slice) {
    const FfxInt32x2 iPxSamplePos = ClampLoad(iPxPos, FfxInt32x2(0, 0), FfxInt32x2(RenderSize()));
    const FfxFloat32x2 fDilatedMotionVector = LoadDilatedMotionVector(iPxSamplePos);
    const FfxFloat32x2 fUv = (iPxSamplePos + 0.5f) / RenderSize();
    const FfxFloat32 fScaledAndSignedLumaDiff = ComputeDiff(fUv, fDilatedMotionVector);

    return FfxFloat32x4(
        fScaledAndSignedLumaDiff,
        fScaledAndSignedLumaDiff != 0.0f ? sign(fScaledAndSignedLumaDiff) : 0.0f,
        1.0f,
        0.0f
    );
}

FfxFloat32x4 SpdLoad(FfxInt32x2 tex, FfxUInt32 slice) {
    return FfxFloat32x4(RWLoadPyramid(tex, 5), 0, 0);
}

FfxFloat32x4 SpdReduce4(FfxFloat32x4 v0, FfxFloat32x4 v1, FfxFloat32x4 v2, FfxFloat32x4 v3) {
    return (v0 + v1 + v2 + v3) * 0.25f;
}

void SpdStore(FfxInt32x2 pix, FfxFloat32x4 outValue, FfxUInt32 index, FfxUInt32 slice) {
    if (index >= iShadingChangeMipStart) {
        StorePyramid(pix, outValue.xy, index);
    }
}

FfxFloat32x4 SpdLoadIntermediate(FfxUInt32 x, FfxUInt32 y) {
    return FfxFloat32x4(
        spdIntermediateR[x][y],
        spdIntermediateG[x][y],
        spdIntermediateB[x][y],
        spdIntermediateA[x][y]
    );
}

void SpdStoreIntermediate(FfxUInt32 x, FfxUInt32 y, FfxFloat32x4 value) {
    spdIntermediateR[x][y] = value.x;
    spdIntermediateG[x][y] = value.y;
    spdIntermediateB[x][y] = value.z;
    spdIntermediateA[x][y] = value.w;
}
#endif

#if FFX_HALF
FFX_GROUPSHARED FfxFloat16x2 spdIntermediateRG[16][16];
FFX_GROUPSHARED FfxFloat16x2 spdIntermediateBA[16][16];

FfxFloat16x4 SpdLoadSourceImageH(FfxInt32x2 tex, FfxUInt32 slice) {
    return FfxFloat16x4(0, 0, 0, 0);
}

FfxFloat16x4 SpdLoadH(FfxInt32x2 p, FfxUInt32 slice) {
    return FfxFloat16x4(0, 0, 0, 0);
}

void SpdStoreH(FfxInt32x2 p, FfxFloat16x4 value, FfxUInt32 mip, FfxUInt32 slice) {
}

FfxFloat16x4 SpdLoadIntermediateH(FfxUInt32 x, FfxUInt32 y) {
    return FfxFloat16x4(
        spdIntermediateRG[x][y].x,
        spdIntermediateRG[x][y].y,
        spdIntermediateBA[x][y].x,
        spdIntermediateBA[x][y].y
    );
}

void SpdStoreIntermediateH(FfxUInt32 x, FfxUInt32 y, FfxFloat16x4 value) {
    spdIntermediateRG[x][y] = value.xy;
    spdIntermediateBA[x][y] = value.zw;
}

FfxFloat16x4 SpdReduce4H(FfxFloat16x4 v0, FfxFloat16x4 v1, FfxFloat16x4 v2, FfxFloat16x4 v3) {
    return (v0 + v1 + v2 + v3) * FfxFloat16(0.25);
}
#endif

#include "../spd/ffx_spd.glsl"

void ComputeShadingChangePyramid(FfxUInt32x3 WorkGroupId, FfxUInt32 LocalThreadIndex) {
#if FFX_HALF
    SpdDownsampleH(
        FfxUInt32x2(WorkGroupId.xy),
        FfxUInt32(LocalThreadIndex),
        FfxUInt32(MipCount()),
        FfxUInt32(NumWorkGroups()),
        FfxUInt32(WorkGroupId.z),
        FfxUInt32x2(WorkGroupOffset())
    );
#else
    SpdDownsample(
        FfxUInt32x2(WorkGroupId.xy),
        FfxUInt32(LocalThreadIndex),
        FfxUInt32(MipCount()),
        FfxUInt32(NumWorkGroups()),
        FfxUInt32(WorkGroupId.z),
        FfxUInt32x2(WorkGroupOffset())
    );
#endif
}
