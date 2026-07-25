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

struct LumaInstabilityFactorData
{
    FfxFloat32x4 fLumaHistory;
    FfxFloat32 fLumaInstabilityFactor;
};

LumaInstabilityFactorData ComputeLumaInstabilityFactor(LumaInstabilityFactorData data, FfxFloat32 fCurrentFrameLuma)
{
    FfxFloat32 fLumaInstability     = 0.0f;
    const FfxFloat32 fDiffs0        = fCurrentFrameLuma - data.fLumaHistory.x;
    const FfxFloat32 fSimilarity0   = MinDividedByMax(fCurrentFrameLuma, data.fLumaHistory.x, 1.0f);

    FfxFloat32 fMaxSimilarity = fSimilarity0;

    if (fSimilarity0 < 1.0f) {
        const FfxFloat32 fDiffs1 = fCurrentFrameLuma - data.fLumaHistory.y;
        const FfxFloat32 fDiffs2 = fCurrentFrameLuma - data.fLumaHistory.z;
        const FfxFloat32 fDiffs3 = fCurrentFrameLuma - data.fLumaHistory.w;

        if (sign(fDiffs0) == sign(fDiffs1)) {
            fMaxSimilarity = ffxMax(fMaxSimilarity, MinDividedByMax(fCurrentFrameLuma, data.fLumaHistory.y));
        }
        if (sign(fDiffs0) == sign(fDiffs2)) {
            fMaxSimilarity = ffxMax(fMaxSimilarity, MinDividedByMax(fCurrentFrameLuma, data.fLumaHistory.z));
        }
        if (sign(fDiffs0) == sign(fDiffs3)) {
            fMaxSimilarity = ffxMax(fMaxSimilarity, MinDividedByMax(fCurrentFrameLuma, data.fLumaHistory.w));
        }

        fLumaInstability = FfxFloat32(fMaxSimilarity > fSimilarity0);
    }

    data.fLumaHistory.w = data.fLumaHistory.z;
    data.fLumaHistory.z = data.fLumaHistory.y;
    data.fLumaHistory.y = data.fLumaHistory.x;
    data.fLumaHistory.x = fCurrentFrameLuma;

    data.fLumaHistory /= Exposure();

    data.fLumaInstabilityFactor = fLumaInstability * FfxFloat32(data.fLumaHistory.w != 0.0f);

    return data;
}

void LumaInstability(FfxInt32x2 iPxPos)
{
    LumaInstabilityFactorData data;
    data.fLumaInstabilityFactor = 0.0f;
    data.fLumaHistory = FfxFloat32x4(0.0f, 0.0f, 0.0f, 0.0f);

    const FfxFloat32x2 fDilatedMotionVector = LoadDilatedMotionVector(iPxPos);
    const FfxFloat32x2 fUv = (iPxPos + 0.5f) / RenderSize();
    const FfxFloat32x2 fUvCurrFrameJittered = fUv + Jitter() / RenderSize();
    const FfxFloat32x2 fUvPrevFrameJittered = fUv + PreviousFrameJitter() / PreviousFrameRenderSize();
    const FfxFloat32x2 fReprojectedUv = fUvPrevFrameJittered + fDilatedMotionVector;

    if (IsUvInside(fReprojectedUv))
    {
        const FfxFloat32x2 fUvReactive_HW = ClampUv(fUvCurrFrameJittered, RenderSize(), MaxRenderSize());

        const FfxFloat32x4 fDilatedReactiveMasks = SampleDilatedReactiveMasks(fUvReactive_HW);
        const FfxFloat32 fReactiveMask = ffxSaturate(fDilatedReactiveMasks.x);
        const FfxFloat32 fDisocclusion = ffxSaturate(fDilatedReactiveMasks.y);
        const FfxFloat32 fShadingChange = ffxSaturate(fDilatedReactiveMasks.z);
        const FfxFloat32 fAccumulation = ffxSaturate(fDilatedReactiveMasks.w);

        const FfxBoolean bAccumulationFactor = fAccumulation > 0.9f;

        const FfxBoolean bComputeInstability = bAccumulationFactor;

        if (bComputeInstability) {

            const FfxFloat32x2 fUv_HW = ClampUv(fUvCurrFrameJittered, RenderSize(), MaxRenderSize());
            const FfxFloat32 fCurrentFrameLuma = SampleCurrentLuma(fUv_HW) * Exposure();

            const FfxFloat32x2 fReprojectedUv_HW = ClampUv(fReprojectedUv, PreviousFrameRenderSize(), MaxRenderSize());
            data.fLumaHistory                    = SampleLumaHistory(fReprojectedUv_HW) * DeltaPreExposure() * Exposure();

            data = ComputeLumaInstabilityFactor(data, fCurrentFrameLuma);

            const FfxFloat32 fVelocityWeight = 1.0f - ffxSaturate(Get4KVelocity(fDilatedMotionVector) / 20.0f);
            data.fLumaInstabilityFactor *= fVelocityWeight * (1.0f - fDisocclusion) * (1.0f - fReactiveMask) * (1.0f - fShadingChange);
        }
    }

    StoreLumaHistory(iPxPos, data.fLumaHistory);
    StoreLumaInstability(iPxPos, data.fLumaInstabilityFactor);
}
