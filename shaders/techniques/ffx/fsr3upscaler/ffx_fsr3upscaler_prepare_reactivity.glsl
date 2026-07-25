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

void AccumulateDisocclusionSample(
    FfxInt32x2 iBasePos,
    FfxInt32x2 iOffset,
    FfxFloat32 fWeight,
    FfxFloat32 fCurrentDepthViewSpace,
    FfxFloat32 fHalfViewportWidth,
    FFX_PARAMETER_INOUT FfxFloat32 fDisocclusion,
    FFX_PARAMETER_INOUT FfxFloat32 fWeightSum,
    FFX_PARAMETER_INOUT FfxBoolean bPotentialDisocclusion)
{
    if (!bPotentialDisocclusion) {
        return;
    }

    const FfxInt32x2 iSamplePos = ClampLoad(iBasePos, iOffset, FfxInt32x2(RenderSize()));
    if (IsOnScreen(iSamplePos, RenderSize()) && fWeight > fReconstructedDepthBilinearWeightThreshold) {
        const FfxFloat32 fPrevNearestDepthViewSpace = GetViewSpaceDepth(LoadReconstructedPrevDepth(iSamplePos));
        const FfxFloat32 fDepthDifference = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;

        bPotentialDisocclusion = fDepthDifference > FSR3UPSCALER_FP32_MIN;
        if (bPotentialDisocclusion) {
            const FfxFloat32 fDepthThreshold = ffxMax(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);
            const FfxFloat32 Ksep = 1.37e-05f;
            const FfxFloat32 fRequiredDepthSeparation = Ksep * fHalfViewportWidth * fDepthThreshold;

            fDisocclusion += ffxSaturate(FfxFloat32(fRequiredDepthSeparation / fDepthDifference)) * fWeight;
            fWeightSum += fWeight;
        }
    }
}

FfxFloat32 ComputeDisocclusions(FfxFloat32x2 fUv, FfxFloat32x2 fMotionVector, FfxFloat32 fCurrentDepthViewSpace)
{
    const FfxFloat32 fNearestDepthInMeters = ffxMin(fCurrentDepthViewSpace * ViewSpaceToMetersFactor(), FSR3UPSCALER_FP16_MAX);
    const FfxFloat32 fReconstructedDeptMvThreshold = ReconstructedDepthMvPxThreshold(fNearestDepthInMeters);

    fMotionVector *= FfxFloat32(Get4KVelocity(fMotionVector) > fReconstructedDeptMvThreshold);

    const FfxFloat32x2 fReprojectedUv       = fUv + fMotionVector;
    const BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fReprojectedUv, RenderSize());

    FfxFloat32 fDisocclusion            = 0.0f;
    FfxFloat32 fWeightSum               = 0.0f;
    FfxBoolean bPotentialDisocclusion   = true;
    const FfxFloat32 fHalfViewportWidth = length(FfxFloat32x2(RenderSize()) * 0.5f);

    AccumulateDisocclusionSample(bilinearInfo.iBasePos, bilinearInfo.iOffset00, bilinearInfo.fWeight00, fCurrentDepthViewSpace, fHalfViewportWidth, fDisocclusion, fWeightSum, bPotentialDisocclusion);
    AccumulateDisocclusionSample(bilinearInfo.iBasePos, bilinearInfo.iOffset10, bilinearInfo.fWeight10, fCurrentDepthViewSpace, fHalfViewportWidth, fDisocclusion, fWeightSum, bPotentialDisocclusion);
    AccumulateDisocclusionSample(bilinearInfo.iBasePos, bilinearInfo.iOffset01, bilinearInfo.fWeight01, fCurrentDepthViewSpace, fHalfViewportWidth, fDisocclusion, fWeightSum, bPotentialDisocclusion);
    AccumulateDisocclusionSample(bilinearInfo.iBasePos, bilinearInfo.iOffset11, bilinearInfo.fWeight11, fCurrentDepthViewSpace, fHalfViewportWidth, fDisocclusion, fWeightSum, bPotentialDisocclusion);

    fDisocclusion = (bPotentialDisocclusion && fWeightSum > 0) ? ffxSaturate(1.0f - fDisocclusion / fWeightSum) : 0.0f;

    return fDisocclusion;
}

FfxFloat32 ComputeMotionDivergence(FfxFloat32x2 fUv, FfxFloat32x2 fMotionVector, FfxFloat32 fCurrentDepthSample)
{
    const FfxInt32x2 iPxReprojectedPos = FfxInt32x2((fUv + fMotionVector) * RenderSize());
    const FfxFloat32 fReprojectedDepth = LoadDilatedDepth(iPxReprojectedPos);
    const FfxFloat32x2 fReprojectedMotionVector = LoadDilatedMotionVector(iPxReprojectedPos);

    const FfxFloat32 fReprojectedVelocity = Get4KVelocity(fReprojectedMotionVector);
    const FfxFloat32 f4KVelocity = Get4KVelocity(fMotionVector);

    const FfxFloat32 fMaxLen = max(length(fMotionVector), length(fReprojectedMotionVector));

    const FfxFloat32 fNucleusDepthInMeters = GetViewSpaceDepthInMeters(fReprojectedDepth);
    const FfxFloat32 fCurrentDepthInMeters = GetViewSpaceDepthInMeters(fCurrentDepthSample);

    const FfxFloat32 fDistanceFactor = MinDividedByMax(fNucleusDepthInMeters, fCurrentDepthInMeters);
    const FfxFloat32 fVelocityFactor = ffxSaturate(f4KVelocity / 10.0f);
    const FfxFloat32 fMotionVectorFieldConfidence = (1.0f - ffxSaturate(fReprojectedVelocity / f4KVelocity)) * fDistanceFactor * fVelocityFactor;

    return fMotionVectorFieldConfidence;
}

FfxFloat32 DilateReactiveMasks(FfxInt32x2 iPxPos)
{
    FfxFloat32 fDilatedReactiveMasks = 0.0f;
    
    FFX_UNROLL
    for (FfxInt32 y = -1; y <=1; y++)
    {
        FFX_UNROLL
        for (FfxInt32 x = -1; x <= 1; x++)
        {
            const FfxInt32x2 sampleCoord = ClampLoad(iPxPos, FfxInt32x2(x, y), FfxInt32x2(RenderSize()));
            fDilatedReactiveMasks = ffxMax(fDilatedReactiveMasks, LoadReactiveMask(sampleCoord));
        }
    }

    return fDilatedReactiveMasks;
}

FfxFloat32 DilateTransparencyAndCompositionMasks(FfxInt32x2 iPxPos, FfxFloat32x2 fUv)
{
    const FfxFloat32x2 fUvTransparencyAndCompositionMask = ClampUv(fUv, RenderSize(), GetTransparencyAndCompositionMaskResourceDimensions());
    return SampleTransparencyAndCompositionMask(fUvTransparencyAndCompositionMask);
}

void ClassifyThinFeatureSample(
    FfxFloat32 fSample,
    FfxFloat32 fNucleus,
    FfxFloat32 fLumaRange,
    FfxFloat32 fThreshold,
    FfxUInt32 uBit,
    FFX_PARAMETER_INOUT FfxUInt32 uPatternMask,
    FFX_PARAMETER_INOUT FfxFloat32 fDissimilarLumaMin,
    FFX_PARAMETER_INOUT FfxFloat32 fDissimilarLumaMax)
{
    const FfxFloat32 fDifference = abs(fSample - fNucleus) / fLumaRange;
    if (fDifference < fThreshold) {
        uPatternMask |= uBit;
    } else {
        fDissimilarLumaMin = ffxMin(fDissimilarLumaMin, fSample);
        fDissimilarLumaMax = ffxMax(fDissimilarLumaMax, fSample);
    }
}

FfxFloat32 ComputeThinFeatureConfidence(FfxInt32x2 iPxPos)
{
    /*
     1 2 3
     4 0 5
     6 7 8
    */

    const FfxFloat32 fSample0 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(+0, +0), FfxInt32x2(RenderSize()))) * Exposure();
    const FfxFloat32 fSample1 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(-1, -1), FfxInt32x2(RenderSize()))) * Exposure();
    const FfxFloat32 fSample2 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(+0, -1), FfxInt32x2(RenderSize()))) * Exposure();
    const FfxFloat32 fSample3 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(+1, -1), FfxInt32x2(RenderSize()))) * Exposure();
    const FfxFloat32 fSample4 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(-1, +0), FfxInt32x2(RenderSize()))) * Exposure();
    const FfxFloat32 fSample5 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(+1, +0), FfxInt32x2(RenderSize()))) * Exposure();
    const FfxFloat32 fSample6 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(-1, +1), FfxInt32x2(RenderSize()))) * Exposure();
    const FfxFloat32 fSample7 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(+0, +1), FfxInt32x2(RenderSize()))) * Exposure();
    const FfxFloat32 fSample8 = LoadCurrentLuma(ClampLoad(iPxPos, FfxInt32x2(+1, +1), FfxInt32x2(RenderSize()))) * Exposure();

    FfxFloat32 fLumaMin = fSample0;
    FfxFloat32 fLumaMax = fSample0;
    fLumaMin = ffxMin(fLumaMin, fSample1);
    fLumaMin = ffxMin(fLumaMin, fSample2);
    fLumaMin = ffxMin(fLumaMin, fSample3);
    fLumaMin = ffxMin(fLumaMin, fSample4);
    fLumaMin = ffxMin(fLumaMin, fSample5);
    fLumaMin = ffxMin(fLumaMin, fSample6);
    fLumaMin = ffxMin(fLumaMin, fSample7);
    fLumaMin = ffxMin(fLumaMin, fSample8);
    fLumaMax = ffxMax(fLumaMax, fSample1);
    fLumaMax = ffxMax(fLumaMax, fSample2);
    fLumaMax = ffxMax(fLumaMax, fSample3);
    fLumaMax = ffxMax(fLumaMax, fSample4);
    fLumaMax = ffxMax(fLumaMax, fSample5);
    fLumaMax = ffxMax(fLumaMax, fSample6);
    fLumaMax = ffxMax(fLumaMax, fSample7);
    fLumaMax = ffxMax(fLumaMax, fSample8);

    const FfxFloat32 fThreshold = 0.9f;
    FfxFloat32       fDissimilarLumaMin = FSR3UPSCALER_FP32_MAX;
    FfxFloat32       fDissimilarLumaMax = 0.0f;

    FfxUInt32 uPatternMask = 1U; // Flag nucleus as similar.
    const FfxFloat32 fLumaRange = fLumaMax - fLumaMin;
    ClassifyThinFeatureSample(fSample1, fSample0, fLumaRange, fThreshold, 1U << 1, uPatternMask, fDissimilarLumaMin, fDissimilarLumaMax);
    ClassifyThinFeatureSample(fSample2, fSample0, fLumaRange, fThreshold, 1U << 2, uPatternMask, fDissimilarLumaMin, fDissimilarLumaMax);
    ClassifyThinFeatureSample(fSample3, fSample0, fLumaRange, fThreshold, 1U << 3, uPatternMask, fDissimilarLumaMin, fDissimilarLumaMax);
    ClassifyThinFeatureSample(fSample4, fSample0, fLumaRange, fThreshold, 1U << 4, uPatternMask, fDissimilarLumaMin, fDissimilarLumaMax);
    ClassifyThinFeatureSample(fSample5, fSample0, fLumaRange, fThreshold, 1U << 5, uPatternMask, fDissimilarLumaMin, fDissimilarLumaMax);
    ClassifyThinFeatureSample(fSample6, fSample0, fLumaRange, fThreshold, 1U << 6, uPatternMask, fDissimilarLumaMin, fDissimilarLumaMax);
    ClassifyThinFeatureSample(fSample7, fSample0, fLumaRange, fThreshold, 1U << 7, uPatternMask, fDissimilarLumaMin, fDissimilarLumaMax);
    ClassifyThinFeatureSample(fSample8, fSample0, fLumaRange, fThreshold, 1U << 8, uPatternMask, fDissimilarLumaMin, fDissimilarLumaMax);

    const FfxBoolean bIsRidge = fSample0 > fDissimilarLumaMax || fSample0 < fDissimilarLumaMin;

    if (FFX_FALSE == bIsRidge)
    {
        return 0.0f;
    }

    const FfxUInt32 uUpperLeftMask = (1U << 1) | (1U << 2) | (1U << 4) | 1U;
    const FfxUInt32 uUpperRightMask = (1U << 2) | (1U << 3) | (1U << 5) | 1U;
    const FfxUInt32 uLowerLeftMask = (1U << 4) | (1U << 6) | (1U << 7) | 1U;
    const FfxUInt32 uLowerRightMask = (1U << 5) | (1U << 7) | (1U << 8) | 1U;
    if ((uPatternMask & uUpperLeftMask) == uUpperLeftMask ||
        (uPatternMask & uUpperRightMask) == uUpperRightMask ||
        (uPatternMask & uLowerLeftMask) == uLowerLeftMask ||
        (uPatternMask & uLowerRightMask) == uLowerRightMask) {
        return 0.0f;
    }

    return 1.0f - fLumaMin / fLumaMax;
}

FfxFloat32 UpdateAccumulation(FfxInt32x2 iPxPos, FfxFloat32x2 fUv, FfxFloat32x2 fMotionVector, FfxFloat32 fDisocclusion, FfxFloat32 fShadingChange)
{
    const FfxFloat32x2 fReprojectedUv = fUv + fMotionVector;
    FfxFloat32 fAccumulation = 0.0f;

    if (IsUvInside(fReprojectedUv)) {
        const FfxFloat32x2 fReprojectedUv_HW = ClampUv(fReprojectedUv, PreviousFrameRenderSize(), MaxRenderSize());
        fAccumulation                        = ffxSaturate(SampleAccumulation(fReprojectedUv_HW));
    }
    const FfxFloat32 fAccumulationAddedPerFrame= AccumulationAddedPerFrame(); //default is 0.333

    // Assume at frame N+0 fShadingChange is 1.0, and all subsequent frames fShadingChange is 0.0 and fDisocclusion is 0.0. Then,
    // frame N+0 fAccumulation will be 0.000
    // frame N+2 fAccumulation will be 0.000 + 0.333 * 1 == 0.333
    // frame N+3 fAccumulation will be 0.000 + 0.333 * 2 == 0.666
    // frame N+4 fAccumulation will be 0.000 + 0.333 * 3 == 0.999
    fAccumulation = ffxLerp(fAccumulation, 0.0f, fShadingChange);

    const FfxFloat32 fMinDisocclusionAccumulation = MinDisocclusionAccumulation(); //default is -0.333
    // Assume at frame N+0 fDisocclusion is 1.0, and all subsequent frames fShadingChange is 0.0 and fDisocclusion is 0.0. Then,
    // frame N+0 fAccumulation will be -0.333f (but normalized to store in unorm)
    // frame N+1 fAccumulation will be -0.333f + 0.333 * 1 == 0.000
    // frame N+2 fAccumulation will be -0.333f + 0.333 * 2 == 0.333
    // frame N+3 fAccumulation will be -0.333f + 0.333 * 3 == 0.666
    // frame N+4 fAccumulation will be -0.333f + 0.333 * 4 == 0.999
    fAccumulation = ffxLerp(fAccumulation, ffxMin(fMinDisocclusionAccumulation, fAccumulation), fDisocclusion);
    
    fAccumulation *= FfxFloat32(round(fAccumulation * 100.0f) > 1.0f);

    // Update for next frame, normalize to store in unorm
    const FfxFloat32 fAccumulatedFramesToStore = ffxSaturate(fAccumulation + fAccumulationAddedPerFrame);
    StoreAccumulation(iPxPos, fAccumulatedFramesToStore);

    return fAccumulation;
}

FfxFloat32 ComputeShadingChange(FfxFloat32x2 fUv)
{
    // NOTE: Here we re-apply jitter, will be reverted again when sampled in accumulation pass
    const FfxFloat32x2 fShadingChangeUv = ClampUv(fUv - Jitter() / RenderSize(), ShadingChangeRenderSize(), ShadingChangeMaxRenderSize());
    const FfxFloat32 fShadingChange = ffxSaturate(SampleShadingChange(fShadingChangeUv));

    return fShadingChange;
}

void PrepareReactivity(FfxInt32x2 iPxPos)
{
    const FfxFloat32x2 fUv = (iPxPos + 0.5f) / RenderSize();
    const FfxFloat32x2 fMotionVector = LoadDilatedMotionVector(iPxPos);

    // Discard small mvs
    const FfxFloat32 f4KVelocity = Get4KVelocity(fMotionVector);

    const FfxFloat32x2 fDilatedUv = fUv + fMotionVector;
    const FfxFloat32 fDilatedDepth = LoadDilatedDepth(iPxPos);
    const FfxFloat32 fDepthInMeters = GetViewSpaceDepthInMeters(fDilatedDepth);

    const FfxFloat32 fDisocclusion = ComputeDisocclusions(fUv, fMotionVector, GetViewSpaceDepth(fDilatedDepth));
    const FfxFloat32 fShadingChange = ffxMax(DilateReactiveMasks(iPxPos), ComputeShadingChange(fUv));

    const FfxFloat32 fMotionDivergence = ComputeMotionDivergence(fUv, fMotionVector, fDilatedDepth);
    const FfxFloat32 fDilatedTransparencyAndComposition = DilateTransparencyAndCompositionMasks(iPxPos, fUv);
    const FfxFloat32 fFinalReactiveness = ffxMax(fMotionDivergence, fDilatedTransparencyAndComposition);

    const FfxFloat32 fAccumulation = UpdateAccumulation(iPxPos, fUv, fMotionVector, fDisocclusion, fShadingChange);

    const FfxFloat32x4 fOutput = FfxFloat32x4(fFinalReactiveness, fDisocclusion, fShadingChange, fAccumulation);

    StoreDilatedReactiveMasks(iPxPos, fOutput);

    const FfxFloat32 fLockStrength = ComputeThinFeatureConfidence(iPxPos);
    if (fLockStrength > (1.0f / 100.0f))
    {
        StoreNewLocks(ComputeHrPosFromLrPos(FfxInt32x2(iPxPos)), fLockStrength);
    }
}
