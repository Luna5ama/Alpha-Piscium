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

void StoreReconstructedDepthSample(
    FfxInt32x2 iBasePos,
    FfxInt32x2 iOffset,
    FfxFloat32 fWeight,
    FfxFloat32 fDepth)
{
    if (fWeight > fReconstructedDepthBilinearWeightThreshold) {
        const FfxInt32x2 iStorePos = iBasePos + iOffset;
        if (IsOnScreen(iStorePos, RenderSize())) {
            StoreReconstructedDepth(iStorePos, fDepth);
        }
    }
}

void ReconstructPrevDepth(FfxInt32x2 iPxPos, FfxFloat32 fDepth, FfxFloat32x2 fMotionVector)
{
    const FfxFloat32 fNearestDepthInMeters = ffxMin(GetViewSpaceDepthInMeters(fDepth), FSR3UPSCALER_FP16_MAX);
    const FfxFloat32 fReconstructedDeptMvThreshold = ReconstructedDepthMvPxThreshold(fNearestDepthInMeters);

    // Discard small mvs
    fMotionVector *= FfxFloat32(Get4KVelocity(fMotionVector) > fReconstructedDeptMvThreshold);

    const FfxFloat32x2 fUv = (iPxPos + FfxFloat32(0.5)) / RenderSize();
    const FfxFloat32x2 fReprojectedUv = fUv + fMotionVector;
    const BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fReprojectedUv, RenderSize());

    // Project current depth into previous frame locations.
    // Push to all pixels having some contribution if reprojection is using bilinear logic.
    StoreReconstructedDepthSample(bilinearInfo.iBasePos, bilinearInfo.iOffset00, bilinearInfo.fWeight00, fDepth);
    StoreReconstructedDepthSample(bilinearInfo.iBasePos, bilinearInfo.iOffset10, bilinearInfo.fWeight10, fDepth);
    StoreReconstructedDepthSample(bilinearInfo.iBasePos, bilinearInfo.iOffset01, bilinearInfo.fWeight01, fDepth);
    StoreReconstructedDepthSample(bilinearInfo.iBasePos, bilinearInfo.iOffset11, bilinearInfo.fWeight11, fDepth);
}

struct DepthExtents
{
    FfxFloat32 fNearest;
    FfxInt32x2 fNearestCoord;
    FfxFloat32 fFarthest;
};

void UpdateDepthExtents(FfxInt32x2 iPos, FfxFloat32 fDepth, FFX_PARAMETER_INOUT DepthExtents extents)
{
    if (IsOnScreen(iPos, RenderSize())) {
#if FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH
        extents.fFarthest = ffxMin(extents.fFarthest, fDepth);
        if (fDepth > extents.fNearest)
#else
        extents.fFarthest = ffxMax(extents.fFarthest, fDepth);
        if (fDepth < extents.fNearest)
#endif
        {
            extents.fNearestCoord = iPos;
            extents.fNearest = fDepth;
        }
    }
}

DepthExtents FindDepthExtents(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    DepthExtents extents;
    const FfxInt32x2 iPos0 = iPxPos;
    const FfxInt32x2 iPos1 = iPxPos + FfxInt32x2(+1, +0);
    const FfxInt32x2 iPos2 = iPxPos + FfxInt32x2(+0, +1);
    const FfxInt32x2 iPos3 = iPxPos + FfxInt32x2(+0, -1);
    const FfxInt32x2 iPos4 = iPxPos + FfxInt32x2(-1, +0);
    const FfxInt32x2 iPos5 = iPxPos + FfxInt32x2(-1, +1);
    const FfxInt32x2 iPos6 = iPxPos + FfxInt32x2(+1, +1);
    const FfxInt32x2 iPos7 = iPxPos + FfxInt32x2(-1, -1);
    const FfxInt32x2 iPos8 = iPxPos + FfxInt32x2(+1, -1);

    // Pull out the depth loads to allow SC to batch them.
    const FfxFloat32 fDepth0 = LoadInputDepth(iPos0);
    const FfxFloat32 fDepth1 = LoadInputDepth(iPos1);
    const FfxFloat32 fDepth2 = LoadInputDepth(iPos2);
    const FfxFloat32 fDepth3 = LoadInputDepth(iPos3);
    const FfxFloat32 fDepth4 = LoadInputDepth(iPos4);
    const FfxFloat32 fDepth5 = LoadInputDepth(iPos5);
    const FfxFloat32 fDepth6 = LoadInputDepth(iPos6);
    const FfxFloat32 fDepth7 = LoadInputDepth(iPos7);
    const FfxFloat32 fDepth8 = LoadInputDepth(iPos8);

    // find closest depth
    extents.fNearestCoord = iPos0;
    extents.fNearest = fDepth0;
    extents.fFarthest = fDepth0;
    UpdateDepthExtents(iPos1, fDepth1, extents);
    UpdateDepthExtents(iPos2, fDepth2, extents);
    UpdateDepthExtents(iPos3, fDepth3, extents);
    UpdateDepthExtents(iPos4, fDepth4, extents);
    UpdateDepthExtents(iPos5, fDepth5, extents);
    UpdateDepthExtents(iPos6, fDepth6, extents);
    UpdateDepthExtents(iPos7, fDepth7, extents);
    UpdateDepthExtents(iPos8, fDepth8, extents);

    return extents;
}

FfxFloat32x2 DilateMotionVector(FfxInt32x2 iPxPos, const DepthExtents depthExtents)
{
#if FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS
    const FfxInt32x2 iSamplePos       = iPxPos;
    const FfxInt32x2 iMotionVectorPos = depthExtents.fNearestCoord;
#else
    const FfxInt32x2 iSamplePos       = ComputeHrPosFromLrPos(iPxPos);
    const FfxInt32x2 iMotionVectorPos = ComputeHrPosFromLrPos(depthExtents.fNearestCoord);
#endif

    const FfxFloat32x2 fDilatedMotionVector = LoadInputMotionVector(iMotionVectorPos);

    return fDilatedMotionVector;
}

FfxFloat32 GetCurrentFrameLuma(FfxInt32x2 iPxPos)
{
    //We assume linear data. if non-linear input (sRGB, ...),
    //then we should convert to linear first and back to sRGB on output.
    const FfxFloat32x3 fRgb = ffxMax(FfxFloat32x3(0, 0, 0), LoadInputColor(iPxPos));
    const FfxFloat32 fLuma  = RGBToLuma(fRgb);

    return fLuma;
}

void PrepareInputs(FfxInt32x2 iPxPos)
{
    const DepthExtents depthExtents = FindDepthExtents(iPxPos);
    const FfxFloat32x2 fDilatedMotionVector = DilateMotionVector(iPxPos, depthExtents);

    ReconstructPrevDepth(iPxPos, depthExtents.fNearest, fDilatedMotionVector);

    StoreDilatedMotionVector(iPxPos, fDilatedMotionVector);
    StoreDilatedDepth(iPxPos, depthExtents.fNearest);

    const FfxFloat32 fFarthestDepthInMeters = ffxMin(GetViewSpaceDepthInMeters(depthExtents.fFarthest), FSR3UPSCALER_FP16_MAX);
    StoreFarthestDepth(iPxPos, fFarthestDepthInMeters);

    const FfxFloat32 fLuma = GetCurrentFrameLuma(iPxPos);
    StoreCurrentLuma(iPxPos, fLuma);
}
