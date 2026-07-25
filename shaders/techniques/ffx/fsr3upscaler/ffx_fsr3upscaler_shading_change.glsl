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

FFX_STATIC const FfxInt32 s_MipLevelsToUse = 3;

struct ShadingChangeLumaInfo
{
    FfxFloat32 fMip0;
    FfxFloat32 fMip1;
    FfxFloat32 fMip2;
};

ShadingChangeLumaInfo ComputeShadingChangeLuma(FfxFloat32x2 fUv)
{
    ShadingChangeLumaInfo info;

    const FfxFloat32x2 fMipUv = ClampUv(fUv, ShadingChangeRenderSize(), GetSPDMipDimensions(0));
    const FfxFloat32x2 fSample0 = SampleSPDMipLevel(fMipUv, 0);
    const FfxFloat32x2 fSample1 = SampleSPDMipLevel(fMipUv, 1);
    const FfxFloat32x2 fSample2 = SampleSPDMipLevel(fMipUv, 2);
    info.fMip0 = abs(fSample0.x * fSample0.y);
    info.fMip1 = abs(fSample1.x * fSample1.y);
    info.fMip2 = abs(fSample2.x * fSample2.y);

    return info;
}

void ShadingChange(FfxInt32x2 iPxPos)
{
    if (IsOnScreen(FfxInt32x2(iPxPos), ShadingChangeRenderSize())) {

        const FfxFloat32x2 fUv = (iPxPos + 0.5f) / ShadingChangeRenderSize();
        const FfxFloat32x2 fUvJittered = fUv + Jitter() / RenderSize();

        const ShadingChangeLumaInfo info = ComputeShadingChangeLuma(fUvJittered);

        const FfxFloat32 fScale = 1.0f + iShadingChangeMipStart / s_MipLevelsToUse;
        const FfxFloat32 fShadingChange = ffxMax(info.fMip0, ffxMax(info.fMip1, info.fMip2)) * fScale;
        
        StoreShadingChange(iPxPos, ffxSaturate(fShadingChange));
    }
}
