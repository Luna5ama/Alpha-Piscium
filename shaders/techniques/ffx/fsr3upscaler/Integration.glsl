#ifndef INCLUDE_ffx_fsr3upscaler_Integration_glsl
#define INCLUDE_ffx_fsr3upscaler_Integration_glsl

#include "/Base.glsl"
#include "/techniques/ffx/ffx_core.glsl"

#define FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH 1
#define FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS 1
#define FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT 1
#define FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING 1
#define FSR3UPSCALER_BIND_SRV_FRAME_INFO 1

#if defined(FSR3_BIND_PREPARE_INPUTS)
layout(rgba16f) uniform image2D uimg_rgba16f;
layout(r32f) uniform image2D uimg_r32f;
layout(r32ui) uniform coherent uimage2D uimg_fsr3ReconstructedDepth;
layout(rgba16f) uniform image2D uimg_fsr3UpscaleAtlas;
#elif defined(FSR3_BIND_LUMA_PYRAMID)
layout(rgba16f) uniform coherent image2D uimg_rgba16f;
#elif defined(FSR3_BIND_SHADING_CHANGE_PYRAMID)
layout(rgba16f) uniform coherent image2D uimg_rgba16f;
#elif defined(FSR3_BIND_SHADING_CHANGE)
layout(rgba8) uniform image2D uimg_rgba8;
#elif defined(FSR3_BIND_PREPARE_REACTIVITY)
layout(rgba8) uniform image2D uimg_rgba8;
layout(rgba16f) uniform image2D uimg_fsr3UpscaleAtlas;
#elif defined(FSR3_BIND_LUMA_INSTABILITY)
layout(rgba16f) uniform image2D uimg_rgba16f;
layout(rgba16f) uniform image2D uimg_fsr3UpscaleAtlas;
#elif defined(FSR3_BIND_ACCUMULATE)
layout(rgba16f) uniform image2D uimg_rgba16f;
layout(rgba16f) uniform image2D uimg_fsr3UpscaleAtlas;
#elif defined(FSR3_BIND_RCAS)
layout(rgba16f) uniform image2D uimg_fsr3UpscaleAtlas;
#else
#error FSR3 binding set is not defined
#endif

FfxInt32x2 RenderSize() {
    return FfxInt32x2(uval_mainImageSizeI);
}

FfxInt32x2 UpscaleSize() {
    return FfxInt32x2(uval_viewImageSize);
}

FfxBoolean FSR3HistoryReset() {
    return frameCounter <= 1 || global_taaResetFactor.z < 0.5 || int(global_fsr3FrameInfo.w) != frameCounter - 1;
}

FfxInt32x2 PreviousFrameRenderSize() {
    return RenderSize();
}

FfxInt32x2 PreviousFrameUpscaleSize() {
    return UpscaleSize();
}

FfxInt32x2 MaxRenderSize() {
    return RenderSize();
}

FfxFloat32x2 DownscaleFactor() {
    return FfxFloat32x2(RenderSize()) / FfxFloat32x2(UpscaleSize());
}

FfxFloat32x2 Jitter() {
    return FfxFloat32x2(uval_taaJitter);
}

FfxFloat32x2 PreviousFrameJitter() {
    return FfxFloat32x2(uval_prevTaaJitter);
}

FfxInt32 FrameIndex() {
    return FSR3HistoryReset() ? 0 : frameCounter;
}

FfxFloat32 JitterSequenceLength() {
    FfxFloat32 upscaleRatio = FfxFloat32(UpscaleSize().x) / FfxFloat32(RenderSize().x);
    return FfxFloat32(ffxMax(FfxInt32(1), FfxInt32(8.0f * upscaleRatio * upscaleRatio)));
}

FfxFloat32 Exposure() {
    return global_fsr3FrameInfo.x;
}

FfxFloat32 DeltaPreExposure() {
    return 1.0f;
}

FfxFloat32 DeltaTime() {
    return frameTime;
}

FfxFloat32 ViewSpaceToMetersFactor() {
    return 1.0f;
}

FfxFloat32 VelocityFactor() {
    return 1.0f;
}

FfxFloat32 AccumulationAddedPerFrame() {
    return 1.0f / 3.0f;
}

FfxFloat32 MinDisocclusionAccumulation() {
    return -1.0f / 3.0f;
}

FfxFloat32 SampleLanczos2Weight(FfxFloat32 x) {
    if (x < 1.0e-5f) return 1.0f;
    if (x >= 2.0f) return 0.0f;
    const FfxFloat32 piX = 3.141592653589793f * x;
    return sin(piX) / piX * sin(0.5f * piX) / (0.5f * piX);
}

FfxFloat32x4 DeviceToViewSpaceTransformFactors() {
    return FfxFloat32x4(0.0f, near, global_camProjInverse[0][0], -global_camProjInverse[1][1]);
}

struct FSR3BilinearContext {
    FfxInt32x2 p00;
    FfxInt32x2 p10;
    FfxInt32x2 p01;
    FfxInt32x2 p11;
    FfxFloat32x2 weight;
};

FSR3BilinearContext FSR3CreateBilinearContext(FfxFloat32x2 uv, FfxInt32x2 size) {
    FfxFloat32x2 texel = uv * FfxFloat32x2(size) - 0.5f;
    FfxInt32x2 base = FfxInt32x2(floor(texel));
    FSR3BilinearContext context;
    context.p00 = clamp(base, FfxInt32x2(0), size - 1);
    context.p10 = clamp(base + FfxInt32x2(1, 0), FfxInt32x2(0), size - 1);
    context.p01 = clamp(base + FfxInt32x2(0, 1), FfxInt32x2(0), size - 1);
    context.p11 = clamp(base + FfxInt32x2(1, 1), FfxInt32x2(0), size - 1);
    context.weight = ffxFract(texel);
    return context;
}

FfxFloat32x4 FSR3FilterBilinear(
    FfxFloat32x4 s00,
    FfxFloat32x4 s10,
    FfxFloat32x4 s01,
    FfxFloat32x4 s11,
    FfxFloat32x2 weight
) {
    FfxFloat32x4 row0 = ffxLerp(s00, s10, weight.x);
    FfxFloat32x4 row1 = ffxLerp(s01, s11, weight.x);
    return ffxLerp(row0, row1, weight.y);
}

#define FSR3_FILTER_TILE(fetch, context) FSR3FilterBilinear( \
    fetch(context.p00), fetch(context.p10), fetch(context.p01), fetch(context.p11), context.weight)

FfxFloat32x3 LoadInputColor(FfxInt32x2 pos) {
    FfxInt32x2 p = clamp(pos, FfxInt32x2(0), RenderSize() - 1);
    return texelFetch(usam_main, p, 0).rgb;
}

FfxFloat32 LoadInputDepth(FfxInt32x2 pos) {
    FfxInt32x2 p = clamp(pos, FfxInt32x2(0), RenderSize() - 1);
    FfxFloat32 viewZ = texelFetch(usam_gbufferSolidViewZ, p, 0).r;
    return ffxSaturate(near / -viewZ);
}

FfxFloat32x2 LoadInputMotionVector(FfxInt32x2 pos) {
    return history_fsr3Motion_fetch(clamp(pos, FfxInt32x2(0), RenderSize() - 1)).xy;
}

FfxFloat32 LoadReactiveMask(FfxInt32x2 pos) {
    return history_fsr3Motion_fetch(clamp(pos, FfxInt32x2(0), RenderSize() - 1)).z;
}

FfxFloat32 SampleTransparencyAndCompositionMask(FfxFloat32x2 uv) {
    FSR3BilinearContext context = FSR3CreateBilinearContext(uv, RenderSize());
    return FSR3_FILTER_TILE(history_fsr3Motion_fetch, context).w;
}

FfxInt32x2 GetTransparencyAndCompositionMaskResourceDimensions() {
    return RenderSize();
}

#ifdef FSR3_BIND_PREPARE_INPUTS
void StoreDilatedMotionVector(FfxInt32x2 pos, FfxFloat32x2 value) {
    history_fsr3DilatedMotion_store(pos, FfxFloat32x4(value, 0.0f, 0.0f));
}
#endif

FfxFloat32x2 LoadDilatedMotionVector(FfxInt32x2 pos) {
    return history_fsr3DilatedMotion_fetch(clamp(pos, FfxInt32x2(0), RenderSize() - 1)).xy;
}

#ifdef FSR3_BIND_PREPARE_INPUTS
void StoreDilatedDepth(FfxInt32x2 pos, FfxFloat32 value) {
    history_fsr3DilatedDepth_store(pos, FfxFloat32x4(value));
}
#endif

FfxFloat32 LoadDilatedDepth(FfxInt32x2 pos) {
    return history_fsr3DilatedDepth_fetch(clamp(pos, FfxInt32x2(0), RenderSize() - 1)).r;
}

#ifdef FSR3_BIND_PREPARE_INPUTS
void StoreReconstructedDepth(FfxInt32x2 pos, FfxFloat32 value) {
    imageAtomicMax(uimg_fsr3ReconstructedDepth, pos, floatBitsToUint(value));
}
#endif

FfxFloat32 LoadReconstructedPrevDepth(FfxInt32x2 pos) {
    FfxInt32x2 p = clamp(pos, FfxInt32x2(0), RenderSize() - 1);
    return uintBitsToFloat(texelFetch(usam_fsr3ReconstructedDepth, p, 0).r);
}

#ifdef FSR3_BIND_PREPARE_INPUTS
void StoreFarthestDepth(FfxInt32x2 pos, FfxFloat32 value) {
    history_fsr3FarthestDepth_store(pos, FfxFloat32x4(value));
}
#endif

FfxFloat32 LoadFarthestDepth(FfxInt32x2 pos) {
    return history_fsr3FarthestDepth_fetch(clamp(pos, FfxInt32x2(0), RenderSize() - 1)).r;
}

#ifdef FSR3_BIND_LUMA_PYRAMID
void StoreFarthestDepthMip1(FfxInt32x2 pos, FfxFloat32 value) {
    if (all(lessThan(pos, ffxMax(FfxInt32x2(1), (RenderSize() + 1) / 2)))) {
        history_fsr3FarthestDepthMip1_store(pos, FfxFloat32x4(value));
    }
}
#endif

FfxInt32x2 GetFarthestDepthMip1ResourceDimensions() {
    return ffxMax(FfxInt32x2(1), (RenderSize() + 1) / 2);
}

FfxFloat32 SampleFarthestDepthMip1(FfxFloat32x2 uv) {
    FfxInt32x2 size = GetFarthestDepthMip1ResourceDimensions();
    FSR3BilinearContext context = FSR3CreateBilinearContext(uv, size);
    return FSR3_FILTER_TILE(history_fsr3FarthestDepthMip1_fetch, context).r;
}

FfxInt32x2 FSR3RenderHistoryTexel(FfxInt32x2 pos, FfxInt32x2 tile) {
    return pos + FfxInt32x2(tile.x * RenderSize().x, UpscaleSize().y + tile.y * RenderSize().y);
}

#define FSR3_CURRENT_LUMA_EVEN_FETCH(pos) texelFetch(usam_fsr3UpscaleAtlas, FSR3RenderHistoryTexel(pos, FfxInt32x2(0, 0)), 0)
#define FSR3_CURRENT_LUMA_ODD_FETCH(pos) texelFetch(usam_fsr3UpscaleAtlas, FSR3RenderHistoryTexel(pos, FfxInt32x2(1, 0)), 0)
#define FSR3_LUMA_HISTORY_EVEN_FETCH(pos) texelFetch(usam_fsr3UpscaleAtlas, FSR3RenderHistoryTexel(pos, FfxInt32x2(0, 1)), 0)
#define FSR3_LUMA_HISTORY_ODD_FETCH(pos) texelFetch(usam_fsr3UpscaleAtlas, FSR3RenderHistoryTexel(pos, FfxInt32x2(1, 1)), 0)

#ifdef FSR3_BIND_PREPARE_INPUTS
void StoreCurrentLuma(FfxInt32x2 pos, FfxFloat32 value) {
    FfxInt32x2 tile = (frameCounter & 1) == 0 ? FfxInt32x2(0, 0) : FfxInt32x2(1, 0);
    imageStore(uimg_fsr3UpscaleAtlas, FSR3RenderHistoryTexel(pos, tile), FfxFloat32x4(value, 0.0f, 0.0f, 0.0f));
}
#endif

FfxFloat32 LoadCurrentLuma(FfxInt32x2 pos) {
    FfxInt32x2 p = clamp(pos, FfxInt32x2(0), RenderSize() - 1);
    return (frameCounter & 1) == 0 ? FSR3_CURRENT_LUMA_EVEN_FETCH(p).r : FSR3_CURRENT_LUMA_ODD_FETCH(p).r;
}

FfxFloat32 LoadPreviousLuma(FfxInt32x2 pos) {
    if (FSR3HistoryReset()) return 0.0f;
    FfxInt32x2 p = clamp(pos, FfxInt32x2(0), PreviousFrameRenderSize() - 1);
    return (frameCounter & 1) == 0 ? FSR3_CURRENT_LUMA_ODD_FETCH(p).r : FSR3_CURRENT_LUMA_EVEN_FETCH(p).r;
}

FfxFloat32 SampleCurrentLuma(FfxFloat32x2 uv) {
    FSR3BilinearContext context = FSR3CreateBilinearContext(uv, RenderSize());
    if ((frameCounter & 1) == 0) return FSR3_FILTER_TILE(FSR3_CURRENT_LUMA_EVEN_FETCH, context).r;
    return FSR3_FILTER_TILE(FSR3_CURRENT_LUMA_ODD_FETCH, context).r;
}

#ifdef FSR3_BIND_LUMA_INSTABILITY
void StoreLumaHistory(FfxInt32x2 pos, FfxFloat32x4 value) {
    FfxInt32x2 tile = (frameCounter & 1) == 0 ? FfxInt32x2(0, 1) : FfxInt32x2(1, 1);
    imageStore(uimg_fsr3UpscaleAtlas, FSR3RenderHistoryTexel(pos, tile), value);
}
#endif

FfxFloat32x4 SampleLumaHistory(FfxFloat32x2 uv) {
    if (FSR3HistoryReset()) return FfxFloat32x4(0.0f);
    FSR3BilinearContext context = FSR3CreateBilinearContext(uv, PreviousFrameRenderSize());
    if ((frameCounter & 1) == 0) return FSR3_FILTER_TILE(FSR3_LUMA_HISTORY_ODD_FETCH, context);
    return FSR3_FILTER_TILE(FSR3_LUMA_HISTORY_EVEN_FETCH, context);
}

#ifdef FSR3_BIND_PREPARE_REACTIVITY
void StoreAccumulation(FfxInt32x2 pos, FfxFloat32 value) {
    FfxInt32x2 tile = (frameCounter & 1) == 0 ? FfxInt32x2(0, 0) : FfxInt32x2(1, 0);
    FfxInt32x2 atlasPos = FSR3RenderHistoryTexel(pos, tile);
    FfxFloat32x4 data = imageLoad(uimg_fsr3UpscaleAtlas, atlasPos);
    data.g = value;
    imageStore(uimg_fsr3UpscaleAtlas, atlasPos, data);
}
#endif

FfxFloat32 SampleAccumulation(FfxFloat32x2 uv) {
    if (FSR3HistoryReset()) return 0.0f;
    FSR3BilinearContext context = FSR3CreateBilinearContext(uv, PreviousFrameRenderSize());
    if ((frameCounter & 1) == 0) return FSR3_FILTER_TILE(FSR3_CURRENT_LUMA_ODD_FETCH, context).g;
    return FSR3_FILTER_TILE(FSR3_CURRENT_LUMA_EVEN_FETCH, context).g;
}

#ifdef FSR3_BIND_PREPARE_REACTIVITY
void StoreDilatedReactiveMasks(FfxInt32x2 pos, FfxFloat32x4 value) {
    history_fsr3ReactiveMasks_store(pos, value);
}
#endif

FfxFloat32x4 SampleDilatedReactiveMasks(FfxFloat32x2 uv) {
    FSR3BilinearContext context = FSR3CreateBilinearContext(uv, RenderSize());
    return FSR3_FILTER_TILE(history_fsr3ReactiveMasks_fetch, context);
}

#ifdef FSR3_BIND_SHADING_CHANGE
void StoreShadingChange(FfxInt32x2 pos, FfxFloat32 value) {
    history_fsr3ShadingChange_store(pos, FfxFloat32x4(value));
}
#endif

FfxFloat32 SampleShadingChange(FfxFloat32x2 uv) {
    FSR3BilinearContext context = FSR3CreateBilinearContext(uv, GetFarthestDepthMip1ResourceDimensions());
    return FSR3_FILTER_TILE(history_fsr3ShadingChange_fetch, context).r;
}

#ifdef FSR3_BIND_LUMA_INSTABILITY
void StoreLumaInstability(FfxInt32x2 pos, FfxFloat32 value) {
    history_fsr3FarthestDepth_store(pos, FfxFloat32x4(value));
}
#endif

FfxFloat32 SampleLumaInstability(FfxFloat32x2 uv) {
    FSR3BilinearContext context = FSR3CreateBilinearContext(uv, RenderSize());
    return FSR3_FILTER_TILE(history_fsr3FarthestDepth_fetch, context).r;
}

FfxInt32x2 GetSPDMipDimensions(FfxInt32 level) {
    FfxInt32 divisor = 1 << (level + 1);
    return ffxMax(FfxInt32x2(1), (RenderSize() + divisor - 1) / divisor);
}

FfxInt32x2 FSR3PyramidOffset(FfxInt32 level) {
    if (level == 0) return FfxInt32x2(0);
    FfxInt32x2 offset = FfxInt32x2(GetSPDMipDimensions(0).x, 0);
    for (FfxInt32 i = 1; i < level; i++) offset.y += GetSPDMipDimensions(i).y;
    return offset;
}

#if defined(FSR3_BIND_LUMA_PYRAMID) || defined(FSR3_BIND_SHADING_CHANGE_PYRAMID)
void StorePyramid(FfxInt32x2 pos, FfxFloat32x2 value, FfxUInt32 level) {
    #if defined(FSR3_BIND_LUMA_PYRAMID)
    // Rebuild log luma from the unaffected 64x64 linear-luma reduction.
    if (level == 5u) value.x = log(ffxMax(value.y, 6.10e-5f));
    #endif
    history_fsr3Pyramid_store(FSR3PyramidOffset(FfxInt32(level)) + pos, FfxFloat32x4(value, 0.0f, 0.0f));
}

FfxFloat32x2 RWLoadPyramid(FfxInt32x2 pos, FfxUInt32 level) {
    return history_fsr3Pyramid_load(FSR3PyramidOffset(FfxInt32(level)) + pos).xy;
}
#endif

FfxFloat32x2 SampleSPDMipLevel(FfxFloat32x2 uv, FfxInt32 level) {
    FfxInt32x2 size = GetSPDMipDimensions(level);
    FfxFloat32x2 texel = uv * FfxFloat32x2(size) - 0.5f;
    FfxInt32x2 base = FfxInt32x2(floor(texel));
    FfxFloat32x2 weight = ffxFract(texel);
    FfxInt32x2 p00 = clamp(base, FfxInt32x2(0), size - 1);
    FfxInt32x2 p10 = clamp(base + FfxInt32x2(1, 0), FfxInt32x2(0), size - 1);
    FfxInt32x2 p01 = clamp(base + FfxInt32x2(0, 1), FfxInt32x2(0), size - 1);
    FfxInt32x2 p11 = clamp(base + FfxInt32x2(1, 1), FfxInt32x2(0), size - 1);
    FfxInt32x2 offset = FSR3PyramidOffset(level);
    FfxFloat32x2 row0 = ffxLerp(history_fsr3Pyramid_fetch(offset + p00).xy, history_fsr3Pyramid_fetch(offset + p10).xy, weight.x);
    FfxFloat32x2 row1 = ffxLerp(history_fsr3Pyramid_fetch(offset + p01).xy, history_fsr3Pyramid_fetch(offset + p11).xy, weight.x);
    return ffxLerp(row0, row1, weight.y);
}

FfxInt32 MipCount() {
    return 12;
}

FfxInt32 NumWorkGroups() {
    FfxInt32x2 groups = (RenderSize() + 63) / 64;
    return groups.x * groups.y;
}

FfxInt32x2 WorkGroupOffset() {
    return FfxInt32x2(0);
}

void SPD_IncreaseAtomicCounter(inout FfxUInt32 counter) {
    counter = atomicAdd(global_atomicCounters[15], 1u);
}

void SPD_ResetAtomicCounter() {
    global_atomicCounters[15] = 0u;
}

FfxFloat32x4 LoadFrameInfo() {
    FfxFloat32x4 frameInfo = FSR3HistoryReset() ? FfxFloat32x4(1.0f, 1.0e4f, 1.0f, 0.0f) : global_fsr3FrameInfo;
    frameInfo.y = 1.0e4f;
    return frameInfo;
}

FfxFloat32x4 FrameInfo() {
    return global_fsr3FrameInfo;
}

void StoreFrameInfo(FfxFloat32x4 value) {
    global_fsr3FrameInfo = value;
}

FfxInt32x2 FSR3History1Texel(FfxInt32x2 pos) {
    return pos;
}

FfxInt32x2 FSR3History2Texel(FfxInt32x2 pos) {
    return pos + FfxInt32x2(UpscaleSize().x, 0);
}

FfxInt32x2 FSR3OutputTexel(FfxInt32x2 pos) {
    return pos + FfxInt32x2(UpscaleSize().x * 2, 0);
}

FfxFloat32x4 LoadHistory(FfxInt32x2 pos) {
    if (FSR3HistoryReset()) return FfxFloat32x4(0.0f);
    FfxInt32x2 p = clamp(pos, FfxInt32x2(0), PreviousFrameUpscaleSize() - 1);
    return (frameCounter & 1) == 0
        ? texelFetch(usam_fsr3UpscaleAtlas, FSR3History2Texel(p), 0)
        : texelFetch(usam_fsr3UpscaleAtlas, FSR3History1Texel(p), 0);
}

#ifdef FSR3_BIND_ACCUMULATE
void StoreInternalColorAndWeight(FfxInt32x2 pos, FfxFloat32x4 value) {
    if ((frameCounter & 1) == 0) imageStore(uimg_fsr3UpscaleAtlas, FSR3History1Texel(pos), value);
    else imageStore(uimg_fsr3UpscaleAtlas, FSR3History2Texel(pos), value);
}
#endif

FfxFloat32x4 LoadRCAS_Input(FfxInt32x2 pos) {
    FfxInt32x2 p = clamp(pos, FfxInt32x2(0), UpscaleSize() - 1);
    return (frameCounter & 1) == 0
        ? texelFetch(usam_fsr3UpscaleAtlas, FSR3History1Texel(p), 0)
        : texelFetch(usam_fsr3UpscaleAtlas, FSR3History2Texel(p), 0);
}

#ifdef FSR3_BIND_ACCUMULATE
FfxFloat32 LoadRwNewLocks(FfxInt32x2 pos) {
    return imageLoad(uimg_fsr3UpscaleAtlas, FSR3OutputTexel(pos)).a;
}
#endif

#if defined(FSR3_BIND_PREPARE_REACTIVITY) || defined(FSR3_BIND_ACCUMULATE)
void StoreNewLocks(FfxInt32x2 pos, FfxFloat32 value) {
    FfxInt32x2 atlasPos = FSR3OutputTexel(pos);
    FfxFloat32x4 data = imageLoad(uimg_fsr3UpscaleAtlas, atlasPos);
    data.a = value;
    imageStore(uimg_fsr3UpscaleAtlas, atlasPos, data);
}
#endif

#endif
