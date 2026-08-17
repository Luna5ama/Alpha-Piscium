# Rendering Pipeline Overview

Language: English | [简体中文](../../sc/modules/pipeline.md)

[`scripts/programs.main.kts`](../../../scripts/programs.main.kts) is the source of truth for Alpha Piscium's compute
order. It emits root `.csh` wrappers and [
`scripts/programs.shaders.properties`](../../../scripts/programs.shaders.properties) in DSL order. `PREPARE` and
`DEFERRED` currently have no entries.

## High-level frame order

```text
setup (initialization / resize)
  ↓
begin (frame data, clears, LUT preparation)
  ↓
shadow geometry → shadowcomp
  ↓
prepare (currently empty)
  ↓
opaque geometry / G-buffer
  ↓
deferred (currently empty)
  ↓
water and translucent geometry
  ↓
scene-preparation, GI, caustics, cloud, shadow, and lighting passes
  ↓
DOFPrepare, TAAPrepare, then the selected internal Off/TAA/FSR 3 or external-SR reconstruction path
  ↓
Bloom downsample/upsample
  ↓
PostComposite display transform, exposure, next-frame state, and OverlayComposite
  ↓
final (dither and screen output)
```

This is the project-local dependency order, not a replacement for Iris program-stage semantics.

## Setup

[`InitGlobalData`](../../../shaders/pass/setup/InitGlobalData.comp.glsl) initializes global data. [
`ClearRGBA32UI`](../../../shaders/pass/setup/ClearRGBA32UI.glsl), [
`ClearRGBA16F`](../../../shaders/pass/setup/ClearRGBA16F.glsl), [
`ClearRGB10A2`](../../../shaders/pass/setup/ClearRGB10A2.glsl), [
`ClearRGBA8`](../../../shaders/pass/setup/ClearRGBA8.glsl), and [
`ClearR32F`](../../../shaders/pass/setup/ClearR32F.glsl) run format-oriented clears. Format-level clears avoid a
dedicated shader for every tile.

## Begin

Same-index `_a`, `_b`, and `_c` are parallel entry points explicitly grouped in the program DSL. [
`scripts/shaders.properties`](../../../scripts/shaders.properties) enables `allowConcurrentCompute=true`, so entries in
a group must be independent and cannot rely on suffix-letter execution order; only numbered groups express the pipeline
sequencing recorded here. Current groups:

| Group                                                                                                                                                                                                                                                                                                                                      | Work                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [`UpdateGlobalData`](../../../shaders/pass/begin/UpdateGlobalData.comp.glsl), [`ClearRTWSM`](../../../shaders/pass/begin/ClearRTWSM.comp.glsl), [`GenerateTransmittance`](../../../shaders/techniques/atmospherics/air/lut/GenerateTransmittance.comp.glsl)                                                                                | Update global data, clear `persistent_rtwsm_importance2D`, generate transmittance LUT |
| [`SliceEndPoints`](../../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl), [`GenerateMultiSctr`](../../../shaders/techniques/atmospherics/air/lut/GenerateMultiSctr.comp.glsl), [`ClearScreen1`](../../../shaders/pass/begin/ClearScreen1.comp.glsl), [`ClearScreen2`](../../../shaders/pass/begin/ClearScreen2.comp.glsl) | Slice endpoints, multi-scattering LUT, screen clears 1/2                              |
| [`Sample`](../../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl), [`GenerateSkyViewLUT`](../../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl)                                                                                                                                                 | Cloud ambient-LUT sampling and sky-view LUT                                           |
| [`Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl), [`ClearEnvProbe`](../../../shaders/pass/begin/ClearEnvProbe.comp.glsl), [`InitThreadGroupTilling`](../../../shaders/pass/begin/InitThreadGroupTilling.glsl)                                                                                           | Cloud ambient-LUT gather, environment-probe clear, thread-group tiling init           |
| [`ClearScreen3`](../../../shaders/pass/begin/ClearScreen3.comp.glsl)                                                                                                                                                                                                                                                                       | `VOXY` only: clear the third screen-resource group                                    |

## Shadow and Shadowcomp

After frame preparation, root `shadow*.vsh/.gsh/.fsh` wrappers connect world, entity, cutout, water, and block shadow
geometry to [`shaders/pass/geometry/ShadowPass.*.glsl`](../../../shaders/pass/geometry/). [
`EvaluateShadowWaterNormal`](../../../shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl) then prepares shadow-water
normals for later sampling. See [Shadows](shadows.md).

## Geometry and G-buffer

Geometry entry points are not numbered by the program DSL. Active root `gbuffers_*` and `dh_*` wrappers include
`GBufferSolid.*.glsl` or `GBufferTranslucent.*.glsl`, while some wrappers are explicit NOOPs; `voxy_*` implements
`voxy_emitFragment` directly. Their outputs feed composite material, depth, normal, and translucency work.
See [Geometry, G-buffer, and Materials](geometry-materials.md) and [Water and Translucency](water-translucency.md).

## Scene preparation

| Order                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Pass flow                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`VoxyMerge`](../../../shaders/pass/composite/VoxyMerge.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Optional Voxy merge                                                                                                       |
| [`EnvProbeUpdate1ReprojectScatter`](../../../shaders/pass/composite/EnvProbeUpdate1ReprojectScatter.comp.glsl), [`HiZGen`](../../../shaders/pass/composite/HiZGen.csh), [`EnvProbeUpdate2ReprojectDilate`](../../../shaders/pass/composite/EnvProbeUpdate2ReprojectDilate.comp.glsl), [`GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl), [`ShadowSampleSetup`](../../../shaders/pass/composite/ShadowSampleSetup.comp.glsl), [`GIDenoiserEdgeDilation`](../../../shaders/pass/composite/GIDenoiserEdgeDilation.comp.glsl), [`GIDenoiserReproject`](../../../shaders/pass/composite/GIDenoiserReproject.comp.glsl) | Environment-probe reproject/scatter/dilate; Hi-Z; denoiser edge classification/dilation/reprojection; shadow-sample setup |
| [`EvaluateScreenPixelSize`](../../../shaders/pass/composite/EvaluateScreenPixelSize.comp.glsl), [`CausticsPhotonTrace`](../../../shaders/pass/composite/CausticsPhotonTrace.comp.glsl), [`CausticsRemap`](../../../shaders/pass/composite/CausticsRemap.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Optional water caustics: pixel size → photon trace → remap                                                                |
| [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl), [`clouds/ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Optional cumulus render → temporal/spatial accumulation                                                                   |
| [`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl), [`ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl), [`EnvProbeUpdate3ReprojectGather`](../../../shaders/pass/composite/EnvProbeUpdate3ReprojectGather.comp.glsl), [`DirectLighting`](../../../shaders/pass/composite/DirectLighting.glsl)                                                                                                                                                                                                                                                                                                                                                                            | SSS shadow samples → main shadow sample; environment-probe gather and direct lighting                                     |
| [`DOFFocus`](../../../shaders/pass/composite/DOFFocus.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Automatic DOF focus when DOF is enabled and focus is not manual                                                           |
| [`EnvProbeUpdate4ProjectCurrent`](../../../shaders/pass/composite/EnvProbeUpdate4ProjectCurrent.comp.glsl), [`GIReSTIRInitalSampleRayGenTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayGenTrace.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Environment-probe project-current and GI initial ray generation/trace                                                     |

[`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl) uses indirect dispatch from SSBO 0 offset

32. [`EnvProbeUpdate2ReprojectDilate`](../../../shaders/pass/composite/EnvProbeUpdate2ReprojectDilate.comp.glsl) is
    reused with `PASS=1/2` for its scatter/dilate variants.

## ReSTIR GI and denoising

| Order | Pass / stage                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Purpose                                                                                           |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| 1     | [`GIReSTIRInitalSampleRaySort`](../../../shaders/pass/composite/GIReSTIRInitalSampleRaySort.comp.glsl), [`GIReSTIRInitalSampleRayFinishTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayFinishTrace.comp.glsl)                                                                                                                                                                                                                                                                                             | Optional sort/finish for long initial SST paths (`SETTING_GI_INITIAL_SST_STEPS >= 64`)            |
| 2     | [`GIReSTIRTemporalReuse`](../../../shaders/pass/composite/GIReSTIRTemporalReuse.comp.glsl), [`GIReSTIRDuplicationMapDecorrelate`](../../../shaders/pass/composite/GIReSTIRDuplicationMapDecorrelate.comp.glsl)                                                                                                                                                                                                                                                                                                           | Temporal reuse and optional duplication-map decorrelation                                         |
| 3     | [`GIReSTIRPairedSpatialReuse`](../../../shaders/pass/composite/GIReSTIRPairedSpatialReuse.comp.glsl) × 1–4                                                                                                                                                                                                                                                                                                                                                                                                               | Up to four indirect batches gated at reuse counts `> 0/7/14/21`, dispatched from SSBO 0 offset 48 |
| 4     | [`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl), [`GIReSTIRSpatialReuseRaySort`](../../../shaders/pass/composite/GIReSTIRSpatialReuseRaySort.comp.glsl), [`GIReSTIRSpatialReuseTrace`](../../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl)                                                                                                                                                                                                         | Shades selected samples, sorts spatial rays, and completes tracing                                |
| 5     | [`GIDenoiserAccum`](../../../shaders/pass/composite/GIDenoiserAccum.comp.glsl), [`GIDenoiserAntiFireFly`](../../../shaders/pass/composite/GIDenoiserAntiFireFly.comp.glsl), [`GIDenoiserGIMip`](../../../shaders/pass/composite/GIDenoiserGIMip.comp.glsl), [`GIDenoiserHistoryFix`](../../../shaders/pass/composite/GIDenoiserHistoryFix.comp.glsl), [`GIDenoiserBlur`](../../../shaders/pass/composite/GIDenoiserBlur.comp.glsl), [`GIDenoiserPostBlur`](../../../shaders/pass/composite/GIDenoiserPostBlur.comp.glsl) | Denoiser accumulation, optional anti-firefly, GI mip, history fix, and optional blur/post-blur    |
| 6     | [`SSTStepDebug`](../../../shaders/pass/composite/SSTStepDebug.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Optional SST-step debug                                                                           |

See [Global Illumination](global-illumination.md).

## Volumetrics and translucency

[`EpipolarScatteringAir`](../../../shaders/pass/composite/EpipolarScatteringAir.comp.glsl)
↓
[`EpipolarScatteringWater`](../../../shaders/pass/composite/EpipolarScatteringWater.comp.glsl)
↓
[`TranslucentBackComposite`](../../../shaders/pass/composite/TranslucentBackComposite.glsl)
↓
[`TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl)
↓
[`TranslucentComposite`](../../../shaders/pass/composite/TranslucentComposite.glsl)
↓
[`IMapCollapse`](../../../shaders/techniques/rtwsm/IMapCollapse.comp.glsl)
↓
optional [
`VolumetricLocalCompositeBreakFix`](../../../shaders/pass/composite/VolumetricLocalCompositeBreakFix.comp.glsl)

The last stage uses indirect dispatch from SSBO 0 offset 0.

## Post-processing

| Range                                                                                                                                                                                                                                                             | Flow                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`DOFPrepare`](../../../shaders/pass/composite/DOFPrepare.comp.glsl)                                                                                                                                                                                              | Optional DOF prepare                                                            |
| [`TAAPrepare`](../../../shaders/pass/composite/TAAPrepare.comp.glsl) → [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) → [`FXAA`](../../../shaders/pass/composite/FXAA.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | Internal non-FSR3 temporal AA, optional spatial AA, and sharpening               |
| [`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) → [`FSR3PrepareInputs`](../../../shaders/pass/composite/FSR3PrepareInputs.comp.glsl) → FSR3 pyramid/reactivity stages → [`FSR3Accumulate`](../../../shaders/pass/composite/FSR3Accumulate.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | Internal FSR3 temporal upscaling and shared RCAS output                         |
| [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) → [`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) → [`superresolution.v3.json`](../../../shaders/superresolution.v3.json) | External SR consumes `colortex31` motion and exposed-linear HDR, then writes full-resolution `colortex0` immediately before Bloom |
| [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl)                                                                                                                                                                                                            | Downsample levels 1–10 and upsample levels 10–2, capped by `SETTING_BLOOM_PASS` |
| [`IMapBlur`](../../../shaders/techniques/rtwsm/IMapBlur.comp.glsl) → [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl)                                                                                                                   | RTWSM importance blur, post composite, and display transform                    |
| [`GetWarp`](../../../shaders/techniques/rtwsm/GetWarp.comp.glsl) → [`ExposureMip`](../../../shaders/pass/composite/ExposureMip.comp.glsl)                                                                                                                         | Next-frame RTWSM warp and exposure mip                                          |
| [`ExposureGather`](../../../shaders/pass/composite/ExposureGather.comp.glsl) → [`Write2DWarp`](../../../shaders/techniques/rtwsm/Write2DWarp.comp.glsl)                                                                                                           | Exposure gather and RTWSM 2D warp write                                         |
| [`FinalGlobalDataUpdate`](../../../shaders/pass/composite/FinalGlobalDataUpdate.comp.glsl) → [`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl)                                                                                     | Final global-data update and overlay composite                                  |

Root [`final.fsh`](../../../shaders/final.fsh) includes [
`Final.frag.glsl`](../../../shaders/pass/composite/Final.frag.glsl), which dithers the completed `colortex0` image and
writes the screen output. See [Post-processing](post-processing.md).

## Conditions and generation

`cond(...)` emits `program.<name>.enabled` preprocessor branches; `indirect(...)` emits a dispatch buffer/offset;
`define(...)` is a normal define, and `constDefine(...)` goes in Iris's const section.

After changing the program DSL, run:

```sh
cd scripts
kotlin programs.main.kts
```

If the program property fragment changes, run:

```sh
cd scripts
kotlin options.main.kts
```

This aggregates it into final [`shaders/shaders.properties`](../../../shaders/shaders.properties); options invokes the
program generator again.
