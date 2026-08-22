# Global Illumination

Language: English | [简体中文](../../sc/modules/global-illumination.md)

The current GI path is a screen-space ReSTIR/SST pipeline backed by an environment probe and a separate spatiotemporal
denoiser. Pass order comes from [`scripts/programs.main.kts`](../../../scripts/programs.main.kts); shared algorithms
live under [`shaders/techniques/gi/`](../../../shaders/techniques/gi/) and entry points are in [
`shaders/pass/composite/`](../../../shaders/pass/composite/).

## Code map

| Path                                                                                                                                                                                                                      | Responsibility                                 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| [`shaders/techniques/gi/Common.glsl`](../../../shaders/techniques/gi/Common.glsl)                                                                                                                                         | Common GI data and accessors                   |
| [`InitialSample.glsl`](../../../shaders/techniques/gi/InitialSample.glsl), [`RaySort.glsl`](../../../shaders/techniques/gi/RaySort.glsl), [`FinishTrace.comp.glsl`](../../../shaders/techniques/gi/FinishTrace.comp.glsl) | Initial sampling and long-SST sort/finish      |
| [`Reservoir.glsl`](../../../shaders/techniques/gi/Reservoir.glsl), [`PairwiseMISMetadata.glsl`](../../../shaders/techniques/gi/PairwiseMISMetadata.glsl)                                                                  | Reservoir encoding and pairwise-reuse metadata |
| [`ResampleMaterial.glsl`](../../../shaders/techniques/gi/ResampleMaterial.glsl)                                                                                                                                           | Material representation used during reuse      |
| [`Reproject.glsl`](../../../shaders/techniques/gi/Reproject.glsl), [`ReprojectInfo.glsl`](../../../shaders/techniques/gi/ReprojectInfo.glsl)                                                                              | History reprojection                           |
| [`Irradiance.glsl`](../../../shaders/techniques/gi/Irradiance.glsl)                                                                                                                                                       | Shared irradiance and shading calculations     |
| [`DenoiserEdgeClassification.glsl`](../../../shaders/techniques/gi/DenoiserEdgeClassification.glsl), [`DenoiseBlur.glsl`](../../../shaders/techniques/gi/DenoiseBlur.glsl)                                                | Denoiser edge and blur kernels                 |
| [`shaders/techniques/EnvProbe.glsl`](../../../shaders/techniques/EnvProbe.glsl)                                                                                                                                           | Environment-probe mapping and sampling         |
| [`shaders/techniques/SST2.glsl`](../../../shaders/techniques/SST2.glsl), [`HiZ.glsl`](../../../shaders/techniques/HiZ.glsl), [`HiZCheck.glsl`](../../../shaders/techniques/HiZCheck.glsl)                                 | Screen-space tracing and Hi-Z queries          |

## Input preparation

| Order | Stage / pass                                                                                                                                                                                                                                                                                                                                                                                             | Purpose                                                                           |
|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 1     | Geometry                                                                                                                                                                                                                                                                                                                                                                                                 | Produces depth, normals, roughness, material, and light-map inputs                |
| 2     | [`HiZGen`](../../../shaders/pass/composite/HiZGen.csh), [`GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl), [`GIDenoiserEdgeDilation`](../../../shaders/pass/composite/GIDenoiserEdgeDilation.comp.glsl), [`GIDenoiserReproject`](../../../shaders/pass/composite/GIDenoiserReproject.comp.glsl) | Builds Hi-Z, classifies/dilates GI edges, and reprojects history early            |
| 3     | [`DirectLighting`](../../../shaders/pass/composite/DirectLighting.glsl)                                                                                                                                                                                                                                                                                                                                  | Consumes the same G-buffer/shadow state before GI, so material decoding is shared |

## Environment probe

The probe preserves low-frequency/history scene information for GI queries that leave the current screen. Its update is
interleaved with GI preparation:

| Order | Pass                                                                                                           | Purpose                                                   |
|-------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| 1     | [`EnvProbeUpdate1ReprojectScatter`](../../../shaders/pass/composite/EnvProbeUpdate1ReprojectScatter.comp.glsl) | Reprojects and scatters the old probe                     |
| 2     | [`EnvProbeUpdate2ReprojectDilate`](../../../shaders/pass/composite/EnvProbeUpdate2ReprojectDilate.comp.glsl)   | Fills reprojection holes twice with `PASS=1` and `PASS=2` |
| 3     | [`EnvProbeUpdate3ReprojectGather`](../../../shaders/pass/composite/EnvProbeUpdate3ReprojectGather.comp.glsl)   | Gathers valid reprojected data                            |
| 4     | [`EnvProbeUpdate4ProjectCurrent`](../../../shaders/pass/composite/EnvProbeUpdate4ProjectCurrent.comp.glsl)     | Projects current-frame results back into the probe        |

The runtime resources are `uimg_envProbe`, declared as 1024×768 RGBA32UI in [
`shaders/shaders.properties`](../../../shaders/shaders.properties), and the fixed 1024×768 RGBA16F
`persistent_envProbeTemp` in [`shaders/shadesmith.json`](../../../shaders/shadesmith.json). [
`ClearEnvProbe`](../../../shaders/pass/begin/ClearEnvProbe.comp.glsl) clears the probe when needed.

## ReSTIR/SST pass flow

| Order | Pass                                                                                                                                                                                                                         | Notes                                                              |
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| 1     | [`GIReSTIRInitalSampleRayGenTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayGenTrace.comp.glsl)                                                                                                               | Generates initial candidates and starts SST                        |
| 2     | [`GIReSTIRInitalSampleRaySort`](../../../shaders/pass/composite/GIReSTIRInitalSampleRaySort.comp.glsl), [`GIReSTIRInitalSampleRayFinishTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayFinishTrace.comp.glsl) | Only for initial SST steps ≥ 64; sorts and completes long paths    |
| 3     | [`GIReSTIRTemporalReuse`](../../../shaders/pass/composite/GIReSTIRTemporalReuse.comp.glsl)                                                                                                                                   | Reprojects previous reservoirs, samples, hit normals, and material |
| 4     | [`GIReSTIRDuplicationMapDecorrelate`](../../../shaders/pass/composite/GIReSTIRDuplicationMapDecorrelate.comp.glsl)                                                                                                           | Optional decorrelation                                             |
| 5     | [`GIReSTIRPairedSpatialReuse`](../../../shaders/pass/composite/GIReSTIRPairedSpatialReuse.comp.glsl) × 1–4                                                                                                                   | Pairwise reuse in batches of seven base samples                    |
| 6     | [`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl)                                                                                                                         | Shades selected samples                                            |
| 7     | [`GIReSTIRSpatialReuseRaySort`](../../../shaders/pass/composite/GIReSTIRSpatialReuseRaySort.comp.glsl)                                                                                                                       | Compacts rays still requiring a trace                              |
| 8     | [`GIReSTIRSpatialReuseTrace`](../../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl)                                                                                                                           | Completes spatial visibility/SST                                   |

The four spatial-reuse passes use `PASS_INDEX` 0–3 and `PASS_BASE_SAMPLE_INDEX` 0/7/14/21, dispatched indirectly from
SSBO 0 offset 48. `history_restir_reservoirTemporal`, `history_restir_primary`, `history_restir_prevSample`, and
`history_restir_prevHitNormal` retain previous-frame inputs; `transient_restir_reservoirTemporal`,
`transient_restir_primary`, `transient_restir_spatialInput`, and `transient_restir_pairwiseMISMetadata` connect the
current-frame stages. [
`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl) copies the current
temporal reservoir and primary data to their fixed history tiles while performing the final current-frame reads. All
tile definitions live in [`shaders/shadesmith.json`](../../../shaders/shadesmith.json).

## GI denoising

After ReSTIR shading, the pipeline runs:

| Order | Pass                                                                                                                                                               | Purpose                                                                            |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| 1     | [`GIDenoiserAccum`](../../../shaders/pass/composite/GIDenoiserAccum.comp.glsl)                                                                                     | Temporal accumulation; updates `history_gi1`…`history_gi5`                         |
| 2     | [`GIDenoiserAntiFireFly`](../../../shaders/pass/composite/GIDenoiserAntiFireFly.comp.glsl)                                                                         | Optional anti-firefly pass                                                         |
| 3     | [`GIDenoiserGIMip`](../../../shaders/pass/composite/GIDenoiserGIMip.comp.glsl)                                                                                     | Builds diffuse/specular mip inputs through indirect dispatch from SSBO 0 offset 16 |
| 4     | [`GIDenoiserHistoryFix`](../../../shaders/pass/composite/GIDenoiserHistoryFix.comp.glsl)                                                                           | Repairs low-confidence history                                                     |
| 5     | [`GIDenoiserBlur`](../../../shaders/pass/composite/GIDenoiserBlur.comp.glsl), [`GIDenoiserPostBlur`](../../../shaders/pass/composite/GIDenoiserPostBlur.comp.glsl) | Optional blur and post-blur passes                                                 |

Reprojection also depends on `history_viewZ`, historical/current view normals, geometry normals, edge masks, roughness,
and average view-Z. Tile lifetime and format changes belong in [
`shaders/shadesmith.json`](../../../shaders/shadesmith.json), not only in samplers.

## Settings

GI settings are organized in the GI and denoiser screens of [
`scripts/options.main.kts`](../../../scripts/options.main.kts):

- Trace: `SETTING_GI_INITIAL_SST_STEPS`, `SETTING_GI_VALIDATE_SST_STEPS`, `SETTING_GI_SST_THICKNESS`.
- Probe/sky: `SETTING_GI_PROBE_FADE_START/END`, `SETTING_GI_MC_SKYLIGHT_ATTENUATION`.
- Reuse: `SETTING_GI_TEMPORAL_REUSE_LIMIT`, `SETTING_GI_SPATIAL_REUSE`, `SETTING_GI_SPATIAL_REUSE_COUNT`,
  `SETTING_GI_DECORRELATE`.
- Denoiser: spatial enable/sample counts, history lengths, fast-history clamping, flicker suppression, anti-firefly, and
  history-fix weights.

Profiles mainly scale SST steps, spatial reuse count, and denoiser sample counts. Any new `SETTING_*` must be registered
in the options DSL before GLSL or program conditions use it.

## Maintenance invariants

- Change reservoir packing, MIS metadata, and all producers/consumers together.
- Temporal tiles must agree with current/previous jitter, camera transforms, and G-buffer semantics.
- Edge classification/dilation stays before reprojection and accumulation.
- A spatial batch-size change must update program thresholds, base indices, and the indirect queue layout together.
- Validate convergence, camera motion, disocclusion, screen edges, and history reset after setting changes.
