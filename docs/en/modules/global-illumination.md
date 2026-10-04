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
| 5     | [`GIReSTIRPairedSpatialReuse`](../../../shaders/pass/composite/GIReSTIRPairedSpatialReuse.comp.glsl) × 1–4                                                                                                                   | Pairwise reuse in batches of seven base samples; accumulates the specular BRDF-ratio resolve |
| 6     | [`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl)                                                                                                                         | Shades selected samples, blends specular toward the ratio resolve by roughness, and queues neighbor visibility rays |
| 7     | [`GIReSTIRSpatialReuseTrace`](../../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl)                                                                                                                           | Traces the compacted visibility queue; occluded samples contribute zero except the ratio specular |

The four spatial-reuse passes use `PASS_INDEX` 0–3 and `PASS_BASE_SAMPLE_INDEX` 0/7/14/21, dispatched indirectly from
SSBO 0 offset 48. `history_restir_reservoirTemporal`, `history_restir_primary`, `history_restir_prevSample`, and
`history_restir_prevHitNormal` retain previous-frame inputs; `transient_restir_reservoirTemporal`,
`transient_restir_primary`, `transient_restir_spatialInput`, and `transient_restir_pairwiseMISMetadata` connect the
current-frame stages. [
`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl) copies the current
temporal reservoir and primary data to their fixed history tiles while performing the final current-frame reads. All
tile definitions live in [`shaders/shadesmith.json`](../../../shaders/shadesmith.json).

Spatial shading writes provisional diffuse/specular results immediately. Neighbor selections that need voxel
visibility store the octahedral `resultY` direction, its hit distance, and the ratio blend (ratio specular and ReSTIR
weight) in `transient_restir_pairwiseMISMetadata`; each 16×16 tile then appends its
rays to SSBO 1 as one contiguous run ordered by world-direction octant and Morton position, counted by
`global_restirVisibilityRayCount` (reset in [`UpdateGlobalData`](../../../shaders/pass/begin/UpdateGlobalData.comp.glsl)).
The trace pass launches over the screen-sized queue capacity, exits past that count, and replaces only provisional
results that fail the voxel visibility test with zero diffuse and the ratio-only specular.

Specular is `mix(ratioSpec, restirSpec, roughness)`, with the linear GGX roughness of the center. The ratio resolve
runs over the center's and every same-plane paired pixel's temporal sample, across all spatial batches. Each sample
contributes its own-frame specular estimate `L·W·f_o` under the center material, capped by the sample's own target BRDF,
with weight `min(f_r / f_o, 1)`: `f_r` is the center's specular BRDF toward the sample hit and `f_o` the same BRDF in
the sample's own frame. It has no visibility or Jacobian term. The paired passes keep the weighted mean and weight sum in
`transient_ssgiSpecOut`, which is otherwise unused until spatial shading writes the final result.

## GI denoising

After ReSTIR shading, the pipeline runs:

| Order | Pass                                                                                                                                                               | Purpose                                                                            |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| 1     | [`GIDenoiserAccum`](../../../shaders/pass/composite/GIDenoiserAccum.comp.glsl)                                                                                     | Temporal accumulation; updates `history_gi1`…`history_gi5`                         |
| 2     | [`GIDenoiserAntiFireFly`](../../../shaders/pass/composite/GIDenoiserAntiFireFly.comp.glsl)                                                                         | Optional anti-firefly pass                                                         |
| 3     | [`GIDenoiserGIMip`](../../../shaders/pass/composite/GIDenoiserGIMip.comp.glsl)                                                                                     | Builds diffuse/specular mip inputs through indirect dispatch from SSBO 0 offset 16 |
| 4     | [`GIDenoiserHistoryFix`](../../../shaders/pass/composite/GIDenoiserHistoryFix.comp.glsl)                                                                           | Repairs low-confidence history                                                     |
| 5     | [`GIDenoiserBlur`](../../../shaders/pass/composite/GIDenoiserBlur.comp.glsl), [`GIDenoiserPostBlur`](../../../shaders/pass/composite/GIDenoiserPostBlur.comp.glsl) | Optional blur and post-blur passes                                                 |

Accum also updates `history_gi6`, whose RG channels store diffuse/specular luminance second moments with the fast-history
alpha. Reprojection uses the same bilinear or edge-aware four-tap weights as fast color; specular uses the virtual point.
The first moment is the working-space luminance of accumulated fast color. After accumulation, `transient_gi_variance`
stores diffuse/specular variance in RG and absolute standard deviation in BA, before anti-firefly, history fix, or blur.
HistoryFix filters variance with a 5x5 tent kernel (separable weights `[1, 2, 3, 2, 1]`, total weight 81) and stores
`transient_gi_filteredVariance`: RG contains filtered variance and BA its square root. Screen edges clamp to the
nearest pixel; non-solid pixels contribute zero. `SETTING_DEBUG_DENOISER` displays filtered variance or absolute
standard deviation as grayscale, using the common debug exposure and gamma controls. M2 history remains temporally
accumulated without spatial filtering; invalid history resets with fast color.

Both blur passes compute a stability factor `1 / (1 + variance / (fastLuminance² + 1e-6))`, independently per lobe.
This interpolates effective history lengths from 1 to their accumulated values, and hit-distance factors from 1 to
their existing values. Higher variance shortens effective history and relaxes hit-distance suppression before the
existing accumulation-factor and kernel calculations. Radius limits, geometric weights, and specular roughness
shaping continue to apply; stored temporal history lengths remain unchanged.

Reprojection also depends on `history_viewZ`, historical/current view normals, geometry normals, edge masks, roughness,
and average view-Z. Tile lifetime and format changes belong in [
`shaders/shadesmith.json`](../../../shaders/shadesmith.json), not only in samplers.

## Estimator contracts

- Voxel traces distinguish three outcomes (`voxel_traceExhausted` in [
  `VoxelTrace.glsl`](../../../shaders/techniques/voxel/VoxelTrace.glsl)): a hit, a grid exit that is shaded as sky (the
  grid-boundary assumption), and step exhaustion, which is a legal zero-radiance sample for initial candidates and is
  dropped by radiance cache (RC) candidates and revalidation.
- RC faces count zero-radiance samples in M and in the control-variate weight but never select them, so a face can hold
  a valid estimate without a selected sample (`rc_reservoirValid` versus `rc_reservoirHasSample` in [
  `RadianceCache.glsl`](../../../shaders/techniques/gi/RadianceCache.glsl)).
- A face estimate is the cosine-weighted mean incident radiance C_n. RC lookups and RC bounces shade it as uniform
  incidence, `(albedo * dielectric * (1 - F(NoV)) + E_spec(NoV)) * C_n` (`resampleMaterial_uniformIncidenceAlbedo` in [
  `ResampleMaterial.glsl`](../../../shaders/techniques/gi/ResampleMaterial.glsl)), independent of the face reservoir's
  sample direction.

## Monte Carlo reference

`SETTING_GI_USE_REFERENCE` (Debug screen) replaces the displayed GI with an FP32 running mean of the one-sample
estimate of the initial candidate, for measuring bias and variance of the sampler.

| Order | Pass | Purpose |
|-------|------|---------|
| 1 | [`GIReferenceAccumulate`](../../../shaders/pass/composite/GIReferenceAccumulate.comp.glsl) | Runs after the voxel fallback, accumulates `L / q` shaded like the reuse output into `uimg_giReferenceDiff/Spec` (RGBA32F; diffuse alpha holds the sum of squared luminance, specular alpha the sample count); resets on the first frame or any view, projection, or camera-position change |
| 2 | [`TranslucentBackComposite`](../../../shaders/pass/composite/TranslucentBackComposite.glsl) | Reads the mean from the reference images instead of the denoised GI |

Modes: 1 full Li; 2 frozen Li, which drops the RC term at hits so the integrand does not change over time; 3 furnace,
which accumulates `cos / (pi q)` and must average to 1 on flat surfaces with normal mapping off. Reuse and denoising
still run but do not reach the screen. The accumulation assumes static time of day.

## Path guiding

`SETTING_GI_PATH_GUIDING` (off by default, requires the RC) gives part of the diffuse initial-candidate and RC update
techniques to two folded von Mises-Fisher lobes learned per RC face from the final ReSTIR samples. Code lives in [
`PathGuiding.glsl`](../../../shaders/techniques/gi/PathGuiding.glsl); the sampler and mixture pdf are in [
`InitialSample.glsl`](../../../shaders/techniques/gi/InitialSample.glsl) and [
`RadianceCacheUpdate.glsl`](../../../shaders/techniques/gi/RadianceCacheUpdate.glsl).

RC fresh rays read the matching previous-side face statistics and apply the same decay and cap as [
`GIPathGuidePrepare`](../../../shaders/pass/composite/GIPathGuidePrepare.comp.glsl) in registers before fitting the lobes.
At most half of the 256 technique bins are guided; the remaining bins use cosine sampling.
The fresh CV estimate is `Li * (cos(theta) / pi) / q`, and the fresh RIS weight includes the same proposal correction.
Reservoir radiance stays unweighted, so temporal CV, spatial cosine-measure reconnection, and M clamping keep their
existing contracts. Training still comes only from final screen-space ReSTIR samples. Faces without usable statistics
use cosine sampling with correction 1. Fixed debug lobes also apply to RC rays. RC still drops exhausted traces;
changing the proposal can change that rejection rate, so this does not make the complete RC estimator unbiased.

| Order | Pass | Purpose |
|-------|------|---------|
| 1 | [`GIPathGuidePrepare`](../../../shaders/pass/composite/GIPathGuidePrepare.comp.glsl) | After the RC update, same indirect face list: moves each face's statistics from its previous-side slot to its current slot with a 0.9 decay and a 256-sample cap |
| 2 | [`GIReSTIRInitalSampleHiZ`](../../../shaders/pass/composite/GIReSTIRInitalSampleHiZ.comp.glsl) | Fits each lobe on read (at least 8 samples, mean resultant length at least 0.1, kappa at most 32); the fitted lobes share half of the diffuse technique bins in proportion to their sample counts |
| 3 | [`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl), [`GIReSTIRSpatialReuseTrace`](../../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl) | Write one training record per pixel (final sample direction, kept with probability f_d / (f_d + f_s)); the trace pass drops records of occluded neighbor samples |
| 4 | [`GIPathGuideSplat`](../../../shaders/pass/composite/GIPathGuideSplat.comp.glsl) | At quarter resolution, adds one rotating pixel per 4×4 tile to the nearer of its face's two online direction clusters with atomics (a direction with cosine below 0.5 to the first cluster seeds an empty second one); the result is used from the next frame |

Resources:

- SSBO 6 `pg_stats`: one uvec4 per lobe, RC slot, and side (fixed-point direction sums and sample count), 32 B × 2 ×
  `SETTING_RC_POOL_SIZE` (16 MiB at the default pool), declared only when guiding is enabled.
- `transient_pathGuide_trainRecord` (RG32UI): shares an existing RG32UI atlas slot, so the atlas does not grow.
- GlobalData `pg_*Counter` fields, shown by `SETTING_DEBUG_VOXEL_COUNTER`.

`SETTING_DEBUG_PATH_GUIDE` 1–4 replace the learned lobe with fixed world-up lobes for sampler validation; 5 and 6 show
the learned lobe with the larger share and the coverage through Debug Output.

Limitations: two lobes per face still cannot represent more light directions or parallax inside a face, and pixels whose
face has no RC slot are not guided. A single lobe and a lobe re-aimed at the face's mean hit point were evaluated and
removed: the dual lobe matched or beat the single lobe in every scene and the hit-point lobe in all but one, at lower
cost than the latter. With screen-ray guiding only (+1.7% `composite_total`) the 2026-09-30 equal-time error reduction was 21% in
`flatroom-hard-lighting-720p`, 13% in `gnlxc-normal-1`, 11% in `night-gi-1`, and within ±3% in the other scenes; only
flatroom passes the 20% hard-scene gate, so the setting stays off by default.

## Settings

GI settings are organized in the GI and denoiser screens of [
`scripts/options.main.kts`](../../../scripts/options.main.kts):

- Trace: `SETTING_GI_INITIAL_SST_STEPS`, `SETTING_GI_VALIDATE_SST_STEPS`, `SETTING_GI_SST_THICKNESS`.
- Probe/sky: `SETTING_GI_PROBE_FADE_START/END`, `SETTING_GI_MC_SKYLIGHT_ATTENUATION`.
- Reuse: `SETTING_GI_TEMPORAL_REUSE_LIMIT`, `SETTING_GI_SPATIAL_REUSE`, `SETTING_GI_SPATIAL_REUSE_COUNT`,
  `SETTING_GI_DECORRELATE`.
- Guiding: `SETTING_GI_PATH_GUIDING`.
- Debug screen: `SETTING_GI_USE_REFERENCE`, `SETTING_DEBUG_PATH_GUIDE`, `SETTING_DEBUG_RC_MODE`.
- Denoiser: spatial enable/sample counts, history lengths, fast-history clamping, flicker suppression, anti-firefly, and
  history-fix weights.

Profiles mainly scale SST steps, spatial reuse count, and denoiser sample counts. Any new `SETTING_*` must be registered
in the options DSL before GLSL or program conditions use it.

## Maintenance invariants

- Change reservoir packing, MIS metadata, and all producers/consumers together.
- Temporal tiles must agree with current/previous jitter, camera transforms, and G-buffer semantics.
- Edge classification/dilation stays before reprojection and accumulation.
- A spatial batch-size change must update program thresholds, base indices, and the indirect queue layout together.
- The ratio-resolve producer in the paired passes, its read in spatial shading, and the trace-pass re-blend share the
  `transient_ssgiSpecOut` and provisional-record layouts and change together.
- Validate convergence, camera motion, disocclusion, screen edges, and history reset after setting changes.
