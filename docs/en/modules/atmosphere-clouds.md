# Atmosphere, Sky, and Clouds

Language: English | [简体中文](../../sc/modules/atmosphere-clouds.md)

Alpha Piscium separates low-frequency LUTs, view-dependent sky-view/epipolar data, volumetric clouds, and final local
composition. Atmosphere and cloud-source preparation run in `begin`, cloud transport runs in `deferred`, clouds render early in composite, and air/water volumes resolve
after GI but before translucency.

## Code map

| Path                                                                                                                                                                                                                                         | Responsibility                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| [`shaders/techniques/atmospherics/air/lut/`](../../../shaders/techniques/atmospherics/air/lut/)                                                                                                                                              | Atmosphere LUT generation and API                    |
| [`atmospherics/air/Raymarching*.glsl`](../../../shaders/techniques/atmospherics/air/)                                                                                                                                                        | Air-medium ray marching                              |
| [`SliceEndPoints.comp.glsl`](../../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl)                                                                                                                                          | Epipolar slice endpoints                             |
| [`EpipolarScattering.comp.glsl`](../../../shaders/techniques/atmospherics/air/EpipolarScattering.comp.glsl)                                                                                                                                  | Air epipolar-scattering kernel                       |
| [`Cumulus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cumulus.glsl), [`Cirrus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cirrus.glsl), [`Mediums.glsl`](../../../shaders/techniques/atmospherics/clouds/Mediums.glsl) | Cloud density, media, and phase functions            |
| [`RenderVolumetric.comp.glsl`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl)                                                                                                                                   | Main low-cloud render                                |
| [`Accum.comp.glsl`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)                                                                                                                                                      | Temporal/spatial cloud accumulation and upscale      |
| [`atmospherics/clouds/amblut/`](../../../shaders/techniques/atmospherics/clouds/amblut/)                                                                                                                                                     | Cloud ambient-light LUT sampling and gathering       |
| [`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl), [`LocalComposite.glsl`](../../../shaders/techniques/atmospherics/LocalComposite.glsl)                                                                     | Shared sky and local-volume composition              |
| [`shaders/util/Celestial.glsl`](../../../shaders/util/Celestial.glsl)                                                                                                                                                                        | Project sun, moon, star-map, and constellation logic |

## Begin: LUT and frame preparation

The current program-list order is:

| Order | Pass                                                                                                        | Purpose                                            |
|-------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| 1     | [`GenerateTransmittance`](../../../shaders/techniques/atmospherics/air/lut/GenerateTransmittance.comp.glsl) | Updates the 256×64 `persistent_transmittanceLUT`   |
| 2     | [`SliceEndPoints`](../../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl)                   | Updates epipolar geometry for the frame            |
| 3     | [`GenerateMultiSctr`](../../../shaders/techniques/atmospherics/air/lut/GenerateMultiSctr.comp.glsl)         | Updates the 32×32 `persistent_multiSctrLUT`        |
| 4     | [`clouds/amblut/Sample`](../../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl)           | Alternates cumulus and cirrus heights; samples 1024 full-sphere atmospheric rays and stores direct ground reflection separately for cirrus |
| 5     | [`GenerateSkyViewLUT`](../../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl)       | Writes `uimg_skyViewLUT`                           |
| 6     | [`clouds/amblut/Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl)           | Writes the 32×33 `persistent_cloudsAmbLUT`: a 32×32 directional cirrus tile and 25 cumulus values in the last row (cosine-weighted sky irradiance and 24 clear-air directions); reduces all subgroups and resets inactive history on transitions |

Fixed LUT dimensions are declared in [`shaders/shadesmith.json`](../../../shaders/shadesmith.json);
screen-sized/configurable images are declared in [`scripts/shaders.properties`](../../../scripts/shaders.properties).

Both `uimg_skyViewLUT` screen dimensions use `SETTING_SKYVIEW_RES`; its six slices hold scattering/transmittance pairs
for the full atmosphere and the regions below/above cirrus. `uimg_epipolarData` uses `SETTING_EPIPOLAR_SLICES` for its
width, and maps `SETTING_SLICE_SAMPLES` to a height of 385, 769, 1537, or 3073. Adding a supported sample count requires
updating both the property-size mapping and shader indexing logic.

## Cloud passes

When `SETTING_CLOUDS_CU` is enabled:

| Order | Pass                                                                                             | Purpose                                                                                                                                                                                                      |
|-------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | Ray marches convective towers with high-resolution single scattering, a cached MS predictor, and sparse local transport residuals |
| 2     | [`clouds/ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)          | Applies history accumulation, confidence/variance processing, and upscaling to the low-resolution result; this is the [`ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl) stage |
| 3     | Later air/local composition                                                                      | Reads the accumulated cloud scattering and transmittance                                                                                                                                                     |

The main screen tiles are `transient_lowCloudRender`, `transient_lowCloudAccumulated`, and `history_lowCloud` (
RGBA32UI). The first three words pack signed FP16 scattering and transmission; the last packs FP16 history length and
opacity-weighted depth in atmosphere kilometres. The hand-maintained property fragment declares custom cloud phase-LUT, cirrus, cumulus detail, and curl
textures. High cirrus is sampled by the shared sky/cloud path rather than a separate compute program.

### Cumulus congestus

[`Cumulus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cumulus.glsl) joins a broad, shallow condensation base
to three thermals of different heights. Each thermal combines an elongated body and a round crown. Cell seeds vary
height, lean, and side branches; coverage controls footprint width independently of height. Neighbor bounds cover
the base and leaning thermals. The normalized envelope keeps erosion proportional to the cell radius.

A four-octave Worley texture shapes the large billows, with vertical stretching and curl displacement, followed by
two finer erosion scales. Noise perturbs the condensation boundary over a finite density transition; liquid-water
density rises above the base. The default layer starts at 1 km and is 3.5 km deep. The 128³ R8 billow texture uses
the Strepitus recipe in [`cumulus_detail1.json`](../../../data/strepitus/cumulus_detail1.json); the second detail
texture supplies fine erosion.

[`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) uses the spherical-shell
intersection in [`clouds/Common.glsl`](../../../shaders/techniques/atmospherics/clouds/Common.glsl). The main march covers the entire
visible interval with nonuniform scouting bins, using 128–256 samples by default. Eight bisections locate an
empty-to-occupied interface. Occupied bins are then subdivided so each camera segment has at most 0.5 nominal optical
depth, using the density upper bound. This replaces evaluation at one mean free path inside a long opaque segment.
The constant-medium Beer-Lambert integral uses each actual segment length; already integrated intervals are never
counted twice. Density quadrature and empty-space scouting remain finite-resolution approximations. Ground
intersections clip the march; the cloud interval is capped at 60 km, with a maximum camera distance of 120 km.
When residual transmittance falls below `0.005`, the march completes the opaque tail with the local source and sets
transmission to zero.

Each occupied sample integrates the inexpensive cached multiple-scattering predictor and foreground air. Expensive
single scattering is evaluated at opacity-weighted reservoir depths, using exactly the same spectral camera weights.
There are two direct-light samples by default, or four when the local sample budget is four. Sun and moon use
actual density, atmospheric transmittance, Earth occlusion, and the full phase function at the sampled solar/lunar
disc direction.
Two continuous phase-importance directions estimate sky/ground single scattering at each selected depth, with
an equal mixture of the three AP0 phase densities as the PDF. Fine shadow rays use ratio tracking for the first eight
majorant optical depths, then a 16-sample exponential column for the distant tail. This resolves near-field occluders
without exponentiating a sparsely sampled density integral; distant quadrature remains approximate.
Near-field transmission below `1e-6` uses Russian roulette with survival compensation instead of tracing a negligible
opaque tail. This changes variance without discarding its expected contribution.

Cumulus ambient directions store clear-air radiance separately from ground reflection. At a fine scattering point,
[`Transport.glsl`](../../../shaders/techniques/atmospherics/clouds/Transport.glsl) intersects the incident ray with the
planet and evaluates sun/moon reflection there, including the actual cloud column from ground to light and the
atmospheric return transmission. This replaces the uniform, unshadowed direct ground contribution. The coarse
boundary evaluates this at the camera's horizontal position and layer midpoint. Diffuse ground illumination still
uses a clear-sky approximation, with cosine-weighted `E_sky / pi` replacing twice an unweighted hemisphere average;
coupled cloud/ground multiple scattering is not solved.

Shadow-column exponential bins are anchored to a quarter of the minimum extinction mean free path, so a distant
layer exit no longer stretches the first shadow segment across small surface details. Columns cover up to 12 km
and stop after becoming optically opaque (minimum channel depth 12). Distant shadow quadrature is still coarse.
The old extinction octaves, decaying scattering-order weights, and `0.15` ambient optical-depth scale remain removed.
Camera and local light paths always use extinction, never transport extinction or an MS brightness scale.

Air scattering and transmission are integrated incrementally to occupied cloud samples with 4–32 samples per segment.
Their clear-air radiance is weighted by the cloud opacity at that distance. [`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl)
then composites the result over the sky; it no longer places cumulus at a fixed middle-altitude plane.
When viewed from above cirrus, its foreground scattering and transmission are applied to the cumulus contribution.
Cloud history reprojects at its opacity-weighted depth with horizontal wind advection and the atmosphere altitude
scale. Depth disagreement reduces history confidence; world-time changes use the existing light-history reset, and
the wind timer wrap rejects history. A single depth still approximates multiple overlapping cloud layers, and vertical
detail motion is not reprojected. Wind translates the field and its detail together; screenshot mode freezes it. A zero coverage, density, or
layer thickness gives an empty cloud result. Cirrus retains its separate lighting model and multiple-scattering-radius
control.

### Shared directional transport and two-channel correction

[`Transport.glsl`](../../../shaders/techniques/atmospherics/clouds/Transport.glsl) stores the cloud-scattered radiance
`S = L - B` in 24 equal-solid-angle cells. Directions are the signed permutations of `(2,1,1)/sqrt(6)`.
The target equation is `A S = sigma_s P B`, with finite-volume upwind streaming and zero incoming scattered radiance.
Empty cells have exactly zero extinction and retain free streaming between clouds. The cached camera source is only
`P S`; sampling it still requires scattering by the actual high-resolution density. Neither `S` nor the uncollided
sky contribution is added a second time. Positive interpolation between four angular directions preserves a constant
field and equal integrated basis weights by cubic symmetry; spatial interpolation uses physical cell centers.

| Order | Pass / stage | Purpose |
|---|---|---|
| 1, begin | [`CloudTransportPrepare`](../../../shaders/pass/begin/CloudTransportPrepare.comp.glsl) | Builds camera-centered, horizontally stretched grid edges, the phase matrix and integrated sun/moon injection; four density samples per cell retain the sampled correlation between density and high-resolution direct transmittance; builds the coefficient hierarchy |
| 2, begin | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl), [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Sweeps uncollided sky/ground boundary light through the grid, forms its scattering source, and initializes `S` to zero |
| 3, deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl) | Performs a directional source iteration with exact triangular wavefront solves of the selected upwind operator |
| 4, deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Computes the true directional residual and restricts its scalar part into two coupled diffusion channels; a multigrid V-cycle uses volume-integrated residuals, a 4³ block-preconditioned CG solve, physical-center prolongation and paired red/black smoothing |
| 5, deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl), [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Lifts the scalar correction through directional transport, minimizes the actual residual along it subject to nonnegative radiance, then updates `S`; repeats stages 3–5 for the selected iteration budget |
| 6, deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Records the final residual and injected/escaped/absorbed power, then preconvolves `P S` for the camera |

The two-channel core uses `a_l = sigma_t - sigma_s beta_l`, `D0 = 1/(3 a1)`, positive slow/fast diffusion roots,
weights `w, 1-w`, and paired exchange `kappa = w(1-w) sigma_s (1-beta2)`. Harmonic shared-face conductances and
separate Marshak escape boundaries preserve the conservative symmetric core. A fitted effective `a3` changes the
intermediate spatial response; it does not change absorption or `D0`. This is the scalar homogeneous P3 response
represented by two channels, not an exact heterogeneous angular P3 solver. In the renderer it preconditions the
directional equation rather than replacing that equation or providing an independent light contribution.

[`cloud-transport.py`](../../../scripts/cloud-transport.py) integrates the actual filtered cumulus Mie LUT and the
[HG+Draine phase function](https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf).
It normalizes the Mie phase before both direct evaluation and moment extraction, constructs a positive symmetric
stochastic angular matrix by cell integration, and fits `a3` against a P255 continued fraction over the grid's
Nyquist band. The LUT stores the two inverse CDFs, two phase matrices, phase moments/normalization, and closure fits
over phase mixtures and transport cell sizes. The existing nominal HG+Draine diameter is 11.54 µm; the independent
20 µm example fit is not copied into this renderer. This use of a lower-order correction follows
[synthetic acceleration](https://arxiv.org/html/2402.10488v2); the particular GPU discretization and finite budget
remain an approximation.

| Resource | Layout and lifetime |
|---|---|
| [`cloud_transport.bin`](../../../shaders/textures/cloud_transport.bin), `usam_cloudTransport` | Static 1024×26 RGBA32F phase/closure asset, 416 KiB |
| SSBO binding 6 | FP32 AP0 directional fields, source and correction, coupled coefficients/errors/RHS at every level, reductions and grid edges; 56,240,912 bytes at 32³ or 449,858,832 bytes at 64³; recreated from current density and lighting each frame |
| `persistent_cloudsAmbLUT` | Existing 32×33 atlas: cirrus remains 32×32; last-row x=0 stores cosine-weighted sky irradiance divided by π, x=1…24 store average clear-air radiance per angular cell |
| `ctStats` in SSBO 6 | Four raw/corrected residuals and accepted step sizes, final relative residual, injected and escaped-plus-absorbed power, and coarse CG residual/iteration count; inspect with a buffer dump |

The SSBO uses one unsized `vec4[]` with explicit offsets; large sized SSBO arrays cause excessive NVIDIA compile
times. Its byte allocation and the buffer-dump reader must change together if this packed layout changes.

The grid spans ±121 km horizontally. Each horizontal axis assigns 75% of its cells uniformly to ±12 km;
a continuously differentiable exponential skirt covers the remaining distance. The inverse sampling map and
physical-center interpolation use the same allocation. The vertical extent covers the spherical cloud layer down to
2.5 km below its nominal base. `SETTING_CLOUDS_MS_RESOLUTION` selects 32³ (default) or 64³;
the inner horizontal spacing is 1 km or 500 m respectively. `SETTING_CLOUDS_MS_ITERATIONS` selects 1, 2 or 4
(default). The field is solved afresh each frame; temporal cloud
accumulation is only used after camera integration, so wind/camera/light changes do not reuse a stale transport solve.

This implementation does not establish angular or spatial convergence of rendered clouds. The 24-direction
representation softens low-order angular detail, coarse upwind cells add numerical diffusion (their thick-cell
diffusion limit differs from physical `D0`), and four subcell source samples can miss narrow illuminated features.
The finite grid uses a spatially uniform environment; direct ground light includes cloud shadows at a representative
position, while local paths evaluate their own ground intersections. Cloud/ground multiple-scattering feedback and
cloud shadows in the clear-air environment remain omitted. A fixed
iteration budget leaves a measured residual and power defect; conservative core coefficients alone do not prove
zero error in the whole interpolated camera image. Increasing the grid size is expensive and is an accuracy control,
not a guarantee that these errors have converged.

Regenerate the asset and run the independent sparse-matrix reference checks from [`scripts/`](../../../scripts/):

```sh
python cloud-transport.py --output ../builds/cloud-transport/phase.json
python test/cloud-transport-check.py --output ../builds/cloud-transport/numerics.json
```

[`cloud-transport-check.py`](../../../scripts/test/cloud-transport-check.py) checks the P3 identity, converged
finite-volume positivity and power balance, thin-density scaling, vacuum-gap streaming, superposition, and the actual
upwind diffusion limit. Its `--gpu-buffer PATH --resolution 32` mode verifies an Iris SSBO dump and reports the actual
fixed-budget residual and power defect. These checks complement target-Iris compilation, buffer capture, images and
pass timings; they do not substitute for image comparison with reference transport or photographs.

### Local scattering paths and sparse residuals

[`LocalTransport.glsl`](../../../shaders/techniques/atmospherics/clouds/LocalTransport.glsl) follows multiple scattering
events through the actual density before consulting the far-field cache. This replaces the single frozen-source
update and its fixed exponential local bins. The coarse two-channel solver and its 24-direction transport fields
retain their spatial operators; local paths use continuous Mie/HG+Draine directions and continuous direct-beam phase evaluation.

| Stage in [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | Operation |
|---|---|
| Camera march | Integrates cached `P S_c`; streams independent depth reservoirs using the maximum absolute RGB scattering weight, including foreground air and the opaque tail; stores the exact camera predictor at each selected depth |
| Direction selection | Mixes an RGB phase proposal with a visible sun/moon phase proposal and uses the full PDF; the beam probability starts at 1/16 and falls as `1 / (16 * event)` during continuation, limiting path-weight variance |
| Local flight | Uses spectral null tracking through true density, over one upstream cell flight length `1 / sum(abs(omega) / cell_size)`; ground clips the flight; escape queries positively interpolated scattered radiance `S_c` at the endpoint |
| Scattering event | Accumulates a cached uncollided-source predictor and streams two independent source reservoirs; samples another continuous direction; at the depth budget, closes with directional `S_c` after the last phase convolution |
| Source reconstruction | Re-evaluates sun/moon and sky/ground light through fine density at the two selected path events, subtracts their source predictors, and applies the event PDF weights |
| Correction | Adds the phase-weighted local estimate minus the stored camera predictor, with the depth PDF correction; retains signed residuals and the original camera transmission |

The path-tracking majorant is `1.5 * max(sigma_t) * CU_DENSITY_MAX`, where the density function is bounded by
`CU_DENSITY_MAX = 1.35 * CU_DENSITY_SCALE`. Set `sigma_ref = mean(sigma_t)` over AP0 channels. Real-event probability
is `sigma_ref * density / majorant`; real events multiply throughput by `sigma_s / sigma_ref`, and null events by
`(majorant - density * sigma_t) / (majorant - density * sigma_ref)`. Padding the proposal keeps null weights positive
and avoids the rare, large spectral weights produced by a tight majorant with least-extinct real events. It changes
sampling variance, not extinction or absorption. A changed density maximum must update this bound. Empty regions transmit freely; the method includes
scattering refill and preserves conservative isotropic equilibrium. The algorithm uses
[null-scattering transport](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators).

`SETTING_CLOUDS_MS_LOCAL_SAMPLES` selects 0, 1, 2 (default), or 4 local paths per camera ray; zero disables the MS
correction while retaining sampled direct lighting. `SETTING_CLOUDS_MS_LOCAL_BOUNCES` selects 1, 4, 8 (default), 16, or 32
scattering events. It replaces the removed local-segment setting. No new screen image, volume allocation, or pass is
needed. The coarse SSBO remains an unsized `vec4[]`; no large sized arrays are introduced.

Along each sampled local path, `ctSky` and the density-correlated beam cache predict the uncollided source. Each event's
proposal is `max(throughput * (prediction + (E_sun + E_moon) / (4*pi)))`. The positive isotropic term keeps missed coarse
cells sampleable; it only changes the proposal and is never added as light. Two independent reservoirs correct the
sum of source predictions, so fine shadow work does not multiply by the whole scattering depth. Conditional on the
sampled path, the entire source predictor cancels in expectation, including after early flight escape.

The entire stored camera predictor is subtracted, rather than a sampled discrete-direction cache value. Therefore the
coarse angular matrix and continuous local phase need not agree: the original predictor cancels in expectation under
the same depth weights. Both single scattering and the signed MS correction use those weights; neither cached `S_c`
nor uncollided light is added twice. Extra local events resolve scattering between small structures instead of merely
attenuating a smooth cache value.

Null tracking removes fixed-step local density quadrature, but the finite flight radius, terminal cache, phase LUT,
shadow columns, and camera quadrature still leave approximation error. Local paths do not feed net flux back into the
coarse solve. More events reduce early cache substitution without proving full transport convergence. Sparse paths
remain noisy, particularly for sharp forward scattering; signed FP16 range saturation, spatial reconstruction and
temporal variance clipping remain biased. The existing temporal filter is retained without an added broad blur;
default history length is 32 frames to stabilize the sparse corrections. Reprojection and disocclusion checks still
control whether history is valid.

[`cloud-local-transport-check.py`](../../../scripts/test/cloud-local-transport-check.py) checks spectral null tracking
against an analytic nonuniform slab, exact gray equilibrium and vacuum transmission, the stored-predictor cancellation
with different coarse/fine angular operators, spectral WRS expectation/variance, and the RGB beam-mixture PDF against
analytic HG convolution. It also checks local scattering-order convergence against an analytic two-stream slab,
optically thin camera integration, and equal-coarse-density narrow channels against a continuous frozen-source ODE.
Additional checks cover the two path-source reservoirs, fixed-order spectral weight variance, near-field shadow tracking against a Gaussian optical
depth integral, Lambertian sky normalization, and attenuation along both legs of a direct ground bounce.
These are model tests, not a full cloud image or global power proof. Its `--gpu-history PATH` mode checks finite signed
scattering, transmission bounds, history length and depth in the frame-end RGBA32UI atlas; only its persistent history
tile still contains valid cloud data at frame end.

```sh
python test/cloud-local-transport-check.py --output ../builds/cloud-transport/local-numerics.json
```

## Air, depth layers, and composition

| Stage               | Pass/code                                                                                                                                                  | Purpose                                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| GI phase            | [`GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl) | Builds the depth layers required by volumetrics                                                                  |
| Air volume          | [`EpipolarScatteringAir`](../../../shaders/pass/composite/EpipolarScatteringAir.comp.glsl)                                                                 | After GI denoising, consumes endpoints, epipolar data, LUTs, shadows, and depth layers to produce air scattering |
| Water/translucency  | Shared code such as [`LocalComposite.glsl`](../../../shaders/techniques/atmospherics/LocalComposite.glsl)                                                  | Merges sky, clouds, and local media for water scattering and translucent back composite/SST/composite            |
| Optional correction | [`VolumetricLocalCompositeBreakFix`](../../../shaders/pass/composite/VolumetricLocalCompositeBreakFix.comp.glsl)                                           | Corrects depth breaks at the end                                                                                 |

## Sky and celestial rendering

Vanilla clouds, sun, moon, sky, and stars are disabled in the project property fragment. Alpha Piscium renders them
through its own sky composite and [`Celestial.glsl`](../../../shaders/util/Celestial.glsl), conditionally binding
`usam_starmap` only when star intensity is nonzero and `usam_constellations` only when its setting is enabled.

## Settings

| Category                | Settings                                                                                                                                        |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Atmosphere scale/ground | Altitude, density scale, and ground albedo                                                                                                      |
| Air                     | Epipolar slices/samples; Mie turbidity/time curve; Mie/Rayleigh/ozone multipliers                                                               |
| Sky/light shafts        | Sky-view resolution, sky samples, shaft samples/shadow samples, depth-break correction, and softness                                            |
| Low cloud               | Upscale factor; history length/confidence/variance; minimum/maximum steps; MS grid resolution/iterations and local samples/scattering depth; height/thickness/density/coverage/phase; wind; and shape frequencies |
| High cloud              | Cirrus height, density, coverage, and phase                                                                                                     |
| Celestial               | Sun/moon radius, distance, temperature/color/albedo; star-map intensity/gamma/bright-star boost; and constellations                             |

All settings are declared in [`scripts/options.main.kts`](../../../scripts/options.main.kts); profiles mainly scale
epipolar resolution, cloud upscale, history length, and march steps.

## Maintenance invariants

- Change LUT coordinate conventions and every API consumer together.
- Keep epipolar counts within the declared size mapping.
- Keep cloud-history render resolution, upscale factor, jitter, and confidence logic aligned.
- Change binary assets, `customTexture.*` declarations, and samplers together.
- Test the full day cycle, weather, water transitions, fast turns, both sides of cloud layers, and history resets.
