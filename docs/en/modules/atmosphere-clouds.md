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
| 4     | [`clouds/amblut/Sample`](../../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl)           | Alternates cumulus and cirrus heights; samples 1024 full-sphere atmospheric rays and stores direct ground reflection separately |
| 5     | [`GenerateSkyViewLUT`](../../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl)       | Writes `uimg_skyViewLUT`                           |
| 6     | [`clouds/amblut/Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl)           | Writes the 32×33 `persistent_cloudsAmbLUT`: a 32×32 directional cirrus tile and 25 cumulus values in the last row (upper-sky integral and 24 incident directions); reduces all subgroups and resets inactive history on transitions |

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
visible interval with nonuniform distance bins, using 128–256 samples by default. When an empty sample is followed by
an occupied sample, four bisections locate the interface and move lighting evaluation within one nominal mean free
path of it. The occupied segment length enters the analytic constant-medium Beer-Lambert integral. This surface
sampling is a biased approximation that preserves illuminated billows at finite step counts. Ground intersections clip the march;
the cloud interval is capped at 60 km, with a maximum distance of 120 km from the camera.
When residual transmittance falls below `0.005`, the march completes the opaque tail with the local source and
sets transmission to zero. This avoids bright-sky leakage and termination bands inside dense clouds.

Each occupied sample evaluates single scattering separately from the cached scattered field. Sun and moon each use
a 16-sample column through the actual density, atmospheric transmittance, Earth occlusion, and the full configured
phase function. Two phase-importance samples estimate sky/ground single scattering with 12-sample density columns;
their PDF is an equal mixture of the three AP0 phase densities. The background environment comes from the directional
ambient LUT. The old extinction octaves, decaying scattering-order weights, and `0.15` ambient optical-depth scale
have been removed. Camera integration always uses extinction, never the transport coefficient or a fitted MS scale.

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
| `persistent_cloudsAmbLUT` | Existing 32×33 atlas: cirrus remains 32×32; last-row x=0 stores the upper-sky integral divided by 4π, x=1…24 store average incident radiance per angular cell |
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
The finite grid uses a spatially uniform sky/ground environment and omits cloud-shadowed ground feedback. A fixed
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

### Local transport and sparse residuals

[`LocalTransport.glsl`](../../../shaders/techniques/atmospherics/clouds/LocalTransport.glsl) uses the coarse solution as
a far-field predictor. Along a selected incoming propagation direction, it traces backward through actual density and
integrates `S_local = T S_boundary + integral(T sigma_s P(B_h + S_c))`. Each segment uses the analytic constant-source
integral, including scattering refill. Extinction remains `sigma_t`; empty segments leave transmission unchanged.
This preserves a conservative constant isotropic equilibrium, unlike an attenuation-only correction.

| Stage in [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | Operation |
|---|---|
| Camera march | Integrates the inexpensive `P S_c` predictor at every occupied sample; streams independent depth reservoirs with the maximum absolute RGB camera scattering weight, including foreground air and the opaque tail |
| Direction selection | Samples the RGB mixture of the same interpolated discrete phase-matrix row used by the predictor, with explicit phase/proposal weights; continuous phase sampling would not match this cached convolution |
| Local reconstruction | Traces one upstream grid-cell flight length, `1 / sum(abs(omega) / cell_size)`, in exponential distance bins; samples fine density and cached `P S_c`; reevaluates sun/moon columns with 16 density samples and one phase-weighted sky/ground direction with 12, reusing that direction along the local ray |
| Correction | Adds the signed `S_local - S_c` with both depth and direction PDF factors; independent reservoirs are averaged, without changing camera transmission or adding cached radiance twice |

`SETTING_CLOUDS_MS_LOCAL_SAMPLES` selects 0, 1, 2 (default), or 4 reservoirs per camera ray; zero exposes the predictor
for comparison. `SETTING_CLOUDS_MS_LOCAL_STEPS` selects 4, 8 (default), or 16 local segments. These budgets replace the
removed 96³ grid path; they add no screen image, volume allocation, or extra pipeline pass. Direct-beam injection into
local directions retains the integrated angular-cell phase; the camera's direct single scattering retains the sharp
continuous phase. The local sky proposal is independently normalized and uses high-resolution density columns.

Before storage and filtering, the residual estimator targets the selected discrete angular, frozen-source local
model in expectation. It does not make the full transport unbiased: endpoints and secondary sources still use the
coarse cache, local density/shadow quadrature and finite radius leave errors, and optically thick unresolved structures
may require more than one local update. Camera-only residuals do not feed net flux back into the coarse solve.
Signed FP16 storage retains negative estimates; range saturation, spatial reconstruction and temporal variance
clipping remain biased. Existing local reconstruction footprints are retained, without a new broad residual blur.

[`cloud-local-transport-check.py`](../../../scripts/test/cloud-local-transport-check.py) checks noisy-density
equilibrium, thin second-order scaling, exact predictor/PDF consistency, spectral streaming-reservoir expectation and
variance, and two equal-coarse-density arrangements of a narrow illuminated channel. Its local-step convergence
reference integrates the continuous frozen-source ODE; it is not a full cloud transport reference.
Its `--gpu-history PATH` mode checks the frame-end RGBA32UI atlas dump and adjacent metadata for finite signed
scattering, transmission bounds, history length and depth. Only the persistent history tile remains valid at frame end.

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
| Low cloud               | Upscale factor; history length/confidence/variance; minimum/maximum steps; MS grid resolution/iterations and local samples/steps; height/thickness/density/coverage/phase; wind; and shape frequencies |
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
