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

Each occupied scouting bin evaluates the cached predictor and foreground air at its endpoints. Thin optical
segments interpolate these low-frequency quantities while sampling full-detail density. Expensive
single scattering is evaluated at opacity-weighted reservoir depths, using exactly the same spectral camera weights.
There are two direct-light samples by default, or four when the local sample budget is four. Sun and moon use
actual density, atmospheric transmittance, Earth occlusion, and the full phase function at the sampled solar/lunar
disc direction.
Two continuous phase-importance directions estimate sky/ground single scattering at each selected depth, with
an equal mixture of the three AP0 phase densities as the PDF. Fine shadow rays integrate 16 exponential density bins
with true extinction. This removes ratio-tracking noise and unbounded null-event work, but finite quadrature can
still miss distant narrow occluders. HG+Draine is the default (`SETTING_CLOUDS_CU_PHASE_RATIO = 0`); direct evaluation,
sampling, phase matrices and moment coefficients all use the same mixture setting.

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

Air scattering and transmission are integrated to scouting-bin endpoints with 4–32 samples per segment.
Interpolation avoids repeating the air march in every thin optical segment; cloud opacity still weights the
foreground radiance at each depth. [`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl)
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
The target equation is `A S = sigma_s P B`, with conservative characteristic streaming and zero incoming scattered radiance.
Empty cells have exactly zero extinction and retain free streaming between clouds. The cached camera source is only
`P S`; sampling it still requires scattering by the actual high-resolution density. Neither `S` nor the uncollided
sky contribution is added a second time. Positive interpolation between four angular directions preserves a constant
field and equal integrated basis weights by cubic symmetry; spatial interpolation uses physical cell centers.

| Order | Pass / stage | Purpose |
|---|---|---|
| 1, begin | [`CloudTransportPrepare`](../../../shaders/pass/begin/CloudTransportPrepare.comp.glsl) | Builds the grid, phase matrix and coefficient hierarchy; four volume quadrature points define beam chords with eight density taps each, integrating `rho B` from entry transmission and `1-exp(-tau)` |
| 2, begin | [`CloudTransportPrepare`](../../../shaders/pass/begin/CloudTransportPrepare.comp.glsl), `CT_HISTORY` | Reprojects previous `P S` using camera displacement and cloud advection into a separate scratch field; invalid history supplies zero |
| 3, begin | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl), [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Computes uncollided sky/ground light, then initializes the current source from `P B` and the reprojected scattered-light guess |
| 4, deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl) | Solves directional characteristics using a descending shared-memory plane, at most 48 KiB per group |
| 5, deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Restricts the true directional residual into two coupled diffusion channels; the V-cycle uses 4³ block-preconditioned CG, physical-center prolongation and red/black smoothing |
| 6, deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl), [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Lifts and damps the correction while preserving nonnegative radiance; later iterations execute only while residual or power-defect tolerance is unmet |
| 7, deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Records actual residual/power, stores cell-mean scattered radiance and its angular mean, and preconvolves `P S` for the camera and next-frame initialization |

Characteristics distinguish incoming light, outgoing face values and cell averages. With
`tau = sigma_t / sum(abs(omega)/cell_size)`, uncollided face transmission is `exp(-tau)`.
The average is `(1-w) L_out + w L_in`, where `w = 1/tau - 1/(exp(tau)-1)` uses a stable thin-limit series.
Scattering and absorption use that average; shared-face streaming and escape use outgoing values. Residual,
correction and power operators use the same distinction. This replaces transmitting the cell average to the next
cell, which gave thick-cell uncollided transmission `1/(1+tau)`. Finite transverse quadrature and constant sources
within each segment remain spatial approximations.

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
| SSBO binding 6 | FP32 AP0 directional fields, source and correction, coupled coefficients/errors/RHS at every level, reductions and grid edges; 56,240,912 bytes at 32³ or 449,858,832 bytes at 64³; current medium/source with the reprojected previous solution as initial guess |
| `persistent_cloudsAmbLUT` | Existing 32×33 atlas: cirrus remains 32×32; last-row x=0 stores cosine-weighted sky irradiance divided by π, x=1…24 store average clear-air radiance per angular cell |
| `ctStats` in SSBO 6 | Executed residuals/steps, final residual and power, corrected power defects, active iteration count, history validity/displacement, and previous camera/time/light metadata; inspect with a buffer dump |

The SSBO uses one unsized `vec4[]` with explicit offsets; large sized SSBO arrays cause excessive NVIDIA compile
times. Its byte allocation and the buffer-dump reader must change together if this packed layout changes. At frame end,
`ctRadiance` holds outgoing values and `ctLifted` holds cell means. Level-zero fast/weight/exchange coefficient alpha
lanes store the RGB angular mean; these lanes do not participate in the solve.

The grid spans ±121 km horizontally. Each horizontal axis assigns 75% of its cells uniformly to ±12 km;
a continuously differentiable exponential skirt covers the remaining distance. The inverse sampling map and
physical-center interpolation use the same allocation. The vertical extent covers the spherical cloud layer down to
2.5 km below its nominal base. `SETTING_CLOUDS_MS_RESOLUTION` selects 32³ (default) or 64³;
the inner horizontal spacing is 1 km or 500 m respectively. `SETTING_CLOUDS_MS_ITERATIONS` selects 1, 2 or 4
(default), as an upper iteration limit. Previous `P S` is only an initial guess for the current operator and source.
Frame discontinuities, history resets, or sun/moon changes larger than their angular radius reject that guess.
At least one full iteration executes; subsequent iterations stop when both predicted relative residual and corrected
power defect are below `0.005`. Final diagnostics recompute actual residual and power. Cold starts and abrupt changes
can still exhaust the budget before convergence.

This implementation does not establish angular or spatial convergence of rendered clouds. The 24-direction
representation softens low-order angular detail, coarse cells retain spatial error (their thick-cell diffusion limit differs from physical `D0`), and finite
beam-chord quadrature can miss narrow illuminated features.
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
characteristic diffusion limit. Its `--gpu-buffer PATH --resolution 32` mode verifies an Iris SSBO dump and reports the actual
fixed-budget residual and power defect. These checks complement target-Iris compilation, buffer capture, images and
pass timings; they do not substitute for image comparison with reference transport or photographs.

### Local scattering paths and sparse residuals

[`LocalTransport.glsl`](../../../shaders/techniques/atmospherics/clouds/LocalTransport.glsl) reconstructs explicit
low-order scattering through true density and uses a local boundary approximation for the terminal high-order field.
The coarse two-channel solver remains the nonlocal predictor; no larger light volume or new screen allocation is used.

| Stage | Operation |
|---|---|
| Camera integration | Integrates the complete cached predictor and maintains spectral opacity-weighted depth reservoirs, retaining the predictor at each selected point |
| Local flight | Divides three upstream cell lengths into 16 exponential bins; samples the integrated segment source using mean AP0 extinction and spectral opacity-ratio weights; vacuum transmits freely and ground clips escape |
| Local sources | Accumulates nearest-cell sky/beam predictions; separate reservoirs select one sky event and one sun/moon event, with the full continuous beam phase in its proposal; fine shadows correct both predictions |
| Terminal boundary | Locates the true interface and reconstructs high-order angular light from interior mean and exterior incident scattered light; deep interiors and optically thin details use the directional cache |
| Camera correction | Adds phase-weighted local light minus the saved camera predictor, retaining signed residuals and true camera extinction |

`SETTING_CLOUDS_MS_LOCAL_SAMPLES` selects 0, 1, 2 (default), or 4 paths; zero disables local MS correction but retains
direct light. `SETTING_CLOUDS_MS_LOCAL_BOUNCES` selects 1, 2 (default), 4, 8, 16, or 32 explicit events. More events delay
terminal closure but do not guarantee a better finite-sample image.

For segment column `c`, selection probability is `T_ref * (1-exp(-sigma_ref*c))`, with `sigma_ref = mean(sigma_t)`.
The spectral numerator is `T * (sigma_s/sigma_t) * (1-exp(-sigma_t*c))`. Their ratio integrates the selected
piecewise-constant model without null collisions, at the segment's actual density sample. Finite quadrature replaces
unbounded null tracking and its spectral-weight variance. Shadow rays likewise use bounded exponential quadrature.

Beam proposal is the maximum channel of `throughput * phase * (cached_incident + unoccluded_incident)`. Sky has its
own positive proposal, so a sharp beam does not consume its samples. Each correction subtracts its own prediction,
retaining the full source sum in expectation for the sampled path; nearest prediction changes variance, not that
target. The complete camera predictor also cancels under the same weights, even when the coarse angular matrix
and continuous local convolution differ.

Boundary reconstruction uses the unclamped interface from
[`Cumulus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cumulus.glsl), avoiding zero gradients inside a saturated
density ramp. Four probes estimate the normal; a short search and four bisections locate actual vacuum. It applies
only when extrapolation length `2 D0` fits inside the smallest modeled billow scale. Interior light comes from the
coarse angular mean. Twelve characteristic segments reconstruct exterior incident scattered light through actual
density with attenuation and source refill before querying the far-field endpoint.

At one extrapolation length, Marshak matching gives `Phi_0 = (Phi_bulk + Phi_in)/2` and
`J_n = (Phi_bulk - Phi_in)/4`. A normalized positive HG angular field carries these moments, with asymmetry in
`[-0.5, 0.5]`, and blends into the directional cache over `2 D0` inside the patch. Equal isotropic interior/incident
fields remain unchanged under arbitrary density geometry. This is a local high-order approximation: estimated
normals, incident hemispheres, planar patches and distance blending are not an exact finite-cloud angular solution.
Explicit low-order paths and the camera still use the full selected phase function.

Flight radius, quadrature, terminal field and angular LUT remain biased. Local corrections do not feed net flux into
the coarse solve, so its power balance does not prove image-wide conservation. Default history stays at 32 frames
with depth reprojection and variance clipping. History counts use positive reconstruction weights; signed radiance
keeps its existing filter, so Lanczos lobes cannot make confidence negative. Sparse residual noise and temporal/FP16 bias remain possible.

[`cloud-local-transport-check.py`](../../../scripts/test/cloud-local-transport-check.py) checks spectral segment sampling
against a nonuniform analytic slab, vacuum/equilibrium, camera and source reservoir cancellation, angular sampling,
scattering-order and shadow-quadrature convergence, equal-mass narrow channels, Lambertian ground input and boundary
angular power/flux. `--gpu-history PATH` checks finite signed scattering, transmission and depth in the frame-end
history atlas. These model checks do not establish photograph agreement.

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
