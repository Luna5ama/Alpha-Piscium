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
| 1     | [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | Ray marches convective towers with high-resolution single scattering, opacity-weighted lighting samples and local transport |
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

Grey cloud extinction supplies one scalar camera-opacity PDF for all wavelengths. The primary march integrates
transmission and weighted depth, while reservoirs store only selected distances. Let `A = 1-T_cloud` and
`d alpha_cloud = T_cloud sigma_t ds`. Reordering the foreground-air integral gives
`T_cloud L_clear + integral d alpha_cloud [L_air(0,s) + T_air(0,s) albedo (P B + P S)]`.
This follows the [equation of transfer](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer).
The renderer samples the bracket only at selected depths, with separate SS/air and MS sample counts. The expectation
is unchanged for the selected quadrature model; depth-sampling variance replaces repeated cache/air queries.
No per-depth cached predictor or RGB reservoir weight is stored.
There are two direct-light samples by default, or four when the local sample budget is four. Sun and moon use
actual density, atmospheric transmittance, Earth occlusion, and the full phase function at the sampled solar/lunar
disc direction.
One continuous phase-importance direction estimates sky/ground single scattering at each selected depth, with
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
Cumulus uses grey extinction/scattering coefficients `0.1854716894` and `0.1854716020`, the averages of the previous
AP0 coefficients. Illumination and the selected phase function remain RGB. Camera, direct shadows and the first local
flight use this true extinction. Reduced high-order characteristics analytically include repeated within-channel
scattering; their coefficient is distinct from both true extinction and transport extinction `a1`.

Air scattering and transmission are integrated once from the camera to each selected depth, using 4–32 samples.
Cloud-opacity importance sampling weights foreground radiance and cloud lighting consistently; the primary density
loop no longer carries air-lighting state or samples the lighting cache.
[`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl)
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
| 2, begin | [`CloudTransportPrepare`](../../../shaders/pass/begin/CloudTransportPrepare.comp.glsl), `CT_HISTORY` | Reprojects previous `(P - f I) S` using camera displacement and cloud advection into a separate scratch field; invalid history supplies zero |
| 3, begin | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl), [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Computes uncollided sky/ground light, then initializes the current source from `P B` and the reprojected scattered-light guess |
| 4, deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl) | Solves directional characteristics using a descending shared-memory plane, at most 48 KiB per group |
| 5, deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Restricts the true directional residual into two coupled diffusion channels; the V-cycle uses 4³ block-preconditioned CG, physical-center prolongation and red/black smoothing |
| 6, deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl), [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Lifts and damps the correction while preserving nonnegative radiance; later iterations execute only while residual or power-defect tolerance is unmet |
| 7, deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | Records actual residual/power, stores cell-mean scattered radiance and preconvolves `P S` for the camera and next-frame initialization |

Before spatial discretization, repeated scattering within the same angular channel is summed analytically.
For `f = min_RGB(P_jj)`, define `sigma_f = sigma_t - sigma_s f` and `P_f = P - f I`. The equation becomes
`omega_j.grad S_j + sigma_f S_j = sigma_s (P B + P_f S)_j`. This is an algebraic rearrangement of the selected
angular equation, including heterogeneous density. `P_f` stays nonnegative, absorption stays `sigma_t-sigma_s`,
and the equal-solid-angle matrix retains its power balance. The residual phase CDF occupies phase-record alpha;
RGB still stores the complete matrix. PDFs are differences of the stored CDF, including floating-point rounding.

Characteristics distinguish incoming light, outgoing face values and cell averages. The optical depth is
`sigma_f / sum(abs(omega)/cell_size)` for scattered light; uncollided light still uses `sigma_t`.
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
times. Offsets and index arithmetic stay unsigned: on the tested NVIDIA path, large signed offsets at 64³
shifted the scattered-light cache by one vec4 and corrupted reduction addresses. This is an addressing failure, not
a NaN/Inf failure; validate the decoded mean/source relation and power sums at both resolutions. Its byte allocation and the buffer-dump reader must change together if this packed layout changes. At frame end,
`ctRadiance` holds outgoing values and `ctLifted` holds cell means. Coefficient RGB retains the solver parameters;
level-zero fast/weight/exchange alpha lanes are unused and zero. The old boundary mean cache is removed.

The grid spans ±121 km horizontally. Each horizontal axis assigns 75% of its cells uniformly to ±12 km;
a continuously differentiable exponential skirt covers the remaining distance. The inverse sampling map and
physical-center interpolation use the same allocation. The vertical extent covers the spherical cloud layer down to
2.5 km below its nominal base. `SETTING_CLOUDS_MS_RESOLUTION` selects 32³ (default) or 64³;
the inner horizontal spacing is 1 km or 500 m respectively. `SETTING_CLOUDS_MS_ITERATIONS` selects 1, 2 or 4
(default), as an upper iteration limit. Previous `(P - f I) S` is only an initial guess for the current operator and source.
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

### Local scattering paths and opacity sampling

[`LocalTransport.glsl`](../../../shaders/techniques/atmospherics/clouds/LocalTransport.glsl) uses continuous camera
and second-scattering directions, then the same 24-channel equation as the shared solver. High-order flights
analytically sum the common diagonal phase mass; the cache supplies the unresolved distant field. No normal,
planar boundary patch, Marshak-to-HG reconstruction, distance blend or density AO remains in local shading.

| Stage | Operation |
|---|---|
| Camera integration | Integrates true cloud transmission and stores only opacity-weighted selected distances |
| First local flight | Uses true grey extinction and the continuous RGB phase function; the projected angular cache is integrated before selecting its discrete continuation channel |
| High-order flight | Uses `sigma_f` through actual density, sampling the nonnegative residual phase `P - f I`; camera extinction and physical shadow rays remain unchanged |
| Terminal flight | Integrates transmission over 16 exponential density bins; WRS selects a position conditional on scattering, and adds both weighted scattering refill and transmitted endpoint light |
| External sources | Separate positive reservoirs select one sky event and one sun/moon event per path; actual fine shadows evaluate their radiance without subtracting an inaccurate coarse shadow |
| Camera shading | Evaluates full local MS, direct light and foreground air at the selected depths; root directions use stratified spatiotemporal blue noise mapped to half-open bin centers |

`SETTING_CLOUDS_MS_LOCAL_SAMPLES` selects 0, 1 (default), 2, or 4 paths; zero queries the shared MS cache and retains direct light.
`SETTING_CLOUDS_MS_LOCAL_BOUNCES` selects 1, 2, 4 (default), 8, 16, or 32 transport stages. The first
stage is continuous. Subsequent stages include arbitrarily many within-channel events, so this is no longer a
literal physical-scattering count. More stages extend local transport; they do not change an artistic boundary
weight or guarantee lower finite-sample noise.

For the last flight, `T` is its integrated endpoint transmission and `A` is the sum of segment scattering-selection
weights `T_i (1-exp(-tau_i))`. An opacity-weighted reservoir selects position `X`; the estimate is
`T C_j(endpoint) + A (sigma_s/sigma_f) [(P-f I) C + P B]_j(X)`. Conditional position sampling retains the integral
expectation while always evaluating the transmitted endpoint. It replaces an all-or-nothing collision/escape
choice and avoids querying the full cache at every fine density step. Vacuum has `T=1`, `A=0`; constant conservative
equilibrium remains constant. Finite bin quadrature and a three-coarse-cell flight radius still introduce bias.

At intermediate high-order events, `(P-f I) C` is the known conditional expectation of cached continuation.
Adding it and subtracting the sampled `C` before continuing leaves the same finite-depth transport target.
The selected continuous phase LUT's finite sampling accuracy remains an approximation. UNORM blue-noise values are mapped into `[0,1)` bin centers so the RGB phase sampler cannot select a fourth component.

Sky source directions are continuous; high-order source evaluation first samples a uniform direction inside the
selected angular cell. Beam sources use the integrated solar/lunar cell phase. Their positive proposals use
unoccluded incident power and path/phase weights. A completely occluded source contributes exactly zero, rather
than noise from subtracting a falsely lit coarse prediction. Sources, projected continuation and camera-depth sampling keep their own probability factors, so neither
uncollided nor already-scattered light is added twice.

The coarse grid, 24-channel high-order representation, flight radius, density quadrature and terminal field remain
approximations. Local corrections do not feed net flux into the coarse solve, so its power balance does not prove
image-wide conservation. Default history remains 32 frames with depth/wind reprojection and variance clipping;
signed radiance is retained, while history counts use positive reconstruction weights. Temporal/FP16 bias and
finite-sample noise remain possible. Increasing depth refines this chosen model rather than proving photograph
agreement or full angular/spatial convergence.

[`cloud-local-transport-check.py`](../../../scripts/test/cloud-local-transport-check.py) checks grey flight sampling,
vacuum/equilibrium, the coupled air/cloud integral identity, UNORM sampling bounds, camera and source reservoirs, resummation against an independent continuum slab, conditional
continuation, forced-terminal variance, shadow quadrature and equal-mass narrow channels. `--gpu-history PATH`
checks finite signed scattering, transmission and depth. The main transport checker also verifies the phase CDF,
frame-end `P S` against an Iris SSBO dump.

```sh
python test/cloud-local-transport-check.py --output ../builds/cloud-transport/local-numerics.json
```

### CPU radiance reference

The `cloud-light-transport-comparison` work item uses the frozen specification at
`d2f1ef0fe2b1ac9a962a062fe7d2af7f922611ba`, with real-time baseline
`c09cfdc76e79cb3e4e986d16e29730777283e78d` in wt9. Its first CPU stage is independent of the
24-direction cache and local reconstruction. [`cloud-reference.cpp`](../../../scripts/test/cloud-reference.cpp)
traces continuous directions through a fixed union of eight constant-density ellipsoids. Union chord lengths,
overlaps, free flights and solar transmission are analytic. It samples the normalized 11.54 um HG+Draine
phase directly, with no angular LUT, spatial grid or fixed scattering-order cutoff. Roulette starts at event 32.
This follows the [volume transfer equation](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer).

The fixture spans about 150 x 216 x 110 metres. The 32 x 40 orthographic pixel-centre camera is fixed at
Z=-450 m, looking along +Z; X covers [-100,100] m and Y [-10,240] m. Inside the union, density is 0.15,
extinction is 0.027820753413 /m and albedo is 0.999999528504. The sun has radius 0.266 degrees and normal
irradiance 1, directed from +Y (side) or +Z (back). Sky, air, ground and neighboring towers are absent.
The manifest records every ellipsoid, seed scheme, input and raw float64 channel. This is a reusable sample
cloud, not the production eroded procedural density.

The first run, `802ff939-d1e4-4654-bdc1-2c41fdb030b3`, completed 41,943,040 paths with eight independent
replicates and 512/2,048/8,192/32,768 paths per pixel. Phase normalization differs from 1 by 2.11e-9;
sampled phase mean and the analytic sphere single-scattering check are within two standard errors. Union and
vacuum-gap checks pass. Pixel RMS standard error divided by RMS radiance drops from 2.52% to 0.318% for
backlight, but remains 36.2% for side light; its notch mean has 9.32% standard error. **The first side-light
result is not a converged image reference.** Rare near-forward connections dominate its multiple-scattering variance.
[`cloud-reference-report.py`](../../../scripts/test/cloud-reference-report.py) retains single/multiple components,
independent-replicate confidence intervals and comparisons of disjoint sample blocks. Display plots use a separate
explicit linear scale for each light direction, shared by all components in that row; measurements use raw radiance.

Compile the C++20 program with an optimizing compiler, then pass a preallocated output directory, maximum
samples per replicate, replicate count and CPU worker count. Run the Python reporter on the same directory.
Vela records the run and job source. Production-cloud agreement, neighboring-tower influence, sky input, motion
and Iris cost remain separate obligations; no live game or GPU measurements were performed in this stage.

<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"reference.json"} -->
<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"radiance.bin"} -->
<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"convergence.json"} -->
<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"reference.png"} -->
<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"convergence.png"} -->

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
