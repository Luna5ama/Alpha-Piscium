# Atmosphere, Sky, and Clouds

Language: English | [简体中文](../../sc/modules/atmosphere-clouds.md)

Alpha Piscium separates low-frequency LUTs, view-dependent sky-view/epipolar data, volumetric clouds, and final local
composition. Atmosphere preparation runs in `begin`, clouds render early in composite, and air/water volumes resolve
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
| 4     | [`clouds/amblut/Sample`](../../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl)           | Samples previous/current cloud ambient information |
| 5     | [`GenerateSkyViewLUT`](../../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl)       | Writes `uimg_skyViewLUT`                           |
| 6     | [`clouds/amblut/Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl)           | Writes the 32×192 `persistent_cloudsAmbLUT`        |

Fixed LUT dimensions are declared in [`shaders/shadesmith.json`](../../../shaders/shadesmith.json);
screen-sized/configurable images are declared in [`scripts/shaders.properties`](../../../scripts/shaders.properties).

Both `uimg_skyViewLUT` dimensions use `SETTING_SKYVIEW_RES`. `uimg_epipolarData` uses `SETTING_EPIPOLAR_SLICES` for its
width, and maps `SETTING_SLICE_SAMPLES` to a height of 385, 769, 1537, or 3073. Adding a supported sample count requires
updating both the property-size mapping and shader indexing logic.

## Cloud passes

When `SETTING_CLOUDS_CU` is enabled:

| Order | Pass                                                                                             | Purpose                                                                                                                                                                                                      |
|-------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | Ray marches cumulus using the ambient LUT, atmosphere LUTs, shadows/lights, and project cloud noise                                                                                                          |
| 2     | [`clouds/ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)          | Applies history accumulation, confidence/variance processing, and upscaling to the low-resolution result; this is the [`ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl) stage |
| 3     | Later air/local composition                                                                      | Reads the accumulated cloud scattering and transmittance                                                                                                                                                     |

The main screen tiles are `transient_lowCloudRender`, `transient_lowCloudAccumulated`, and `history_lowCloud` (
RGBA32UI). The hand-maintained property fragment declares custom cloud phase-LUT, cirrus, cumulus base/detail, and curl
textures. High cirrus is sampled by the shared sky/cloud path rather than a separate compute program.

### Cumulus isotropic multiple scattering

The cumulus renderer reuses the existing ordered eight-sample sun-light column. For each source sample jittered uniformly in distance within its quadratic-distance bin,
let `U_i` be the prefix optical depth from the start of the light column to that source position, `sigma_s` the scattering coefficient,
`sigma_tr` the transport coefficient, `ds` the sample length, and `r` the source radius. This diffusion approximation
uses `sigma_tr ≈ sigma_t` in the implementation. With dimensionless absorption fraction (albedo deficit) `a = 0.001`
multiplying optical depth, per-channel asymmetry `g`, and `k = sqrt(3a)`, the direct upstream-prefix estimator is

$$
W_i=\frac{(\sigma_s\,ds)\sigma_{tr}}{r},\qquad
\Phi=\sum_{i=1}^{8}W_i e^{-aU_i}
     \left(1-e^{-(1-g)U_i}\right)e^{-kU_i}.
$$

The implementation clamps the build rate `1 - g` to a nonnegative value so transformed color spaces cannot turn
scattering buildup into amplification.

Intensity is applied before a fixed soft compression:

$$
\Phi_{\mathrm{mapped}}=1-e^{-\max(\mathrm{intensity}\,\Phi,0)}.
$$

This bounds each mapped channel below `1`. `SETTING_CLOUDS_CU_ISOTROPIC_MS_INTENSITY` supplies `intensity`; `0`
disables the contribution, the default `1.0` is the current artistic gain, and `0.25` approximately matches the
omitted `3/(4π)` normalization reference. The result is added independently of the existing WDT22
multiple-scattering term; it does not replace or modify that term.

The accumulated `phi_fwd` field is isotropic and follows the prefix estimator, but final view-path readout intentionally
uses `msPhase = mix(UNIFORM_PHASE, layerParam.medium.phase, 0.7)` to retain controlled directional structure. This
rendering choice is layered after the isotropic field, not part of its transport recurrence.

The receiver-local boundary weight is

$$
H(x,z)=\mathrm{thickness}\;\mathrm{saturate}(\mathrm{baseCoverage}_{raw}(x,z)),\qquad
\Delta=\mathrm{clamp}(0.05/\_LOW\_BASE\_FREQ,0.025,0.2),
$$

$$
\partial_xH=\frac{H(x+\Delta,z)-H(x-\Delta,z)}{2\Delta},\qquad
\partial_zH=\frac{H(x,z+\Delta)-H(x,z-\Delta)}{2\Delta},\qquad
N=\mathrm{normalize}(-\partial_xH,1,-\partial_zH),
$$

$$
C_{top}=\mathrm{saturate}\!\left(\frac{N\cdot\mathrm{renderParams.lightDir}+0.5}{1.5}\right),\qquad
C_{bottom}=1-\exp\!\left(-\frac{\max(h_{local},0)}{0.1\,\mathrm{mix}(1,4,h_{column})}\right),\qquad
B_{eff}=C_{top}C_{bottom}.
$$

Here `baseCoverage_raw` is the existing pre-height coverage value, `h_column = saturate(baseCoverage_raw)` reuses the
receiver density lookup, and `h_local` is the normalized receiver height (`0` at the cumulus-layer base and `1` at its
top). The constants correspond to `b = 0`, `p = 1`, and `H_bottom = 0.1`, matching the density model's existing normalized
`0.1` bottom scale. `C_top` uses the actual `renderParams.lightDir`, not the cone-jittered light-march direction. The gate
is evaluated once per occupied receiver sample and multiplies every source weight before accumulation, intensity, and
compression. It costs four extra coverage-proxy evaluations with no extra center lookup, and adds no pass, resource,
texture resource, or density march.

The estimator is adapted from AshenOneArt's [HanPi Volume Cloud implementation](https://github.com/AshenOneArt/HPVolumeCloud/blob/27e799914493de9fa527179312ed72a39d08e225/VolumetricClouds.hlsl)
and [forward-flux derivation](https://github.com/AshenOneArt/HPVolumeCloud/blob/27e799914493de9fa527179312ed72a39d08e225/Docs/PhiFwd_FromRTE.md).

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
| Low cloud               | Upscale factor; history length/confidence/variance; minimum/maximum steps; height/thickness/density/coverage/phase; isotropic multiple-scattering intensity; wind; and shape frequencies |
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
