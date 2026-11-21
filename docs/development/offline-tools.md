# Offline Tools and Release

Daily development only needs the program list, options, Shadesmith, and ZIP scripts described
in [workflows](generators.md). This page keeps the remaining offline tools out of the normal shader-edit loop. Run
Kotlin commands from [`scripts/`](../../scripts/).

## Release

### [`release.main.kts`](../../scripts/release.main.kts)

```powershell
kotlin release.main.kts <version> [-1] [-2] [-3] [-4] [-5] [-6] [-7]
```

This is a maintainer release tool, not a local build command. It requires `changelogs/<version>.md` and local
`scripts/tokens.properties`, then performs numbered ZIP, rename, Git tag/push, GitHub, Modrinth, CurseForge, and
announcement steps. `-N` skips step N.

It switches branches, creates/pushes a tag, and calls external publishing APIs. The ZIP is built from the current
checkout before the script switches to `main`/`dev`, so start on the exact content being released. Skipped steps have
dependencies: later uploads may require the ZIP built/renamed by steps 1/2, and skipping step 4 also skips its final
rename. Do not rely on a stale ZIP. Never commit `tokens.properties`.

## Offline textures and lookup tables

Most of these tools generate static assets committed under [`shaders/textures/`](../../shaders/textures/);
`extract-opac` also writes ignored intermediate CSVs, while `nishina-e` only prints values. Check dimensions, sample
counts, and output paths before running. They are not prerequisites for shader reloads.

| Script | Purpose | Main related asset |
| --- | --- | --- |
| [`reusetex.main.kts`](../../scripts/reusetex.main.kts) | Generates precomputed ReSTIR spatial-reuse sampling textures | [`restir_reusetex0.bin`](../../shaders/textures/restir_reusetex0.bin) … [`restir_reusetex3.bin`](../../shaders/textures/restir_reusetex3.bin) |
| [`gen-spec-brdf-lut.main.kts`](../../scripts/gen-spec-brdf-lut.main.kts) | Generates the split-sum specular BRDF LUT | [`specular_brdf_lut.bin`](../../shaders/textures/specular_brdf_lut.bin) |
| [`gen-f82-table.main.kts`](../../scripts/gen-f82-table.main.kts) | Generates the material Fresnel/F82 lookup table | [`f82.bin`](../../shaders/textures/f82.bin) |
| [`gen-noisetex.main.kts`](../../scripts/gen-noisetex.main.kts) | Generates a 64³ RGBA16 white-noise volume | [`white_noise_64x64x64.bin`](../../shaders/textures/white_noise_64x64x64.bin) |
| [`extract-opac.main.kts`](../../scripts/extract-opac.main.kts) | Parses OPAC/CIE inputs under [`data/`](../../data/) into intermediate CSVs and a runtime cloud phase texture | [`data/opac_raw/`](../../data/opac_raw/), [`opac_cloud_phases.bin`](../../shaders/textures/opac_cloud_phases.bin) |
| [`nishina-e.main.kts`](../../scripts/nishina-e.main.kts) | Computes and prints Nishina-related numeric arrays | Standard output |

Usually run:

```powershell
kotlin <script>.main.kts
```

For runtime texture generators, check binary size/layout and synchronize the `customTexture.*` format, dimensions, and
type in [`scripts/shaders.properties`](../../scripts/shaders.properties). On the shader side, synchronize sampler
dimensionality, integer/float access class, and any hard-coded channel assumptions.

## Bloom kernel fitting

The [local Bloom editor](../../scripts/bloom_dpf_web.py) previews a complete RGB point-spread function from
the base halo, optional Gaussian core, and selected shape. Camera and prism dispersion have independent controls.
The Custom Lobes recipe lets you author directional lobes and optionally link opposing pairs.

From the repository root, run the editor with Python, NumPy, SciPy, and Pillow at idle process priority.

```powershell
python -B scripts/bloom_dpf_web.py
```

Open [the local editor](http://127.0.0.1:8765). Edit the recipe, run the fit, and inspect the fitted response
across the held-out sampling phases at the same exposure as the target. The ZIP export contains the resolved
recipe, fitted coefficients and errors, and [the Custom GLSL table](../../shaders/techniques/bloom/BloomDpfCustom.glsl).
Copy that table to the indicated shader path, reload the pack, and select Custom in the Bloom shape menu.

The [target model](../../scripts/bloom_dpf_target.py) authors the complete response. The
[fitter](../../scripts/bloom_dpf_fit.py) uses the [CPU cascade](../../scripts/bloom_dpf_model.py) to fit the
reconstruction and raw-level injection weights with the same bilinear footprints used at runtime. The
[reference archive](../../scripts/assets/bloom-dpf-reference.npz) freezes the published baseline taps.
The frozen Trollface taps derive from [Carlos Ramirez’s drawing](../../scripts/assets/trollface-source.png),
sourced from the [Wikimedia file page](https://en.wikipedia.org/wiki/File:Trollface.png).

The [built-in coefficient bank](../../scripts/assets/bloom-dpf-kernels.json) owns the selected coefficients.
Regenerate the [built-in GLSL table](../../shaders/techniques/bloom/BloomDpfPresets.glsl) with the following command.

```powershell
python -B scripts/bloom_dpf_fit.py
```

Use `--select-fit` with a reviewed fit for an unmodified built-in recipe to replace its bank entry and regenerate
the table. Review the complete response and recheck Iris compilation and rendering. See
[Bloom](../rendering/post-processing.md#bloom) for the runtime pipeline and limitations.

### Geometry-driven DPF fitting

**First prototype evaluated.** The editor's **Fit** and **Export** actions reach the raster-driven fitter and the
game's **Custom** option. This prototype misses the User 8 quality floor and 1440p budget. An earlier design ranked
the axes supplied by the lobe recipe. Its six-branch candidate restored long diagonals but covered only 69% and
74% of the two vertical rays in their weakest phases. The same-target half-resolution fit covered 100% in both.

Use the complete linear RGB impulse response as the input to sample allocation. Keep the optical origin, pixel
spacing, extent through the outer halo, relative channel energy and existing normalization identical for recipe and
raster input. The equivalence check can pass a lossless floating-point raster directly to the shared fitter.
Choose each level's 2D sample positions, RGB weights and sample count offline by comparing the **composed** DPF
response with the target over multiple input phases. Authored angles may seed a search. Let image evidence select
or discard directions, including directions absent from those seeds. The solver, number of shared filters and use of
full or half resolution are implementation choices. The game runs the exported spatial DPF for the whole kernel
with its core, halo, shape and dispersion in that response.

- Fit the [original User 8 recipe](../../scripts/assets/bloom-dpf-user8-recipe.json) and its identical raster with all
  lobe metadata removed. Both must preserve the same visible details within the same cost limit. Also try an
  off-axis rotation and an existing bitmap shape such as 67 to expose preferred axes and forced symmetry.
- Compare the target, current User 8, old half-resolution fit and candidate at native pixel scale and matched
  exposure. Reserve input phases outside fitting and layout selection. Check the weakest phase for the gaps
  between the 75°, 90° and 105° rays and their opposites, continuity to the horizontal tips, and RGB separation.
  The candidate must match or exceed the old half-resolution quality. Its original run used a 160 px fine radius,
  768 px outer radius and 18 iterations. The saved control is `pure-dpf-joint/routes-2-4/user8/quality/legacy-half-fit.json`
  in the DPF run artifacts (SHA-256 `9033014299d2d1f771dde5a9f43fd533645d84a781033c18f31cac05386b6546`).
  The existing full-frame Small Lamps controls are in the local
  `builds/dpf-user8-layout-repair/user8-small-lamps-comparison.html`.
- Load the normal export in Iris and supply the matched full 1080p Small Lamps frame. Measure standalone Bloom
  against original COD Bloom (`0bf3a72ba247f9121f6c28691f6f6b44bec3c9a2`) at 720p, 1080p and 1440p using matched
  forward and reverse windows. Count every executed Bloom pass for each variant and report Bloom plus PostComposite
  separately. Aim for 3× and stay within 4× at each resolution. The 1440p windows in the implementation evidence
  below include every active pass for the raster-driven candidate. Keep the current presets until quality and cost both pass. If they
  conflict, deliver the closest candidates with their visible differences and measured costs for review.

### Interactive Bloom line profile

**Implemented and verified in the experimental editor.** Dragging a line across the kernel image overlays the
target and fitted brightness curves in one plot. Keep the line visible on the image. Label both curves and use
distance from the first endpoint in full-resolution pixels on the horizontal axis. Redrawing the line or selecting
another held-out phase should update the plot from the available response without starting a fit.

Sample both floating-point linear RGB responses over the same footprint at the same points and convert both to
luminance identically. Use one vertical scale for both curves. Apply the editor's exposure and linear/log display
only after sampling. The wide views contain pooled outer values, so retain their represented footprint and indicate
that resolution. Tie the fitted curve to the active recipe and phase (including the phase mean). Show the target
alone while the matching fit is unavailable. The existing preview paths in `scripts/bloom_dpf_web.py` and the
kernel view in `scripts/bloom_dpf.html` provide the data and interaction entry points.

Use the existing saved User 8 fit for browser acceptance. Draw across adjacent
rays away from the core at native scale, then switch phase. Check the peaks and intervening valley against the
underlying response samples, and check that editing the recipe clears the stale fitted curve. Deliver a rendered
editor view with the line and both curves visible and record its endpoints and phase. Reuse that line on the first
geometry-driven candidate for the fitting comparison.

### Residual-guided fitting of thin features

**Completed experiment for 2.1.** The first raster-driven fit searched for straight ridges through the optical origin.
Its User 8 export selected four axes. At 24 pixels from the core, the reconstructed middle peak reached 56–58%
of the old half-resolution response and both neighboring valleys filled in. The 67 bitmap also lost detail.
The search scores layouts by regional L2, whereas the continuous fit also uses signed pixel derivatives.
These results leave two plausible causes, the proposed sample locations and the score used to choose them.

- Start from the sealed fit and its full RGB response. Compare the existing layout proposals with one alternative
  that proposes a short 2D filter from a **local patch of reconstruction error** and learns where else to apply it.
  The patch may be off-center. Along a thin feature, favor overlapping samples that preserve its continuity. Across
  it, measure missing peaks and excess light in adjacent valleys separately. Use target pixels to find these probes
  for any input. Reuse the existing DPF intermediates and allow ordinary 2D taps where sharing offers little gain.
- Compare proposals at matched sample and intermediate-write costs and fitting effort. First rescore the control
  proposals with the same local contrast criterion, then add the local proposals. This distinguishes a scoring
  improvement from a change in what the sparse filters can represent. Keep the published four phases and the
  horizontal line at y=24 as regression controls. Set aside fresh input phases before selection. Compare User 8
  and its 9.5° rotation with their own old-half controls, and 67 with its saved bitmap control. Inspect peaks,
  intervening valleys, gaps along rays or strokes, horizontal tips, RGB separation and the broad halo at the
  same exposure. Report the strongest remaining failure if the new allocation still misses visible detail.
- Target fewer expensive half-resolution reads. In the two 1440p candidate windows, `composite66` (half-resolution
  reconstruction) took 0.44144 and 0.44126 ms. `composite53` (prefilters and downsample) took 0.08387 and 0.08393 ms.
  At the measured COD budget the complete filter must save 0.09229 and 0.09330 ms respectively. With all other
  stages at their recorded times, those two passes together have 0.43302 and 0.43188 ms. Use this to choose a
  lower-cost layout. Read counts alone cannot predict its GPU time. Take a fresh 1440p paired measurement once
  a candidate clears native quality, then complete the existing 720p/1080p and full-frame checks if it fits 4×.

Reuse the review runs indexed below, including the source snapshot, full-frame viewer and per-program timings.
The current built-in User 8 remains the game preset during this experiment. The experiment ends with a measured
candidate that clears both gates or a comparison showing which gate still binds.

### Residual-footprint experiment result, 2026-11-21

The bounded fixed-pyramid experiment completed three inputs and eight phases per
input. A uses regional scoring; B rescored the identical four existing proposals
with local peaks, valleys and continuity; C adds four off-center residual patches
and their combination. A local-only challenger also received the same 48-gradient
budget when the initial selector rejected all local proposals. Identical fits
were reused. Seven unique fits consumed 336 gradient calls, each covering sixteen
training phases. Four fresh phases were reserved before selection and never used
to choose a layout or fitted iterate.

Read proxies were reduced to about 46.60/46.85/46.35 for User 8/rotation/67.
Carrier writes remained 1/1/.75 full-resolution pixels, respectively, with
identical stage tap counts and levels between arms. These are static costs, not
new GPU timings. User 8's original y=24 center retained only 39.5-43.2% of its
old-half response with A/B and 36.9-40.2% with the local challenger. The bitmap's
local proposal improved on B, but native L2 remained .174-.181 versus .090-.097
for its existing control; its pooled outer error was also worse. Native rays and
strokes still show gaps and beads. All three inputs failed the quality gate, so
no GPU window, Iris reload or preset replacement was attempted for this package.

The experiment is a negative result for this proposal pool, fixed pyramid and
budget, not a representation limit. Full phase-separated measurements, original
coefficients and reproducer sources are indexed in
[the experiment record](../../experiments/dpf-bloom-20261108.md). The initial
execution snapshot and final delivery snapshot are separately identified. No
runtime path or user-facing editor was replaced; ports 8765 and 8768 were retained.

<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"experiment-summary.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"final-selection.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"source-snapshot.zip"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"delivery-source-snapshot.zip"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"source-delta.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"metadata-correction.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"challenger-protocol.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"numeric-check.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"delivery-check.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"user8-protocol.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"user8-selection.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"user8-native-metrics.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"user8-line-y24.png"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"user8-native-eight-phase.png"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"rotated-protocol.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"rotated-selection.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"rotated-native-metrics.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"rotated-line-y24.png"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"rotated-native-eight-phase.png"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"bitmap-protocol.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"bitmap-selection.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"bitmap-native-metrics.json"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"bitmap-line-y24.png"} -->
<!-- vela:evidence {"run":"0d459ffd-adc4-43b5-985b-27a528566ead","path":"bitmap-native-eight-phase.png"} -->

### Separating upsampling from per-level Bloom samples

The old User 8 half-resolution fit resolves the three neighboring rays at y=24. The latest sparse fit loses the
middle peak in all eight phases. Matching taps by position shows that both fits use the same downsample locations
at all eight levels. Per-tap RGB weights differ by less than `3.8e-9`. The rotated and bitmap pairs give the same
result. The upsample filters and the samples that add light at each level both changed materially. For
example, the old User 8 fit has 883 direct half-resolution taps. The selected fit has 32 there plus four shared
filters. Their upsamplers both use nine taps in the final step, with different positions and weights.

Replay two new combinations per input to tell which change accounts for the lost contrast. For User 8 and its
9.5° rotation use their respective old-half controls and selected B fits. For 67 use the saved bitmap control
and selected C fit. Each row keeps its allocation of samples fixed.

| Samples contributing light at each level | Control upsampling | Candidate upsampling |
| --- | --- | --- |
| Control | Saved control | Replay with candidate upsampling |
| Candidate | Replay with control upsampling | Saved candidate |

Swap the complete `upsample` array, keeping the direct `injection` samples, all shared `features` and the
downsample array together on each row. Use the actual per-channel weights and optical origin. Verify the two
parent paths reproduce their saved native and pooled responses within the renderer's numerical precision before
interpreting the swaps. Use the eight archived phases, the full linear RGB targets and the same exposure. The
two new cells on each input need only forward rendering. The saved coefficients are the entire input to this
diagnostic. The ordinary editor, game and GPU are outside its scope.

Show the four responses on the same axes. Compare the height of the y=24 peak **and both neighboring valleys**,
the rays through their tips and their RGB separation, the weak 67 stroke at `(-44,0)`, and the pooled outer
halo for every phase. If changing upsampling helps only with one sample allocation, report that interaction.
Link the native boards and profiles and identify the largest difference that survives the swap. This test
ends with that attribution, including an inconclusive result if neither component alone explains it.

If the compact samples recover the reference detail with control upsampling, the next design question is how
the fitter can learn those reconstruction filters from pixels at the same tap counts. If the dense controls
retain detail with either upsampler and the compact samples lose it with both, study compression starting from
the quality-passing response. An improvement using a control's coefficients still needs independent fitting
from the input raster and a matched GPU budget test before it can qualify as a generated game preset.

### DPF implementation evidence, 2026-11-21

The interactive line profile passed independent source and actual Edge browser
acceptance. The bound phase-(139,3) image records its real floating-point drag
endpoints, 22 samples, Network response and active plot data. The separate
integer-endpoint comparison has 21 samples over four held-out phases.

The first geometry-driven fitter completed normal Fit and Export for User 8,
its identical metadata-free raster, a 9.5-degree rotation and the existing 67
bitmap. This bounded candidate failed native quality against the saved controls.
Iris compilation, the matched 1080p Small Lamps frame and all twelve GPU budget
windows completed. Standalone Bloom costs 2.71x COD at 720p, 3.64-3.66x at 1080p
and 4.59-4.60x at 1440p. The last resolution fails the 4x budget, independently of
the native quality failure. PostComposite is reported separately. The dedicated
game saved and exited normally and its GPU lease was returned. The detailed inputs,
measurements, rejected candidates, independent reviews and remaining work are in
[the DPF experiment record](../../experiments/dpf-bloom-20261108.md).

<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"checks.json"} -->
<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"same-line-comparison.json"} -->
<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"browser-check.json"} -->
<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"candidate-browser-check.json"} -->
<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"user8-profile-mean.png"} -->
<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"user8-profile-phase0.png"} -->
<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"candidate-profile-mean.png"} -->
<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"candidate-profile-phase0.png"} -->
<!-- vela:evidence {"run":"bdf75f9c-4d35-43c7-938a-813d08c3f534","path":"qa-identity-binding.json"} -->
<!-- vela:evidence {"run":"afbc378c-f5b7-41e0-8e35-2ec3507387f1","path":"user8-fit.json"} -->
<!-- vela:evidence {"run":"afbc378c-f5b7-41e0-8e35-2ec3507387f1","path":"user8-response.npz"} -->
<!-- vela:evidence {"run":"afbc378c-f5b7-41e0-8e35-2ec3507387f1","path":"user8-export.zip"} -->
<!-- vela:evidence {"run":"afbc378c-f5b7-41e0-8e35-2ec3507387f1","path":"raster-equivalence.json"} -->
<!-- vela:evidence {"run":"afbc378c-f5b7-41e0-8e35-2ec3507387f1","path":"validation-summary.json"} -->
<!-- vela:evidence {"run":"afbc378c-f5b7-41e0-8e35-2ec3507387f1","path":"user8-native-controls.png"} -->
<!-- vela:evidence {"run":"5d544efd-f3f3-4f3f-acb3-d851ea482e54","path":"rotated-native-controls.png"} -->
<!-- vela:evidence {"run":"5d544efd-f3f3-4f3f-acb3-d851ea482e54","path":"bitmap-native-controls.png"} -->
<!-- vela:evidence {"run":"7b0257e7-97f8-4bfd-8f96-53218fbbd47d","path":"candidate-phase0-bound.json"} -->
<!-- vela:evidence {"run":"7b0257e7-97f8-4bfd-8f96-53218fbbd47d","path":"candidate-phase0-bound.png"} -->
<!-- vela:evidence {"run":"82a818c2-85a3-4e5e-af0d-426d8dec9127","path":"runtime-summary.json"} -->
<!-- vela:evidence {"run":"82a818c2-85a3-4e5e-af0d-426d8dec9127","path":"small-lamps-comparison.html"} -->
<!-- vela:evidence {"run":"82a818c2-85a3-4e5e-af0d-426d8dec9127","path":"comparison-inputs.json"} -->
<!-- vela:evidence {"run":"82a818c2-85a3-4e5e-af0d-426d8dec9127","path":"source-snapshot.zip"} -->
<!-- vela:evidence {"run":"82a818c2-85a3-4e5e-af0d-426d8dec9127","path":"delivery-check.json"} -->
<!-- vela:evidence {"run":"784926ab-6d52-4d8d-bb08-bb5f2402c503","path":"gpu-admission.json"} -->

## Binary texture helpers

[`bintex.main.kts`](../../scripts/bintex.main.kts) is a standalone converter for one or more inputs into a binary
texture with explicit dimensions and channels:

```powershell
kotlin bintex.main.kts <dimensions_joined_by_underscore> <channels> <output> <input...>
```

[`mipsizepadded.main.kts`](../../scripts/mipsizepadded.main.kts) calculates padded mip-layout ratios and prints the
maximum X/Y ratio for offline size inspection. Neither belongs in the regular build loop.

## Program debugging

[`programs-full.ps1`](../../scripts/programs-full.ps1) is a convenience wrapper equivalent to:

```sh
# Run from scripts/
kotlin ./programs.main.kts
kotlin ./options.main.kts
```

It must be run from [`scripts/`](../../scripts/); it does not change to its own directory. Because the options
generator invokes the program generator again, program output appears twice. The shortest complete generation remains [
`options.main.kts`](../../scripts/options.main.kts) alone.

## Cloud verification

Run these deterministic checks from the repository root:

```powershell
uv run scripts/cloud_isotropic_ms_check.py --source-root .
uv run scripts/cloud_boundary_confidence_check.py --source-root .
```

[`cloud_isotropic_ms_check.py`](../../scripts/cloud_isotropic_ms_check.py) checks the isotropic multiple-scattering
recurrence, FP32 stability, sampling distribution, phase readout, and source integration contracts. [
`cloud_boundary_confidence_check.py`](../../scripts/cloud_boundary_confidence_check.py) checks the boundary-confidence
formula, FP64/FP32 bounds and edge behavior, mutation rejection, and its GLSL source contracts. These are mathematical
and static checks; they do not compile shaders or replace target-Iris and visual validation.

## Color and display-transform experiments

| File | Purpose |
| --- | --- |
| [`agxtest.main.kts`](../../scripts/agxtest.main.kts) | Offline numerical experiments for AgX/display-transform formulas |
| [`agxinv.py`](../../scripts/agxinv.py) | Validates or derives the inverse AgX transform |
| [`colorspaces.py`](../../scripts/colorspaces.py) | Generates or checks color-space matrices and constants |
| [`adobe-fresnel.tsv`](../../scripts/adobe-fresnel.tsv) | Offline Fresnel input data, not an executable script |

These tools do not update GLSL automatically. When adopting a result, make the explicit constant/formula change under [
`shaders/util/colors/`](../../shaders/util/colors/) or [
`shaders/techniques/displaytransform/`](../../shaders/techniques/displaytransform/), then validate both numerically
and in a live render.

## Data files

[`sponsors.txt`](../../scripts/sponsors.txt) is read by the options generator and becomes generated option
text/localization. Run `kotlin options.main.kts` after changing it.
