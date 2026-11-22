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

### Upsampling replay result, 2026-11-21

All six parent paths reproduced their archived native and pooled RGB arrays
bit-for-bit (0 ULP). The complete upsample arrays were then crossed without
changing each row's downsample, injection or shared features. All eight archived
phases were retained. The static four-cell boards, original 21-point y=24 line,
extended line covering the rotated neighboring peak, RGB direction profiles and
8x8 pooled wide views are in the local
[replay report](../../.vela/local/runs/984195dd-5f4f-4d9f-95ab-f075a3ef3c42/replay.html).

The main missing fine peaks and weak bitmap stroke follow the sample-allocation
row. With control upsampling, compact User 8 still has only 36.0-37.8% of the
control center peak, and the 67 stroke at (-44,0) retains 31.1-31.7% of its target.
Dense User 8 samples with candidate upsampling preserve 103.9-108.6% of the
control center peak and its two neighboring valleys. This does not clear the
whole-filter gate: exchanging upsampling makes User 8 core/halo errors worse in
both mixed cells. Reconstruction and sample allocation remain coupled.

This CPU diagnostic therefore excludes an upsampling-only explanation or repair
for the dominant missing peaks in these frozen exports. It does not establish
that upsampling is unimportant, that all errors come from compression, or that
any hybrid qualifies as a generated game preset. No fitting, GPU, game or editor
operation occurred. See the phase-resolved evidence and exact source identities
in [the experiment record](../../experiments/dpf-bloom-20261108.md).

<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"input-manifest.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"attribution.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"comparison-summary.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"replay.html"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"reproducer-source.zip"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"source-delta.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"delivery-check.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"report-manifest.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"reproduce.md"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"user8-parent-reproduction.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"user8-analysis.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"user8-line-y24.png"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"user8-line-y24-regression.png"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"user8-p0-native.png"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"rotated-parent-reproduction.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"rotated-analysis.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"rotated-line-y24.png"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"rotated-line-y24-regression.png"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"rotated-p0-native.png"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"bitmap-parent-reproduction.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"bitmap-analysis.json"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"bitmap-line-y24.png"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"bitmap-line-y24-regression.png"} -->
<!-- vela:evidence {"run":"984195dd-5f4f-4d9f-95ab-f075a3ef3c42","path":"bitmap-p0-native.png"} -->

### Compressing a reference Bloom filter

Start the next experiment from the half-resolution stage that already resolves the rays. Its direct samples
specify how much RGB light each neighboring half-resolution texel contributes. Build this grid by spreading
each saved tap over its bilinear footprint, using the positions and channel weights from the frozen renderer.
The [replay report](../../.vela/local/runs/984195dd-5f4f-4d9f-95ab-f075a3ef3c42/replay.html) supplies the
three control filters, the rejected compact fits and the eight-phase native and pooled comparisons.

User 8 illustrates the cost of direct sampling. In its control grid, 560 texels exceed 0.1% of the largest
absolute channel coefficient. Of those, 183 have even x and even y. Each direct bilinear read covers at most
one such texel, so retaining nonzero support at those sites alone requires at least 183 reads per half-resolution
pixel. That contributes 45.75 area-weighted reads per full-resolution pixel, with another 14.32 in the fixed
stages. This bound applies to direct coverage of that particular grid. Visual accuracy permits approximation,
and a shared short filter can contribute the same local pattern at many positions.

- Compare a compact direct filter with a sum of translated, shared short filters plus direct samples for the
  remainder. Choose local patterns, repeat locations, RGB weights and sample counts from the control grid.
  Use the same procedure for User 8, its 9.5° rotation and the 67 bitmap. The frozen control is an offline
  input for this diagnostic. The production generator must ultimately obtain its starting filter from the
  supplied RGB target, including inputs without a recipe.
- Keep the reference downsampling, upsampling and coarser injections together. First verify that the expanded
  grid reproduces the control through that chain within floating-point accumulation error. Evaluate both
  compact representations through the same chain. Count the reads that form each shared value and those that
  sample it later, and report intermediate writes. Use the closest attainable costs at or below the previous
  whole-filter read estimates (46.60, 46.85 and 46.35 respectively). One intermediate cost point can bracket
  a visible loss.
- Select using separate training or monitor phases. Keep the eight archived phases for regression and reserve
  fresh phases before choosing candidates. Judge against the authored target and the matching control at native
  resolution. Report both valleys and the weakest fine peaks, ray continuity and tips, the weak 67 stroke,
  RGB separation, the core and the pooled outer halo. A cost-versus-quality plot or table and the worst
  same-exposure images are sufficient to decide whether compression is worth integrating.

This study ends with the best quality found at the tested costs and a visible example of any remaining loss.
Qualifying the normal Fit and Export path on a new input and measuring the complete Bloom against 4× COD at
720p, 1080p and 1440p are later runtime checks. Static read counts carry no GPU timing claim.

### Reference-compression result, 2026-11-21

The three decoded D1 grids reproduced their reference native responses with
maximum absolute RGB errors of 1.12e-8, 9.31e-9 and 2.33e-10, respectively.
The fixed downsample, upsample and coarse injections were unchanged. Direct and
shared compression used equal D1 read caps and separate formation/later-read and
write accounting. Selection used four monitor phases; the old eight positions
and four predeclared new positions were kept outside selection.

At whole-filter read estimates 46.569/46.819/46.319, shared compression improved
the 67 weak stroke from the direct filter's 84.4% to 93.7% of target. It did not
preserve the star kernels: the weakest tested neighboring peaks were 63.0% of
control for User 8 and 37.6% for its rotation. The strongest visible loss is the
truncation of long rays; the intermediate-cost variants still have gaps and
isolated bright points. Their coefficient-grid fits also dropped the weak outer
D1 support, degrading User 8/rotation's pooled halo. An even/even support bound
does not establish a visual or GPU feasibility limit.

The bounded six-pattern search therefore produced a negative result for keeping
the full reference quality at these costs. It does not establish a general
limit of shared spatial filters. The local
[cost/quality report](../../.vela/local/runs/42f19587-f64f-47ce-8cc3-cb2b86841457/comparison.html)
contains target/control/native figures, per-channel ray profiles, both valleys,
new-phase results and exact coefficients. The detailed method and source
identities are in [the experiment record](../../experiments/dpf-bloom-20261108.md).
Normal Fit/Export, unseen kernels and the three-resolution 4x COD GPU checks
remain unqualified. Existing presets and review services were preserved.

Across all twelve phases, the low-cost shared User 8 filter retains 3.17–3.38% of the reference luminance
128 native pixels along the horizontal ray. The intermediate budget restores that point to about 105%, yet
at 144 pixels both budgets retain only 4.79–5.11%. At phase (165, 90) the low-cost value at 128 pixels is
`1.78e-6` in linear luminance, down from `5.33e-5`. The visible break persists as the sampled span grows.

<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"protocol.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"cost-quality.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"comparison.html"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"compression-audit.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"quality-details.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"support-and-tips.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"reproducer-source.zip"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"selected-coefficients.zip"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"source-manifest.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"result-manifest.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"delivery-check.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"numeric-check.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"reproduce.md"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"user8-grid-reproduction.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"user8-quality.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"user8-line-y24.png"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"user8-line-y24-regression.png"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"user8-p9-native.png"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"rotated-grid-reproduction.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"rotated-quality.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"rotated-line-y24.png"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"rotated-line-y24-regression.png"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"rotated-p9-native.png"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"bitmap-grid-reproduction.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"bitmap-quality.json"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"bitmap-line-y24.png"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"bitmap-line-y24-regression.png"} -->
<!-- vela:evidence {"run":"42f19587-f64f-47ce-8cc3-cb2b86841457","path":"bitmap-p9-native.png"} -->

### Allocating DPF samples along visible features

Use the gaps in the **reconstructed image** to choose where a sample is worth adding, moving or sharing. The
previous search ranked a tap by the coefficient-grid error it removed. On the positive horizontal User 8 ray,
all 49 native samples at x=96 through 144 fall below half the reference line contrast in both low-cost layouts
at every one of the 12 archived phases. The longest such run at the intermediate cost is five pixels. Here
contrast means linear Rec.709 luminance at `(x,0)` minus the brighter of `(x,-4)` and `(x,4)`. The minimum
reference contrast on this interval exceeds `2.54e-5`. Use this line as a regression check. The half-contrast
threshold describes the length of a gap and does not set the visual acceptance bar.

- Trace continuous ridges from the full RGB target. Use local orientation, transverse width and contrast to
  follow a thin line through its tip or around a curved stroke. Respect real gaps, branches and differences
  between channels. The allocator takes pixels as its input. Preset identity, authored angles and review probe
  coordinates are for validation only. A target with a smooth halo can have no ridge to protect.
- Use the longest visibly deficient stretches on training or monitor phases to propose placements **during
  layout selection**. Check both sides of the ridge so that widening it or drawing an isolated bright point
  cannot repair a gap in the score. Use absolute linear RGB deficits and same-exposure images to judge weak
  signals. Compare one fine-level allocation with the same search allowed to move locally smooth energy to a
  coarser level. Base that choice on local width and the composed native response. A narrow tip can be far
  from the center. Preserve total channel energy across level transfers and measure the core and outer halo.
- Reuse the frozen reference grids, renderer, low-cost direct/shared outputs and 12 phase controls from the
  [reference compression run](../../.vela/local/runs/42f19587-f64f-47ce-8cc3-cb2b86841457/comparison.html).
  Use the same spatial DPF representation and per-input read limits, with intermediate writes accounted for
  separately. Start at the low cost. The existing intermediate cost can bracket a failure. Reserve fresh phases
  before choosing placements. For User 8, its 9.5° rotation and 67, show the detected paths, worst continuous
  gap, neighboring peaks and valleys, RGB along the tips, weak strokes, core and pooled halo. Include the worst
  native images at the original exposure and state the actual per-level reads and writes.

The original half-resolution star fits and saved 67 are the quality controls. If the candidate preserves all
three, exercise normal Fit and Export on User 8 and on a separate unseen recipe and its identical unannotated
RGB raster. Check the exported responses against the targets. Load the generated User 8 via Iris Custom for a
matched full-frame 1080p Small Lamps capture and standalone Bloom timings at 1440p, 1080p and 720p against COD,
aiming for 3× with a 4× cap. If native quality still falls short, return the measured cost/quality comparison and
the remaining break. Keep the current game presets during evaluation.


### Feature-continuity result

At the same low read caps, image-guided placements restored the reach of the
previously truncated horizontal ray. The known User 8 regression stretch at
x=96–144 went from a 49-pixel below-half-contrast run to 3 pixels in either
layout, including the four new held-out positions. The probe was used only for
evaluation. Allowing a D2 contribution reduced native RGB error for both star
kernels, and the 67 weak stroke reached close to its intended brightness.

The native gate still fails. The star rays have visible beading and missing
stretches, the split User 8 layout has a separate 27-pixel tracked break, and
both layouts have higher pooled outer-halo error than the star controls. The
[comparison](../../.vela/local/runs/44758c93-76de-4f66-9a63-90228267ee65/comparison.html)
contains the detected paths, same-exposure counterexamples, RGB peak and flank
samples, and measured coefficient counts at each level. See the
[experiment record](../../experiments/dpf-bloom-20261108.md) for the conditional
gate decision, assumptions and evidence identities.

<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"protocol.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"reproduce.md"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"reproducer-source.zip"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"comparison.html"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"summary.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"verification.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"quality-gate.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"diagnostics.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"user8-p1-fine.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"user8-p10-fine.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"rotated-p2-fine.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"bitmap-p1-fine.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"user8-metrics.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"user8-paths.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"user8-detected.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"user8-y24.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"user8-selection.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"user8-assessment.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"rotated-metrics.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"rotated-paths.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"rotated-detected.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"rotated-y24.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"rotated-selection.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"rotated-assessment.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"bitmap-metrics.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"bitmap-paths.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"bitmap-detected.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"bitmap-y24.png"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"bitmap-selection.json"} -->
<!-- vela:evidence {"run":"44758c93-76de-4f66-9a63-90228267ee65","path":"bitmap-assessment.json"} -->
<!-- vela:evidence {"run":"5505c0ef-b6a3-4200-8aca-9009d3d9eeef","path":"invalid-attempt.json"} -->
<!-- vela:evidence {"run":"e06c8d24-cf1c-4665-b150-732d0be56917","path":"invalid-attempt.json"} -->

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
