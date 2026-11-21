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

**Ready for implementation.** Deliver one offline fitter through the editor's **Fit** and **Export** actions and the
game's **Custom** option. In the current DPF checkout, `scripts/bloom_dpf_policy.py` ranks axes supplied by the lobe
recipe and tries a few layouts. Its six-branch User 8 candidate recovers long diagonals, but its weakest tested
phases cover only 69% and 74% of the two vertical rays. The same-target half-resolution fit covers 100% in both.

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
  separately. Aim for 3× and stay within 4× at each resolution. The previous corrected 1440p capture omitted
  active passes and needs a new measurement. Keep the current presets until quality and cost both pass. If they
  conflict, deliver the closest candidates with their visible differences and measured costs for review.

### Interactive Bloom line profile

**Ready for implementation.** In the local editor, dragging a line across the kernel image should overlay the
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

### DPF implementation evidence, 2026-11-21

The interactive line profile passed independent source and actual Edge browser
acceptance. The bound phase-(139,3) image records its real floating-point drag
endpoints, 22 samples, Network response and active plot data. The separate
integer-endpoint comparison has 21 samples over four held-out phases.

The first geometry-driven fitter completed normal Fit and Export for User 8,
its identical metadata-free raster, a 9.5-degree rotation and the existing 67
bitmap. This bounded candidate failed native quality against the saved controls.
Live compilation, the matched Small Lamps frame and complete GPU budget windows
remain in progress through shared-resource admission. The detailed inputs,
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
