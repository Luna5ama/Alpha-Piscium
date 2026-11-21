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
