# Other Scripts

Language: English | [简体中文](../../sc/development/scripts.md)

Daily development only needs the program list, options, Shadesmith, and ZIP scripts described
in [workflows](workflows.md). This page keeps the remaining offline tools out of the normal shader-edit loop. Run
commands from [`scripts/`](../../../scripts/) unless the file is PowerShell or Python.

## Release

### [`release.main.kts`](../../../scripts/release.main.kts)

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

Most of these tools generate static assets committed under [`shaders/textures/`](../../../shaders/textures/);
`extract-opac` also writes ignored intermediate CSVs, while `nishina-e` only prints values. Check dimensions, sample
counts, and output paths before running. They are not prerequisites for shader reloads.

| Script                                                                      | Purpose                                                                       | Main related asset                                                                                                                                  |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`reusetex.main.kts`](../../../scripts/reusetex.main.kts)                   | Generates precomputed ReSTIR spatial-reuse sampling textures                  | [`restir_reusetex0.bin`](../../../shaders/textures/restir_reusetex0.bin) … [`restir_reusetex3.bin`](../../../shaders/textures/restir_reusetex3.bin) |
| [`gen-spec-brdf-lut.main.kts`](../../../scripts/gen-spec-brdf-lut.main.kts) | Generates the split-sum specular BRDF LUT                                     | [`specular_brdf_lut.bin`](../../../shaders/textures/specular_brdf_lut.bin)                                                                          |
| [`gen-f82-table.main.kts`](../../../scripts/gen-f82-table.main.kts)         | Generates the material Fresnel/F82 lookup table                               | [`f82.bin`](../../../shaders/textures/f82.bin)                                                                                                      |
| [`gen-noisetex.main.kts`](../../../scripts/gen-noisetex.main.kts)           | Generates a 64³ RGBA16 white-noise volume                                     | [`white_noise_64x64x64.bin`](../../../shaders/textures/white_noise_64x64x64.bin)                                                                    |
| [`extract-opac.main.kts`](../../../scripts/extract-opac.main.kts)           | Parses OPAC/CIE data into intermediate CSVs and a runtime cloud phase texture | [`data/opac_raw/`](../../../data/opac_raw/), [`opac_cloud_phases.bin`](../../../shaders/textures/opac_cloud_phases.bin)                             |
| [`nishina-e.main.kts`](../../../scripts/nishina-e.main.kts)                 | Computes and prints Nishina-related numeric arrays                            | Standard output                                                                                                                                     |

Usually run:

```powershell
kotlin <script>.main.kts
```

For runtime texture generators, check binary size/layout and synchronize the `customTexture.*` format, dimensions, and
type in [`scripts/shaders.properties`](../../../scripts/shaders.properties). On the shader side, synchronize sampler
dimensionality, integer/float access class, and any hard-coded channel assumptions.

## Bloom kernel fitting

[`fit-bloom.py`](../../../scripts/fit-bloom.py) calibrates the shared five-tap downsample and eight-tap upsample
against a four-scale Gaussian proxy and writes [`BloomTaps.glsl`](../../../shaders/techniques/BloomTaps.glsl).
It checks unit mass and centroids across all 256 proxy phases. The runtime separately combines eight levels with
equal weights. [`fit-bloom-shape.py`](../../../scripts/fit-bloom-shape.py) uses the actual raw level-1 Bloom carrier
for sparse fitting and emits [`BloomShapeTaps.glsl`](../../../shaders/techniques/BloomShapeTaps.glsl).
[`bloom_shape_targets.py`](../../../scripts/bloom_shape_targets.py) defines the directional taps and the glyph
targets. The glyph fits translate by whole level-1 texels to center their visual bounds. The
[`Trollface input`](../../../scripts/assets/trollface-source.png) is Carlos Ramirez's drawing, sourced from the
[Wikipedia file page](https://en.wikipedia.org/wiki/File:Trollface.png).

Both tools require Python, NumPy and SciPy; shape input also uses Pillow. Run from the repository root at low
process priority. Recalibrate the shared taps only when intentionally changing the standard kernel; then refit the
shape carrier as well:

```powershell
python -B scripts/fit-bloom.py
python -B scripts/fit-bloom-shape.py --builtins --output shaders/techniques/BloomShapeTaps.glsl
```

For a custom target, create an output directory and use either input form:

```powershell
python -B scripts/fit-bloom-shape.py --image <weights.png> --carrier bloom --mip 1 --tap-budget 64 --output <fit.json> --glsl <shape.glsl>
python -B scripts/fit-bloom-shape.py --function <kernel.py> --size 129 --carrier bloom --mip 1 --tap-budget 64 --output <fit.json> --glsl <shape.glsl>
```

PNG input is opaque 8/16-bit grayscale containing raw linear weights. Function input is trusted Python defining
`psf(x, y)` and returning an array matching the coordinate grids. Both require odd dimensions, finite nonnegative
weights and positive total mass. The middle texel is the origin; x points right and y down, in full-resolution
pixels. Inputs normalize to unit mass. JSON gather offsets use mip texels; GLSL flips y for atlas coordinates and
merges adjacent fitted taps into equivalent bilinear reads. Runtime GLSL export requires `--carrier bloom --mip 1`.
Call the generated `bloom_shapePrepare()` from every invocation before checking output bounds.

Custom output is a filter fragment for a maintainer to evaluate, not an automatic player-facing option. `--report`
adds a JSON report for `--builtins`. Keep generated source with the generator change; temporary fit reports stay in
ignored output directories. Recheck input phases, edge behavior and Iris rendering after changing weights. See
[Bloom](../modules/post-processing.md#bloom) for the runtime contract and accepted cost.

## Binary texture helpers

[`bintex.main.kts`](../../../scripts/bintex.main.kts) is a standalone converter for one or more inputs into a binary
texture with explicit dimensions and channels:

```powershell
kotlin bintex.main.kts <dimensions_joined_by_underscore> <channels> <output> <input...>
```

[`mipsizepadded.main.kts`](../../../scripts/mipsizepadded.main.kts) calculates padded mip-layout ratios and prints the
maximum X/Y ratio for offline size inspection. Neither belongs in the regular build loop.

## Program debugging

[`programs-full.ps1`](../../../scripts/programs-full.ps1) is a convenience wrapper equivalent to:

```sh
# Run from scripts/
kotlin ./programs.main.kts
kotlin ./options.main.kts
```

It must be run from [`scripts/`](../../../scripts/); it does not change to its own directory. Because the options
generator invokes the program generator again, program output appears twice. The shortest complete generation remains [
`options.main.kts`](../../../scripts/options.main.kts) alone.

## Cloud verification

Run these deterministic checks from the repository root:

```powershell
uv run scripts/cloud_isotropic_ms_check.py --source-root .
uv run scripts/cloud_boundary_confidence_check.py --source-root .
```

[`cloud_isotropic_ms_check.py`](../../../scripts/cloud_isotropic_ms_check.py) checks the isotropic multiple-scattering
recurrence, FP32 stability, sampling distribution, phase readout, and source integration contracts. [
`cloud_boundary_confidence_check.py`](../../../scripts/cloud_boundary_confidence_check.py) checks the boundary-confidence
formula, FP64/FP32 bounds and edge behavior, mutation rejection, and its GLSL source contracts. These are mathematical
and static checks; they do not compile shaders or replace target-Iris and visual validation.

## Color and display-transform experiments

| File                                                      | Purpose                                                          |
|-----------------------------------------------------------|------------------------------------------------------------------|
| [`agxtest.main.kts`](../../../scripts/agxtest.main.kts)   | Offline numerical experiments for AgX/display-transform formulas |
| [`agxinv.py`](../../../scripts/agxinv.py)                 | Validates or derives the inverse AgX transform                   |
| [`colorspaces.py`](../../../scripts/colorspaces.py)       | Generates or checks color-space matrices and constants           |
| [`adobe-fresnel.tsv`](../../../scripts/adobe-fresnel.tsv) | Offline Fresnel input data, not an executable script             |

These tools do not update GLSL automatically. When adopting a result, make the explicit constant/formula change under [
`shaders/util/colors/`](../../../shaders/util/colors/) or [
`shaders/techniques/displaytransform/`](../../../shaders/techniques/displaytransform/), then validate both numerically
and in a live render.

## Data files

[`sponsors.txt`](../../../scripts/sponsors.txt) is read by the options generator and becomes generated option
text/localization. Run `kotlin options.main.kts` after changing it.
