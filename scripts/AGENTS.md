# Changing generators and packaging

Run Kotlin entry points from `scripts/`. For a new option-controlled pass, follow the [worked registration example](../docs/development/generators.md#add-a-compute-pass-controlled-by-a-setting). The [program list](programs.main.kts) owns order, grouping, conditions and indirect dispatch. The [options source](options.main.kts) owns settings, screens, profiles and player-facing translations. Edit the [DSL](options.lib.kts) when its behavior needs to change.

## Build the tracked outputs

| Change | Action from this directory |
| --- | --- |
| Pass registration, options, sponsors or [hand-maintained Iris properties](shaders.properties) | Run `kotlin options.main.kts`. It invokes the program generator and aggregates the property fragments. |
| [Textile layout](../shaders/shadesmith.json) | Run `.\shadesmith.ps1`, then `kotlin options.main.kts`. Also run Shadesmith before the first options generation in a fresh checkout to create the image/atlas fragment. |
| Included GLSL only | Reload the pack in Iris. |
| Offline texture or LUT | Verify dimensions, binary format and channels against the texture declaration and shader readers. See the [offline tools](../docs/development/offline-tools.md). |

Review the generated wrappers, option and Textile bindings, languages and final shader properties affected by the change. Commit tracked outputs with the owning source. Keep intermediate property fragments, caches, local Java/output configuration and release credentials out of Git.

## Package or release

For a local test ZIP, refresh any affected generated files, then run `kotlin make-zip.main.kts`. [Packaging](make-zip.kts) runs Shadesmith and assembles the pack under `../builds/`. Inspect the archive for the changed shader or asset. The [release script](release.main.kts) can switch branches, push tags, upload and announce a release. Run it only for an explicitly requested release from the intended revision.
