# Shader Pack Generation and Packaging

Run Kotlin generators from [`scripts/`](../../scripts/). You need Git, a `kotlin` command that runs `.main.kts`, a
JDK compatible with [Shadesmith](../../scripts/shadesmith.jar), and a target Minecraft/Iris instance for shader
compilation and visual checks. See the [code layout](project-structure.md) and [pipeline dependencies](../rendering/pipeline.md)
when locating a pass or resource.

## First generation in a fresh checkout

Run `./shadesmith.ps1` once from `scripts/` before the first options generation. It creates the ignored
`scripts/shadesmith.shaders.properties` fragment with the image and atlas declarations consumed by the options generator.

## What to run after a change

| Maintained input | Command from `scripts/` | Output |
| --- | --- | --- |
| Included `.glsl` implementation | None. Reload the pack in Iris. | Iris reads the source through its wrapper. |
| [`programs.main.kts`](../../scripts/programs.main.kts) | `kotlin options.main.kts` | Numbered compute wrappers, the ignored `programs.shaders.properties` fragment, and final properties. |
| [`options.main.kts`](../../scripts/options.main.kts), [`options.lib.kts`](../../scripts/options.lib.kts), [`sponsors.txt`](../../scripts/sponsors.txt) | `kotlin options.main.kts` | [`Options.glsl`](../../shaders/base/Options.glsl), [`TextOptions.glsl`](../../shaders/base/TextOptions.glsl), [language files](../../shaders/lang/), wrappers, and final properties. Edit the library when changing the DSL itself. |
| Hand-maintained [`scripts/shaders.properties`](../../scripts/shaders.properties) | `kotlin options.main.kts` | Merges custom resources, SSBOs, blending, and uniforms into [`shaders/shaders.properties`](../../shaders/shaders.properties). |
| [`shaders/shadesmith.json`](../../shaders/shadesmith.json) | `./shadesmith.ps1` followed by `kotlin options.main.kts` | [`Textile.glsl`](../../shaders/base/Textile.glsl), ignored preprocessed output and image/atlas properties, and final properties. |

`options.main.kts` runs the program generator itself and aggregates `scripts/*.shaders.properties`. Its optional version
argument overrides the UI version selected from [`changelogs/`](../../changelogs/). For a wrapper-numbering preview
alone, run `kotlin programs.main.kts`.

## Add a compute pass controlled by a setting

1. Register a new `SETTING_*` in [`scripts/options.main.kts`](../../scripts/options.main.kts) using the existing
   `toggle`, `slider`, `constToggle`, or `constSlider` patterns. Add the appropriate screen/profile placement and
   English and Simplified Chinese `lang` entries. This is also the source for editing an existing setting.
2. Place the shader entry under [`shaders/pass/<stage>/`](../../shaders/pass/) and add `pass(...)` to the appropriate
   `ProgramType` block in [`scripts/programs.main.kts`](../../scripts/programs.main.kts) at its dependency position.
   For a toggle, use `cond("defined(SETTING_EXAMPLE)")`. Use a value comparison for a numeric setting.
   `cond` generates the `program.<name>.enabled` branches. Use `define` for a regular shader define,
   `constDefine` for Iris's const section, and `indirect` for a dispatch buffer index and offset. Iris
   `workGroups` and `workGroupsRender` values must be literals or macros expanding to literals.
3. Regenerate from the repository root with

   ```powershell
   Set-Location scripts
   kotlin options.main.kts
   ```

4. Review the generated wrappers, options, languages, and properties. In Iris, compile the affected enabled and
   disabled branches and exercise representative profiles, history resets, and scenes.

Entries in a single `pass(pathA, pathB, ...)` share one number and may run concurrently. Keep producer/consumer
dependencies in separate numbered slots. When removing a pass, remove its obsolete entry and any unused settings
or program declarations. The generator replaces the numbered wrappers.

## Add a Textile tile

Choose `transient_*` for temporary screen tiles, `history_*` for screen history, or `persistent_*` for fixed-size
cross-frame resources under `fixed`. Use a supported format and specify `width` and `height` for a fixed tile.
Use the generated Textile bindings in GLSL and review overlap, formats, read/write stages, and history reset behavior.
During ordinary edits, run Shadesmith for tile-layout changes. It also runs for initial bootstrap and ZIP packaging.

## Local Java and Shadesmith configuration

Keep `scripts/config.properties` machine-local. It can set the JDK root and preprocessed shader-pack output directory.

```properties
JAVA_PATH=C:/path/to/jdk
SHADESMITH_OUTPUT=./shadesmitth
```

Without `JAVA_PATH`, [`shadesmith.ps1`](../../scripts/shadesmith.ps1) derives `java.home` from `java` on PATH and
[`make-zip.kts`](../../scripts/make-zip.kts) uses the JVM running Kotlin. Relative `SHADESMITH_OUTPUT` paths are based at
`scripts/` for the documented commands. The default directory really is spelled `shadesmitth`.

## Build a test ZIP

From `scripts/`, run `kotlin make-zip.main.kts [version] [--no-commit-hash]`.
The script runs Shadesmith and packages the processed shaders, runtime languages and textures, changelogs,
licenses, and root player READMEs under the ignored `builds/` directory. The default filename includes the short
commit hash. It also includes the branch name with slashes replaced by hyphens, except on `main` or when the branch
name contains `dev` as a slash-, hyphen-, or underscore-separated component.
`--no-commit-hash` omits the hash and branch suffix.

## Before committing

Run `git diff --check` and `git diff --cached --check`. Commit tracked generated results alongside their sources.
Keep generated property fragments, the preprocessed `shadesmitth/` directory, and AOT caches out of Git.
For a runtime change, reload the pack in the target Iris instance and check compilation and the resulting scene.
