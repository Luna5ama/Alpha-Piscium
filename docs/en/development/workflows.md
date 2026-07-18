# Common Workflows and Configuration

Language: English | [简体中文](../../sc/development/workflows.md)

This page documents Alpha Piscium's generation chain. Refer to
the [Iris Shaders documentation](https://github.com/IrisShaders/docs) for the property syntax itself. Run Kotlin
commands from [`scripts/`](../../../scripts/); PowerShell examples assume the repository root.

## File ownership

| File                                                                                                                                                                             | Owner                                                                                                                                 | Edit directly              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| [`scripts/programs.main.kts`](../../../scripts/programs.main.kts)                                                                                                                | Pass order, wrappers, conditions, and indirect dispatch                                                                               | Yes                        |
| [`scripts/programs.shaders.properties`](../../../scripts/programs.shaders.properties)                                                                                            | Ignored intermediate from the program generator                                                                                       | No                         |
| [`scripts/shadesmith.shaders.properties`](../../../scripts/shadesmith.shaders.properties)                                                                                        | Ignored image/atlas fragment from Shadesmith                                                                                          | No                         |
| [`scripts/options.main.kts`](../../../scripts/options.main.kts)                                                                                                                  | Settings, profiles, screens, and translations                                                                                         | Yes                        |
| [`scripts/options.lib.kts`](../../../scripts/options.lib.kts)                                                                                                                    | Options DSL implementation                                                                                                            | Only when changing the DSL |
| [`scripts/shaders.properties`](../../../scripts/shaders.properties)                                                                                                              | Hand-maintained properties, custom resources, SSBOs, blending, and uniforms                                                           | Yes                        |
| [`shaders/shaders.properties`](../../../shaders/shaders.properties)                                                                                                              | Aggregated by [`options.main.kts`](../../../scripts/options.main.kts) from all `scripts/*.shaders.properties` files and option output | No                         |
| [`shaders/base/Options.glsl`](../../../shaders/base/Options.glsl), [`TextOptions.glsl`](../../../shaders/base/TextOptions.glsl), [`shaders/lang/*.lang`](../../../shaders/lang/) | [`options.main.kts`](../../../scripts/options.main.kts)                                                                               | No                         |
| [`shaders/{setup,begin,shadowcomp,prepare,deferred,composite}*.csh`](../../../shaders/)                                                                                          | [`programs.main.kts`](../../../scripts/programs.main.kts)                                                                             | No                         |
| [`shaders/shadesmith.json`](../../../shaders/shadesmith.json)                                                                                                                    | Textile tile layout                                                                                                                   | Yes                        |
| [`shaders/base/Textile.glsl`](../../../shaders/base/Textile.glsl)                                                                                                                | Tracked bindings updated by Shadesmith from [`shadesmith.json`](../../../shaders/shadesmith.json)                                     | No                         |

## Local `config.properties`

The repository does not track [`scripts/config.properties`](../../../scripts/). It configures local Java/Shadesmith
tooling only; it is not the Iris [`shaders.properties`](../../../shaders/shaders.properties). Two keys are supported:

```properties
JAVA_PATH=C:/path/to/jdk
SHADESMITH_OUTPUT=./shadesmitth
```

- `JAVA_PATH` is the JDK root. Without it, [`shadesmith.ps1`](../../../scripts/shadesmith.ps1) derives `java.home` from
  `java`, while [`make-zip.kts`](../../../scripts/make-zip.kts) uses the JVM running Kotlin.
- `SHADESMITH_OUTPUT` is the preprocessed shader-pack output root. The default spelling is literally `./shadesmitth`;
  with the documented commands, relative paths are based at [`scripts/`](../../../scripts/).
- [`scripts/.gitignore`](../../../scripts/.gitignore) excludes the file. Keep it machine-local and do not force-add it
  to Git.

## Edit ordinary GLSL

Put shared implementation in [`shaders/techniques/`](../../../shaders/techniques/) or [
`shaders/util/`](../../../shaders/util/), and new entry points in [`shaders/pass/`](../../../shaders/pass/). The current
program list still references a few direct compute entries under `techniques/` (Bloom, RTWSM, and atmospherics); editing
them does not require an unrelated move. A plain `.glsl` edit needs no generator because wrappers include source files
at runtime.

## Add or change a setting

1. Declare the `SETTING_*` in [`scripts/options.main.kts`](../../../scripts/options.main.kts) using an existing
   `toggle`, `slider`, `constToggle`, or `constSlider` pattern; place it in the appropriate screen/profile and provide
   English and Chinese `lang` entries.
2. Only then use it in GLSL or a `cond(...)` in [`programs.main.kts`](../../../scripts/programs.main.kts); do not
   introduce an unregistered `SETTING_*` in GLSL first.
3. Run:

```powershell
cd scripts
kotlin options.main.kts
```

4. Review [`shaders/base/Options.glsl`](../../../shaders/base/Options.glsl), [
   `shaders/lang/en_US.lang`](../../../shaders/lang/en_US.lang), [
   `shaders/lang/zh_CN.lang`](../../../shaders/lang/zh_CN.lang), and [
   `shaders/shaders.properties`](../../../shaders/shaders.properties).

The options generator invokes the program generator first, synchronizing wrappers, program-property fragments, and the
final property file. An optional version argument overrides the UI version; otherwise the highest version in [
`changelogs/`](../../../changelogs/) is used.

## Add, remove, or reorder a pass

1. Put an entry shader under [`shaders/pass/<stage>/`](../../../shaders/pass/); keep reusable code under [
   `techniques/`](../../../shaders/techniques/) or [`util/`](../../../shaders/util/).
2. Add `pass(...)` to the correct `ProgramType` block in true execution order.
3. `pass(pathA, pathB)` creates same-index `name.csh` and `name_a.csh`. With `allowConcurrentCompute=true`, grouped
   entries cannot depend on each other's writes; group only work that can share a dispatch slot concurrently.
4. Apply `define`, `constDefine`, `cond`, and `indirect` as needed. Iris `workGroups` and `workGroupRender` directives
   must remain literals or macros expanding to literals.
5. For a quick numbering preview, run:

   ```sh
   cd scripts
   kotlin programs.main.kts
   ```

   Before committing, run:

   ```sh
   cd scripts
   kotlin options.main.kts
   ```

   This writes the new program fragment into the final property file.

When deleting or replacing a pass, also remove its old entry point, wrapper source, and settings that are no longer
used.

## Add a texture tile

1. Choose the lifetime in [`shaders/shadesmith.json`](../../../shaders/shadesmith.json):
    - `transient_*`: temporary resources under `screen`.
    - `history_*`: history resources under `screen`.
    - `persistent_*`: fixed-size cross-frame resources under `fixed`.
2. Select an existing supported format; fixed tiles also require `width` and `height`.
3. Consume the generated binding from [`shaders/base/Textile.glsl`](../../../shaders/base/Textile.glsl) rather than
   duplicating offsets or formats.
4. Run:

```powershell
cd scripts
./shadesmith.ps1
kotlin options.main.kts
```

5. Shadesmith writes the ignored [
   `scripts/shadesmith.shaders.properties`](../../../scripts/shadesmith.shaders.properties); options aggregates it into
   final [`shaders/shaders.properties`](../../../shaders/shaders.properties). Inspect generated output, tile overlap,
   read/write stages, and history reset behavior. Other than the first generation from a fresh checkout, run Shadesmith
   only when [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) changes.

## Build a ZIP

```powershell
cd scripts
kotlin make-zip.main.kts [version] [--no-commit-hash]
```

The script runs Shadesmith first, then combines its processed [`shaders/`](../../../shaders/) with [
`changelogs/`](../../../changelogs/), [`licenses/`](../../../licenses/), [`shaders/lang/`](../../../shaders/lang/), [
`shaders/textures/`](../../../shaders/textures/), [`LICENSE`](../../../LICENSE), and [`README.md`](../../../README.md)
under `builds/`. Names include the short commit hash and, outside `main`/`dev`, a branch name with `/` converted to `-`.
`--no-commit-hash` removes the hash/branch suffix; version is optional. `builds/` is ignored.

## Before committing

- Plain GLSL changes should not create generated-file churn.
- Program, option, and tile changes must include tracked generated results; do not commit ignored fragments,
  `shadesmitth/`, or the AOT cache.
- Run:

  ```sh
  git diff --check
  git diff --cached --check
  ```
- Reload on the target Iris version and check affected settings, history resets, and representative scenes.
