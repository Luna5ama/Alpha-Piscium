# Development Quick Start

Language: English | [简体中文](../../sc/development/quick-start.md)

## Prerequisites

- A Minecraft/Iris test instance for final shader compilation and visual validation.
- `git`, a `kotlin` command capable of running `.main.kts`, and a JDK compatible with [
  `scripts/shadesmith.jar`](../../../scripts/shadesmith.jar).
- If the default Java is unsuitable, set `JAVA_PATH` in local `scripts/config.properties`;
  see [workflows and configuration](workflows.md).

On a fresh checkout, run Shadesmith once from [`scripts/`](../../../scripts/) before the first options generation:

```sh
cd scripts
./shadesmith.ps1
```

[`scripts/shadesmith.shaders.properties`](../../../scripts/shadesmith.shaders.properties) is an ignored generated
fragment; without it, options would omit image/atlas declarations from final [
`shaders/shaders.properties`](../../../shaders/shaders.properties).

## Shortest development loop

1. Use the [project structure](project-structure.md) and [pipeline overview](../modules/pipeline.md) to locate the entry
   pass and shared implementation.
2. Put new entry shaders under [`shaders/pass/`](../../../shaders/pass/). Existing modules retain a few direct compute
   entries under [`shaders/techniques/`](../../../shaders/techniques/); shared GLSL lives under [
   `shaders/techniques/`](../../../shaders/techniques/) and [`shaders/util/`](../../../shaders/util/).
3. Run a generator only when its owning source changed, using the table below.
4. Reload the pack in Iris and inspect both compilation output and the target scene.
5. Before committing, run:

   ```sh
   git diff --check
   git diff --cached --check
   ```

   Keep tracked generated outputs in the same commit as their source configuration.

## What to run after a change

### Ordinary `.glsl` implementation

No generator. Iris consumes the source through `#include`.

### Pass order, conditions, defines, or indirect dispatch in [
`scripts/programs.main.kts`](../../../scripts/programs.main.kts)

```sh
cd scripts
kotlin programs.main.kts
kotlin options.main.kts
```

Rebuilds wrappers/the ignored program fragment and aggregates final properties.

### Settings, profiles, screens, or localization in [`scripts/options.main.kts`](../../../scripts/options.main.kts)

```sh
cd scripts
kotlin options.main.kts
```

Also runs the program generator and rebuilds options, language files, and final [
`shaders/shaders.properties`](../../../shaders/shaders.properties).

### Hand-maintained textures, images, SSBOs, blending, or uniforms in [
`scripts/shaders.properties`](../../../scripts/shaders.properties)

```sh
cd scripts
kotlin options.main.kts
```

Re-aggregates the hand-maintained fragment into [`shaders/shaders.properties`](../../../shaders/shaders.properties).

### A transient, history, or persistent tile in [`shaders/shadesmith.json`](../../../shaders/shadesmith.json)

Run Shadesmith and options from [`scripts/`](../../../scripts/):

```sh
cd scripts
./shadesmith.ps1
kotlin options.main.kts
```

Updates Textile/image fragments and aggregates final properties.

### Test ZIP

```sh
cd scripts
kotlin make-zip.main.kts [version] [--no-commit-hash]
```

Creates a package under `builds/`; packaging runs Shadesmith first.

Do not run Shadesmith for ordinary GLSL edits; during normal development it is required only for [
`shaders/shadesmith.json`](../../../shaders/shadesmith.json), with fresh-checkout bootstrap as the other exception.

## Source-of-truth map

| Maintained source                                                                                                              | Generated or consumed location                                                                                                                                                                                                           |
|--------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`scripts/programs.main.kts`](../../../scripts/programs.main.kts)                                                              | `shaders/{setup,begin,shadowcomp,prepare,deferred,composite}*.csh`, ignored [`scripts/programs.shaders.properties`](../../../scripts/programs.shaders.properties)                                                                        |
| [`scripts/options.main.kts`](../../../scripts/options.main.kts), [`scripts/options.lib.kts`](../../../scripts/options.lib.kts) | [`shaders/base/Options.glsl`](../../../shaders/base/Options.glsl), [`shaders/base/TextOptions.glsl`](../../../shaders/base/TextOptions.glsl), `shaders/lang/*.lang`, [`shaders/shaders.properties`](../../../shaders/shaders.properties) |
| [`scripts/shaders.properties`](../../../scripts/shaders.properties)                                                            | Hand-maintained fragment merged into final [`shaders/shaders.properties`](../../../shaders/shaders.properties)                                                                                                                           |
| [`scripts/shadesmith.shaders.properties`](../../../scripts/shadesmith.shaders.properties)                                      | Ignored image/atlas fragment aggregated by options                                                                                                                                                                                       |
| [`shaders/shadesmith.json`](../../../shaders/shadesmith.json)                                                                  | Tracked [`shaders/base/Textile.glsl`](../../../shaders/base/Textile.glsl) bindings plus ignored fragment/output in [`scripts/shadesmitth/`](../../../scripts/shadesmitth/)                                                               |
| [`shaders/pass/**`](../../../shaders/pass/)                                                                                    | Standard location for new pass entry points                                                                                                                                                                                              |
| [`shaders/techniques/**`](../../../shaders/techniques/)                                                                        | Shared modules plus a few existing direct compute entries                                                                                                                                                                                |
| [`shaders/util/**`](../../../shaders/util/)                                                                                    | Cross-module utilities                                                                                                                                                                                                                   |

## Minimum validation

Run:

```sh
git diff --check
git diff --cached --check
```

The repository has no single automated entry point that replaces live Iris compilation and visual validation; also check
affected setting/profile branches, history resets, and representative scenes.
